"""Stage B - Dynamics V-Net GNN with per-resolution physics fusion and a deterministic delta head.

Forward order (design doc section 4.3; GNN -> fusion -> next GNN, no circular dependency):

    P0, P1, P2, P1u, P0u = guide(X_t, c_t, sigma=0)          # frozen weights, recomputed every call
    H0  = gnn_0(lift(X_t, c_t));                F0  = fusion_0(H0,  P0)
    H1  = gnn_1(down_01(F0));                   F1  = fusion_1(H1,  P1)
    H2  = gnn_2(down_12(F1));                   F2  = fusion_2(H2,  P2)
    H1u = gnn_3(merge(up_21(F2),  skip=H1));    F1u = fusion_3(H1u, P1u)
    H0u = gnn_4(merge(up_10(F1u), skip=H0));    F0u = fusion_4(H0u, P0u)
    delta = head(F0u);   X_bar = X_t + delta     (delta is already the change over the fixed Delta t)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import nn

from gpclimate.data.variables import VariableSpec
from gpclimate.geometry.graphs import GraphView
from gpclimate.models.lie_physical_gnn import GridLift, GuideConfig, LiePhysicalGNN, broadcast_batch
from gpclimate.nn.attention import PhysicsFusion
from gpclimate.nn.gauge_conv import GaugeStack, GaugeTransfer, Merge, transport_interpolate
from gpclimate.nn.physics_ops import PhysicsOperators
from gpclimate.nn.pigt import PhysicsIntegration
from gpclimate.nn.so2 import EqLinear, Field


@dataclass
class VNetConfig:
    cs: int = 64
    cv: int = 16
    depth: Tuple[int, int, int, int, int] = (2, 2, 2, 2, 2)  # gnn_0 .. gnn_4
    heads: int = 4
    ds: int = 16
    dv: int = 4
    n_basis: int = 8
    hidden: int = 64
    gate_init: float = 0.1
    use_guide: bool = True  # False = plain Dynamics GNN baseline (design doc section 10)
    fusion: str = "gated"  # "gated": F = H + beta Attn(H, P);  "pigt": physics-integrated graph transformer
    pigt_layers: int = 2
    pigt_ops: bool = True  # PIGT: fixed physics-operator tokens O_l as a third attention source
    # stage S8 (graph neural ODE dynamics, Poli et al. 2019 applied on top of the learned physics):
    #   "delta": X_bar = X_t + delta (design doc 4.4, one evaluation per lead);
    #   "ode":   X_bar = X_t + sum_k [ T_guide(X_k) * (lead / K) + f_phi(X_k, P(X_k), c) / K ]  (K fixed sub-steps)
    # ode_physics adds the frozen guide's 6 h tendency T_guide as the known drift, so the V-Net only learns the
    # correction and the zero-initialised head starts from the pure physics integration instead of persistence.
    integrator: str = "delta"  # "delta" | "ode"
    ode_steps: int = 4  # sub-steps per lead (4 x 6 h for a 24 h lead)
    ode_solver: str = "euler"  # "euler" | "heun" (Heun = 2 evaluations per sub-step)
    ode_physics: bool = False
    # Gradient checkpointing of every ODE sub-step while training: activations of a sub-step are recomputed in the
    # backward pass instead of kept, so memory no longer grows with ode_steps x rollout (the guided ODE V-Net at
    # batch 4, rollout 2 needed 14 GB and was OOM-killed on the 16 GB machine, stage17a 2026-11-08).  Same numbers,
    # about one extra forward pass per step.  No effect without gradients (evaluation, sampling).
    ode_checkpoint: bool = True
    # stage13: continuous neural-operator layer on the grid after the mesh V-Net ("" = off, "fft" = Fourier
    # neural operator, "sht" = spherical neural operator via torch_harmonics); residual on the delta, zero-initialised
    neural_op: str = ""
    no_hidden: int = 16
    no_layers: int = 2
    no_modes: Tuple[int, int] = (16, 32)
    # stage14: the V-Net also sees the observed change over the previous lead, X_t - X_{t-lead} (GraphCast-style
    # two-time-step input; in a rollout the previous forecast state plays the role of X_{t-lead}); the guide does not
    prev_state: bool = False
    # stage14b: initial-increment nudging — the first forecast step relaxes toward the model's own error over the
    # previous lead, e_t = X_t - M(X_{t-lead}), with a learned state-dependent coefficient (gpclimate/nn/nudging.py)
    nudge: bool = False
    nudge_hidden: int = 32
    # carry the added term into later rollout steps as the red-noise recursion q_k = rho q_{k-1} (Evensen 2009,
    # Ch. 12, Sect. 12.1.4); False = first step only (incremental-analysis-update style window of one lead)
    nudge_carry: bool = True
    nudge_alpha_init: float = 0.12  # initial uniform coefficient = prior rho (measure it with scripts/model_error_autocorr.py)
    # stage15a: column channels (e.g. sp, z1000, z850, t500, z50, r500, r850, tcwv, u/v1000) as V-Net *inputs only* at
    # the initial time; later rollout steps get zeros and an availability flag 0 (their future values are unknown)
    col_s: int = 0
    col_v: int = 0
    # stage15b: intermediate-time supervision (WeatherODE-style loss on intermediate states, without extra solver
    # evaluations) — an auxiliary head on the shared decoder embedding predicts the changes to X_{t + j 6 h}, j =
    # 1..inter_times (3 = 6/12/18 h inside a 24 h lead).  Used only by the training loss; forecasts are unchanged.
    inter_times: int = 0
    # stage17 (dynamic guide, integrator "ode"): the guide and the fixed operators are re-evaluated on every 6 h path
    # state.  ode_path_forcing: the solar forcing channels follow the sub-step time (toa_t / toa_t1 rotated by the
    # sub-step offset in longitude, exact for the insolation proxy up to the declination drift of < 0.1 deg per 6 h);
    # ode_drift_gain: learnable per-variable gain on the guide's 6 h tendency drift (zero-initialised, needs
    # ode_physics), so training decides how much of the guide's own tendency enters the dynamics.  Path states
    # X_{t + k 6 h} are kept in ``last_path`` for the path-supervision loss (train_eval --inter-times with ode).
    ode_path_forcing: bool = False
    ode_drift_gain: bool = False
    # stage17c: what the guide drift is — "tend": the guide's 6 h tendency head (stage8b / stage17b); "transport": the
    # Lie transport guide G(X) - X = [semi-Lagrangian transport by the guide's flow] - X + source (GuideConfig.flow_head)
    ode_drift: str = "tend"
    # stage18 graph tendency core (PG-Flow tendency core on the grid's face graph, gpclimate/nn/graph_core.py): every
    # 6 h sub-step is  X_{k+1} = SL_{beta V}(X_k) - div_G J_theta + S_theta  with V the guide's flow re-evaluated on X_k
    # (stage17c flow head), beta [cells x level groups] the share of that flow the dynamics use (V-Net head, 0 at
    # start), J_theta an antisymmetric face flux read from the V-Net embedding (conservative by construction, scalars),
    # and S_theta the V-Net's own output as the source.  The guide's core terms (its transport change and its source)
    # enter the V-Net every sub-step: added to the grid embedding and, with core_tokens, as a fourth PIGT token
    # source C next to H / P / O.  Needs integrator "ode", PIGT fusion and a guide with the flow head.
    tendency_core: bool = False
    core_tokens: bool = True
    # stage18d Laplace block in the core (PhyMPGN, ICLR 2025): + one explicit diffusion step of the scalar channels on
    # the face graph, finite-volume Laplace-Beltrami with the covering (cotangent) weights, diffusivity K >= 0 per cell
    # and channel from the V-Net embedding, K = K_ref softplus(gain * head), the head starting uniform at raw = -5
    # (K ~ 700 m^2/s, i.e. ~ off); the gain lets Adam move K by orders of magnitude within a 1000-step fine-tuning.
    # Face numbers are capped for stability.  Needs tendency_core.
    core_laplace: bool = False
    core_k_ref: float = 1.0e5
    core_k_gain: float = 8.0
    # stage18e edge-preserving diffusion in the Laplace block (Kovac & Smith 2009, arXiv 0911.1928, "Regression on a
    # graph": an L1 penalty on the edge differences keeps jumps and fronts while smoothing the rest).  The face
    # diffusivity is multiplied by eps / sqrt((q_j - q_i)^2 + eps^2): linear diffusion across small differences, a
    # bounded flux across fronts (the eps-regularised gradient flow of sum_f |q_j - q_i|, total-variation flow).  eps
    # per scalar variable = the median neighbour difference of the training states (preprocessing, buffer ``tv_eps0``)
    # times a learned factor exp(tv_log_gain) (init 1).  Conservation and the stability cap are unchanged (factor <= 1).
    # Needs core_laplace.
    core_tv: bool = False
    # stage18b: the previous-state input follows the path — sub-step k of an ODE lead gets the state one lead before
    # *its own* time, X_{t + k 6 h - lead} (observed past states in the first lead, the previous lead's path later),
    # instead of X_{t - lead} for all sub-steps.  ``prev`` is then a list of ode_steps states.
    prev_path: bool = False


class DynamicsVNet(nn.Module):
    def __init__(self, spec: VariableSpec, cfg: VNetConfig = VNetConfig(), guide_dims: Tuple[int, int] = (48, 12)):
        super().__init__()
        self.spec, self.cfg = spec, cfg
        cs, cv, nb, hd = cfg.cs, cfg.cv, cfg.n_basis, cfg.hidden
        n_prev_s, n_prev_v = (spec.n_s, spec.n_v) if cfg.prev_state else (0, 0)
        self.use_col = cfg.col_s + cfg.col_v > 0
        # stage18: the guide-free pair has no guide and hence no core (it stays the plain ODE V-Net)
        self.core_on = bool(cfg.tendency_core and cfg.use_guide)
        self.core_dims = (2 * spec.n_s, 2 * spec.n_v) if self.core_on else None
        if self.core_on and not (cfg.fusion == "pigt" and cfg.integrator == "ode"):
            raise ValueError("tendency_core needs PIGT fusion and the ODE integrator")
        n_col_s, n_col_v = (cfg.col_s + 1, cfg.col_v) if self.use_col else (0, 0)  # + availability flag
        self.lift = GridLift(spec.n_s + spec.n_cond_s + n_prev_s + n_col_s, spec.n_v + spec.n_cond_v + n_prev_v + n_col_v,
                             cs, cv, nb, hd)
        self.gnn = nn.ModuleList([GaugeStack(cs, cv, d, nb, hd) for d in cfg.depth])
        self.phys_ops = None
        if cfg.use_guide and cfg.fusion == "pigt":
            pcs, pcv = guide_dims
            ops_dims = None
            if cfg.pigt_ops:
                self.phys_ops = PhysicsOperators(spec.n_s, spec.n_v, spec.n_cond_s)
                ops_dims = self.phys_ops.out_dims
            core_tok = self.core_dims if (self.core_on and cfg.core_tokens) else None
            self.fusion = nn.ModuleList([
                PhysicsIntegration(cs, cv, pcs, pcv, ops_dims, cfg.pigt_layers, cfg.heads, cfg.ds, cfg.dv, nb,
                                   core_dims=core_tok)
                for _ in range(5)])
        elif cfg.use_guide:
            pcs, pcv = guide_dims
            self.fusion = nn.ModuleList([
                PhysicsFusion(cs, cv, pcs, pcv, cfg.heads, cfg.ds, cfg.dv, nb, cfg.gate_init) for _ in range(5)])
        else:
            self.fusion = None
        self.down01 = GaugeTransfer(cs, cv, cs, cv, nb, hd)
        self.down12 = GaugeTransfer(cs, cv, cs, cv, nb, hd)
        self.up21 = GaugeTransfer(cs, cv, cs, cv, nb, hd)
        self.up10 = GaugeTransfer(cs, cv, cs, cv, nb, hd)
        self.merge1 = Merge(cs, cv, cs, cv, cs, cv)
        self.merge0 = Merge(cs, cv, cs, cv, cs, cv)
        # graph-to-grid head; receiver also sees the grid embedding of X_t (local detail skip)
        self.head_dec = GaugeTransfer(cs, cv, cs, cv, nb, hd, extra_cs=cs, extra_cv=cv)
        self.head_out = EqLinear(cs, cv, spec.n_s, spec.n_v)
        self.head_inter = (EqLinear(cs, cv, spec.n_s * cfg.inter_times, spec.n_v * cfg.inter_times)
                           if cfg.inter_times else None)
        self.last_inter: Optional[Field] = None  # stage15b: intermediate-time output of the last forward (delta units)
        self.cont_op = None
        if cfg.neural_op:
            from gpclimate.nn.neural_operator import ContinuousOperatorBlock
            self.cont_op = ContinuousOperatorBlock(spec.n_s, spec.n_v, spec.n_cond_s, spec.n_cond_v, cfg.no_hidden,
                                                   cfg.no_layers, tuple(cfg.no_modes), cfg.neural_op)
        self.drift_gain_s = self.drift_gain_v = None
        if cfg.ode_drift_gain and cfg.use_guide:  # stage17b: real gains keep the vector drift gauge-equivariant
            self.drift_gain_s = nn.Parameter(torch.zeros(spec.n_s))
            self.drift_gain_v = nn.Parameter(torch.zeros(spec.n_v))
        self.core_embed = self.beta_head = self.flux_head = self.laplace_head = None
        if cfg.core_laplace and not cfg.tendency_core:
            raise ValueError("core_laplace (stage18d) is part of the graph tendency core: set tendency_core")
        self.last_emb: Optional[Field] = None
        if self.core_on:  # stage18: all new parts start at zero (a trained V-Net keeps its function)
            from gpclimate.nn.graph_core import FaceFluxHead
            self.core_embed = EqLinear(*self.core_dims, cs, cv)
            self.beta_head = EqLinear(cs, cv, 3, 0)
            self.flux_head = FaceFluxHead(cs, cv, spec.n_s, hd)
            with torch.no_grad():
                for lin in (self.core_embed, self.beta_head):
                    lin.lin_s.weight.zero_()
                    lin.lin_s.bias.zero_()
                    if lin.w_re is not None:
                        lin.w_re.zero_()
                        lin.w_im.zero_()
            if cfg.core_laplace:  # stage18d: diffusivity head, uniform and ~ 0 at the start
                self.laplace_head = EqLinear(cs, cv, spec.n_s, 0)
                with torch.no_grad():
                    self.laplace_head.lin_s.weight.zero_()
                    self.laplace_head.lin_s.bias.fill_(-5.0 / cfg.core_k_gain)
        self.tv_log_gain = None
        if cfg.core_tv:  # stage18e: edge scale per scalar variable (preprocessing value x learned factor, init 1)
            if not cfg.core_laplace:
                raise ValueError("core_tv (stage18e) is the edge-preserving form of the Laplace block: set core_laplace")
            if self.core_on:
                self.register_buffer("tv_eps0", torch.ones(spec.n_s))
                self.tv_log_gain = nn.Parameter(torch.zeros(spec.n_s))
        self.nudge_block = None
        if cfg.nudge:
            from gpclimate.nn.nudging import InitialNudging
            self.nudge_block = InitialNudging(cs, cv, spec.n_s, spec.n_v, cfg.nudge_hidden, cfg.nudge_alpha_init)
        with torch.no_grad():  # start from persistence (delta = 0)
            for head in (self.head_out, self.head_inter):
                if head is None:
                    continue
                head.lin_s.weight.zero_()
                head.lin_s.bias.zero_()
                if head.w_re is not None:
                    head.w_re.zero_()
                    head.w_im.zero_()

    def _fuse(self, k: int, h: Field, physics: Optional[Dict[str, Field]], key: str, edges,
              ops: Optional[Field] = None, core: Optional[Field] = None) -> Field:
        if self.fusion is None or physics is None:
            return h
        if self.cfg.fusion == "pigt":
            return self.fusion[k](h, physics[key], ops, edges, core)
        return self.fusion[k](h, physics[key], edges)

    def core_tokens(self, core: Optional[Field], g: GraphView) -> List[Optional[Field]]:
        """stage18: the guide's core terms pooled (with transport) to every mesh level, or Nones."""
        if core is None or not (self.core_on and self.cfg.core_tokens):
            return [None] * len(g.mesh_edges)
        out, raw = [], transport_interpolate(core, g.grid2mesh)
        for lvl in range(len(g.mesh_edges)):
            if lvl > 0:
                raw = transport_interpolate(raw, g.down[lvl - 1])
            out.append(raw)
        return out

    def core_beta(self, emb: Field) -> torch.Tensor:
        """stage18: share of the guide's flow per cell and level group [B, N, 3] (0 at the start, clipped)."""
        return self.beta_head(emb).s.clamp(-0.5, 2.0)

    def core_diffusivity(self, emb: Field) -> torch.Tensor:
        """stage18d: diffusivity K [B, N, n_s] (m^2/s, >= 0) of the scalar channels per cell."""
        return self.cfg.core_k_ref * nn.functional.softplus(self.cfg.core_k_gain * self.laplace_head(emb).s)

    def core_tv_eps(self) -> Optional[torch.Tensor]:
        """stage18e: edge scale eps [n_s] (normalised units) of the edge-preserving diffusion, None without core_tv."""
        if self.tv_log_gain is None:
            return None
        return self.tv_eps0 * torch.exp(self.tv_log_gain)

    def core_flux(self, emb: Field, faces, gauge: torch.Tensor) -> torch.Tensor:
        """stage18: -div_G J_theta of the scalar channels [B, N, n_s] in units of the face graph's mean edge length
        (an O(1) flux changes a cell by O(1) of the delta scale)."""
        J = self.flux_head(emb, faces, gauge)
        return -faces.divergence(J) * faces.mean_length

    def operator_tokens(self, state: Field, cond: Field, g: GraphView) -> List[Optional[Field]]:
        """Fixed physics-operator terms O_l on every mesh level (state pooled with transport), or Nones."""
        if self.phys_ops is None:
            return [None] * len(g.mesh_edges)
        B = state.s.shape[0]
        cond = broadcast_batch(cond, B)
        n_s = self.spec.n_s
        raw = transport_interpolate(Field(torch.cat([state.s, cond.s], -1), state.v), g.grid2mesh)
        out = []
        for lvl in range(len(g.mesh_edges)):
            if lvl > 0:
                raw = transport_interpolate(raw, g.down[lvl - 1])
            out.append(self.phys_ops(raw.s[..., :n_s], raw.v, raw.s[..., n_s:], g.mesh_sinlat[lvl], g.mesh_edges[lvl]))
        return out

    def column_input(self, col: Optional[Field], B: int, G: int, like: Field) -> Field:
        """stage15a: column channels + availability flag; zeros and flag 0 when ``col`` is None."""
        cfg = self.cfg
        if col is None:
            s = like.s.new_zeros(B, G, cfg.col_s + 1)
            v = like.v.new_zeros(B, G, cfg.col_v)
            return Field(s, v)
        col = broadcast_batch(col, B)
        return Field(torch.cat([col.s, col.s.new_ones(B, G, 1)], -1), col.v)

    def forward(self, state: Field, cond: Field, physics: Optional[Dict[str, Field]], g: GraphView,
                prev: Optional[Field] = None, nudge: Optional[Field] = None, col: Optional[Field] = None,
                core: Optional[Field] = None) -> Field:
        B = state.s.shape[0]
        cond = broadcast_batch(cond, B)
        if isinstance(prev, list):  # stage18b path list: a single forward uses the lead's first entry
            prev = prev[0]
        xs, xv = [state.s, cond.s], [state.v, cond.v]
        if self.cfg.prev_state:  # stage14: observed change over the previous lead (zero when no previous state)
            prev = state if prev is None else prev
            xs.append(state.s - prev.s)
            xv.append(state.v - prev.v)
        x = Field(torch.cat(xs, -1), torch.cat(xv, -1))
        x_in = x
        if self.use_col:  # stage15a: lift input only (the stage13 operator block keeps its own input layout)
            c = self.column_input(col, B, state.s.shape[1], state)
            x_in = Field(torch.cat([x.s, c.s], -1), torch.cat([x.v, c.v], -1))
        if self.core_embed is not None and core is None and physics is not None:
            raise ValueError("the graph tendency core V-Net needs the guide's core terms (``core``)")
        extra = self.core_embed(broadcast_batch(core, B)) if (self.core_embed is not None and core is not None) else None
        grid_emb, m0 = self.lift(x_in, g, extra=extra)
        O = self.operator_tokens(state, cond, g) if (physics is not None and self.cfg.fusion == "pigt") else [None] * 3
        C = self.core_tokens(core, g)
        H0 = self.gnn[0](m0, g.mesh_edges[0])
        F0 = self._fuse(0, H0, physics, "P0", g.attn_edges[0], O[0], C[0])
        H1 = self.gnn[1](self.down01(F0, g.down[0]), g.mesh_edges[1])
        F1 = self._fuse(1, H1, physics, "P1", g.attn_edges[1], O[1], C[1])
        H2 = self.gnn[2](self.down12(F1, g.down[1]), g.mesh_edges[2])
        F2 = self._fuse(2, H2, physics, "P2", g.attn_edges[2], O[2], C[2])
        H1u = self.gnn[3](self.merge1(self.up21(F2, g.up[1]), H1), g.mesh_edges[1])
        F1u = self._fuse(3, H1u, physics, "P1u", g.attn_edges[1], O[1], C[1])
        H0u = self.gnn[4](self.merge0(self.up10(F1u, g.up[0]), H0), g.mesh_edges[0])
        F0u = self._fuse(4, H0u, physics, "P0u", g.attn_edges[0], O[0], C[0])
        emb = self.head_dec(F0u, g.mesh2grid, extra=grid_emb)
        self.last_emb = emb if self.core_on else None  # stage18: read by the core heads
        out = self.head_out(emb)
        if self.head_inter is not None:  # stage15b: intermediate-time changes from the same embedding
            self.last_inter = self.head_inter(emb)
        if self.cont_op is not None:  # stage13: Lie guide -> discrete mesh GNN -> continuous operator on the grid
            H, W = g.extras["grid_hw"]
            out = out + self.cont_op(x, out, H, W, g.extras["grid_lat"], g.extras.get("grid_gauge"))
        if self.nudge_block is not None:  # stage14b: relax toward the (carried) initial increment
            if nudge is not None:
                out = out + self.nudge_block(emb, nudge)
            else:
                self.nudge_block.last_term = None  # nothing added -> nothing to carry
        return out

    def attention_mass(self) -> Optional[Dict[str, List[Optional[List[float]]]]]:
        """PIGT only: attention mass of the dynamics stream per level (P0,P1,P2,P1u,P0u), layer and source."""
        if self.fusion is None or self.cfg.fusion != "pigt":
            return None
        return {k: f.source_mass for k, f in zip(("P0", "P1", "P2", "P1u", "P0u"), self.fusion)}


class LieGuidedForecaster(nn.Module):
    """Physics guide (Stage A) + Dynamics V-Net (Stage B) -> deterministic forecast X_bar."""

    def __init__(self, spec: VariableSpec, guide_cfg: GuideConfig = GuideConfig(), vnet_cfg: VNetConfig = VNetConfig()):
        super().__init__()
        self.spec = spec
        self.guide = LiePhysicalGNN(spec, guide_cfg) if vnet_cfg.use_guide else None
        self.vnet = DynamicsVNet(spec, vnet_cfg, guide_dims=(guide_cfg.cs, guide_cfg.cv))
        # per-variable scale of the normalised one-step change (set from training data)
        self.register_buffer("delta_scale_s", torch.ones(spec.n_s))
        self.register_buffer("delta_scale_v", torch.ones(spec.n_v))
        # scale of the normalised one-data-step (6 h) tendency and data steps per lead (stage S8 physics drift)
        self.register_buffer("tend_scale_s", torch.ones(spec.n_s))
        self.register_buffer("tend_scale_v", torch.ones(spec.n_v))
        self.lead_steps = 4
        self.last_path: Optional[List[Field]] = None  # stage17: path states of the last ODE forward (K - 1 of them)
        names = list(spec.static_scalars) + list(spec.forcing_scalars)
        self._toa_idx = [names.index(n) for n in ("toa_t", "toa_t1") if n in names]
        self._faces = None  # stage18: finite-volume face graph of the grid (lazy)
        self.last_core: Optional[Dict[str, object]] = None  # stage18: term sizes of the last core sub-step
        if self.vnet.core_on and not guide_cfg.flow_head:
            raise ValueError("tendency_core needs a guide with the flow head (GuideConfig.flow_head)")
        if vnet_cfg.tendency_core and vnet_cfg.ode_physics:
            raise ValueError("tendency_core replaces the guide drift: set ode_physics = 0")

    def freeze_guide(self) -> None:
        """Freeze guide *weights*; gradients w.r.t. its inputs still flow (needed for rollout losses)."""
        if self.guide is not None:
            for p in self.guide.parameters():
                p.requires_grad_(False)
            self.guide.eval()

    def physics(self, state: Field, cond: Field, g: GraphView) -> Optional[Dict[str, Field]]:
        if self.guide is None:
            return None
        sigma = torch.zeros(state.s.shape[0], device=state.s.device)
        return self.guide(state, cond, sigma, g)

    def forward(self, state: Field, cond: Field, g: GraphView, prev: Optional[Field] = None,
                nudge: Optional[Field] = None, col: Optional[Field] = None) -> Tuple[Field, Field]:
        """Deterministic forecast and its change in delta-scale units ``d = (X_bar - X_t) / delta_scale``.
        ``prev`` (stage14, ``VNetConfig.prev_state``) is the state one lead earlier; None = zero observed change.
        ``nudge`` (stage14b, ``VNetConfig.nudge``) is the initial increment from :meth:`increment`; None = no
        relaxation (rollout steps after the first).  ``col`` (stage15a, ``VNetConfig.col_s/col_v``) are the column
        input channels at the initial time; None = not available (rollout steps after the first)."""
        if self.vnet.cfg.integrator != "ode":
            prev = self._prev_at(prev, 0)
            P = self.physics(state, cond, g)
            d = self.vnet(state, cond, P, g, prev=prev, nudge=nudge, col=col)
            forecast = Field(state.s + d.s * self.delta_scale_s, state.v + d.v * self.delta_scale_v)
            return forecast, d
        if nudge is not None:
            raise ValueError("nudging (stage14b) is not supported with the ODE integrator")
        forecast = self.integrate(state, cond, g, prev=prev, col=col)
        d = Field((forecast.s - state.s) / self.delta_scale_s, (forecast.v - state.v) / self.delta_scale_v)
        return forecast, d

    def inter_forecasts(self, state: Field) -> List[Field]:
        """stage15b: intermediate-time states X_{t + j 6 h}, j = 1..inter_times, predicted by the intermediate-time head in
        the last forward call that started from ``state`` (same units as ``state``; delta-scale changes)."""
        if self.vnet.cfg.integrator == "ode":  # stage17: the integration path itself (X_{t + k 6 h}, k = 1..K-1)
            assert self.last_path is not None, "no ODE forward has been run yet"
            return list(self.last_path)
        cfg, d = self.vnet.cfg, self.vnet.last_inter
        assert cfg.inter_times > 0 and d is not None, "the V-Net has no intermediate-time head or has not been called"
        n_s, n_v = self.spec.n_s, self.spec.n_v
        ds = d.s.unflatten(-1, (cfg.inter_times, n_s))
        dv = d.v.unflatten(-1, (cfg.inter_times, n_v))
        return [Field(state.s + ds[..., j, :] * self.delta_scale_s, state.v + dv[..., j, :] * self.delta_scale_v)
                for j in range(cfg.inter_times)]

    def increment(self, state: Field, prev_state: Field, prev_cond: Field, g: GraphView,
                  prev_prev: Optional[Field] = None) -> Field:
        """stage14b nudging increment e_t = X_t - M(X_{t-lead}) in delta-scale units: the model's own error over the
        previous lead, from a pre-forecast started at ``prev_state`` (no gradient, itself un-nudged).  ``prev_prev``
        is the state two leads back when the V-Net uses the previous-state input (stage14a)."""
        with torch.no_grad():
            fg, _ = self(prev_state, prev_cond, g, prev=prev_prev)
        return Field((state.s - fg.s) / self.delta_scale_s, (state.v - fg.v) / self.delta_scale_v)

    def carried_nudge(self) -> Optional[Field]:
        """stage14b: the model-error estimate to carry into the next rollout step = the term the last call added
        (red-noise recursion q_k = rho q_{k-1}, Evensen 2009 Sect. 12.1.4); None when nudging is off, not carried
        (``nudge_carry=False``), or the last call added nothing."""
        nb = self.vnet.nudge_block
        if nb is None or not self.vnet.cfg.nudge_carry:
            return None
        return nb.last_term

    # ---------------------------------------------------------------- stage S8: graph neural ODE dynamics
    def physics_drift(self, x: Field, cond: Field, g: GraphView, P: Optional[Dict[str, Field]],
                      use_gain: bool = True) -> Field:
        """Change of the normalised state over one sub-step from the frozen guide's 6 h tendency (times the learned
        stage17b gain unless ``use_gain`` is False, as in the pure-physics diagnostic)."""
        assert self.guide is not None, "ode_physics needs the physics guide"
        if self.vnet.cfg.ode_drift == "transport" or self.vnet.core_on:  # stage18: the guide's own core
            return self.transport_drift(x, g, P, use_gain)
        T = self.guide.physics_tendency(P, x, g)
        assert T is not None, "ode_physics needs a guide with a tendency head or a closure head"
        steps = self.lead_steps / self.vnet.cfg.ode_steps  # data steps (6 h) per sub-step
        gs, gv = self.vnet.drift_gain_s, self.vnet.drift_gain_v
        if gs is not None and use_gain:  # stage17b: learned share of the guide's own tendency
            return Field(T.s * self.tend_scale_s * steps * gs, T.v * self.tend_scale_v * steps * gv)
        return Field(T.s * self.tend_scale_s * steps, T.v * self.tend_scale_v * steps)

    def transport_drift(self, x: Field, g: GraphView, P: Dict[str, Field], use_gain: bool = True,
                        parts: bool = False):
        """stage17c: change of the normalised state over one sub-step from the Lie transport guide,
        G(X) - X = [T_u(X) - X] + S_theta(X) * tend_scale * steps (times the learned gain unless ``use_gain`` is False).
        ``parts=True`` returns the (transport, source) pieces separately, without the gain (diagnostics)."""
        steps = self.lead_steps / self.vnet.cfg.ode_steps
        if self.guide.flow_out is None:
            raise ValueError("ode_drift='transport' needs a guide with the flow head (GuideConfig.flow_head)")
        xt = self.guide.transport(P, x, g, steps)
        S = self.guide.physics_tendency(P, x, g)
        tr = Field(xt.s - x.s, xt.v - x.v)
        src = Field(S.s * self.tend_scale_s * steps, S.v * self.tend_scale_v * steps)
        if parts:
            return tr, src
        ds, dv = tr.s + src.s, tr.v + src.v
        gs, gv = self.vnet.drift_gain_s, self.vnet.drift_gain_v
        if gs is not None and use_gain:
            ds, dv = ds * gs, dv * gv
        return Field(ds, dv)

    def sub_cond(self, cond: Field, g: GraphView, k: float) -> Field:
        """stage17: conditions at sub-step ``k`` of the lead (``k`` may be fractional for Heun's end point).  With
        ``ode_path_forcing`` the solar channels are rotated westward by the elapsed time (15 deg per hour), which is
        how the insolation proxy moves; static fields and the Coriolis parameter do not change."""
        cfg = self.vnet.cfg
        if not cfg.ode_path_forcing or k == 0 or not self._toa_idx:
            return cond
        H, W = g.extras["grid_hw"]
        hours = 6.0 * self.lead_steps * k / cfg.ode_steps
        shift = W * hours / 24.0
        if abs(shift - round(shift)) > 1e-6:
            raise ValueError(f"sub-step offset {hours} h is not a whole number of longitude columns (W={W})")
        s = cond.s.clone()
        for i in self._toa_idx:
            s[..., i] = torch.roll(cond.s[..., i].unflatten(-1, (H, W)), -int(round(shift)), dims=-1).flatten(-2)
        return Field(s, cond.v)

    def rate(self, x: Field, cond: Field, g: GraphView, p_map: Optional[Callable] = None,
             prev: Optional[Field] = None, col: Optional[Field] = None) -> Field:
        """Increment of one sub-step: [physics drift] + learned correction (V-Net output in lead-change units / K).
        Physics features are recomputed from the current sub-step state, so the guide and the learned dynamics stay
        coupled along the whole integration path.  ``p_map`` (diagnostics only) transforms the physics features
        of every sub-step, e.g. zeroing or shuffling them to measure the guide dependence of the ODE dynamics."""
        cfg = self.vnet.cfg
        if self.vnet.core_on:
            return self.core_rate(x, cond, g, p_map, prev=prev, col=col)
        P = self.physics(x, cond, g)
        if p_map is not None and P is not None:
            P = p_map(P)
        d = self.vnet(x, cond, P, g, prev=prev, col=col)
        h = 1.0 / cfg.ode_steps
        f = Field(d.s * self.delta_scale_s * h, d.v * self.delta_scale_v * h)
        if cfg.ode_physics and self.guide is not None:  # the guide-free pair has no guide and hence no drift
            drift = self.physics_drift(x, cond, g, P)
            f = Field(f.s + drift.s, f.v + drift.v)
        return f

    def faces(self, g: GraphView):
        """stage18: the finite-volume face graph of the grid (built once per grid)."""
        H, W = g.extras["grid_hw"]
        if self._faces is None or (self._faces.H, self._faces.W) != (H, W):
            from gpclimate.nn.graph_core import GridFaceGraph
            self._faces = GridFaceGraph(g.extras["grid_lat"], g.extras["grid_lon"])
        return self._faces.to(self.delta_scale_s.device)

    def core_terms(self, x: Field, P: Dict[str, Field], g: GraphView) -> Tuple[torch.Tensor, Field, Field]:
        """stage18: (guide flow [B, N, 3] m/s, guide transport change SL_V(X) - X, guide source * dt) over one
        sub-step, all from the (re-evaluated) guide features ``P`` of the current state."""
        steps = self.lead_steps / self.vnet.cfg.ode_steps
        flow, S = self.guide.core_parts(P, x, g)
        xt = self.guide.transport(None, x, g, steps, flow=flow)
        tr = Field(xt.s - x.s, xt.v - x.v)
        src = Field(S.s * self.tend_scale_s * steps, S.v * self.tend_scale_v * steps)
        return flow, tr, src

    def core_rate(self, x: Field, cond: Field, g: GraphView, p_map: Optional[Callable] = None,
                  prev: Optional[Field] = None, col: Optional[Field] = None) -> Field:
        """stage18 graph tendency core, one 6 h sub-step:
            X_{k+1} - X_k = [SL_{beta V}(X_k) - X_k] - div_G J_theta + S_theta
        V = guide flow on X_k, beta / J_theta from the V-Net embedding, S_theta = the V-Net output (lead-change units
        / K).  The guide's own terms (transport change, source) are the V-Net's core input.  ``p_map`` (diagnostics)
        maps the guide features before anything is read from them (flow, source, tokens)."""
        cfg = self.vnet.cfg
        P = self.physics(x, cond, g)
        if p_map is not None:
            P = p_map(P)
        flow, tr, src = self.core_terms(x, P, g)
        core = Field(torch.cat([tr.s, src.s], -1), torch.cat([tr.v, src.v], -1))
        d = self.vnet(x, cond, P, g, prev=prev, col=col, core=core)
        emb = self.vnet.last_emb
        beta = self.vnet.core_beta(emb)
        steps = self.lead_steps / cfg.ode_steps
        xb = self.guide.transport(None, x, g, steps, flow=flow, beta=beta)
        faces = self.faces(g)
        gauge = g.extras["grid_gauge"].to(x.s.device)
        C = self.vnet.core_flux(emb, faces, gauge) * self.delta_scale_s / cfg.ode_steps
        h = 1.0 / cfg.ode_steps
        S_s, S_v = d.s * self.delta_scale_s * h, d.v * self.delta_scale_v * h
        D, K, eps = None, None, None
        if self.vnet.laplace_head is not None:  # stage18d: Laplace block, diffusion of the transported scalars
            K = self.vnet.core_diffusivity(emb)
            eps = self.vnet.core_tv_eps()  # stage18e: edge-preserving factor on the face diffusivity
            D = faces.diffusion(xb.s, faces.face_mean(K), 6 * 3600.0 * steps, edge_eps=eps)
        with torch.no_grad():  # share of each core term (rms over the batch, normalised units) for the logs
            rms = lambda t: float(t.abs().pow(2).mean().sqrt())  # noqa: E731
            self.last_core = {"beta": [float(b) for b in beta.mean((0, 1))],
                              "beta_sd": [float(b) for b in beta.std((0, 1))],
                              "transport": rms(torch.cat([(xb.s - x.s).flatten(), (xb.v - x.v).abs().flatten()])),
                              "flux": rms(C), "source": rms(torch.cat([S_s.flatten(), S_v.abs().flatten()])),
                              "guide_transport": rms(torch.cat([tr.s.flatten(), tr.v.abs().flatten()]))}
            if D is not None:
                self.last_core.update(diffusion=rms(D), K=[float(k) for k in K.mean((0, 1))])
            if eps is not None:  # edge scale, learned factor and the mean kept share of the diffusivity per variable
                dq = xb.s.index_select(1, faces.dst) - xb.s.index_select(1, faces.src)
                self.last_core.update(tv_eps=[float(e) for e in eps],
                                      tv_gain=[float(v) for v in torch.exp(self.vnet.tv_log_gain)],
                                      tv_keep=[float(v) for v in faces.edge_factor(dq, eps).mean((0, 1))])
        ds = xb.s - x.s + C + S_s + (D if D is not None else 0.0)
        return Field(ds, xb.v - x.v + S_v)

    def integrate(self, state: Field, cond: Field, g: GraphView, p_map: Optional[Callable] = None,
                  prev: Optional[Field] = None, col: Optional[Field] = None) -> Field:
        """Fixed-step integration over the lead (explicit Euler or Heun).  Conditions are held fixed within the lead
        unless ``ode_path_forcing`` (stage17).  ``prev`` (stage14) is the observed change before the lead and is given
        to every sub-step; the column inputs ``col`` (stage15a) exist at the initial time only, i.e. sub-step 0.
        The intermediate path states are kept in ``last_path`` (stage17 path supervision)."""
        cfg = self.vnet.cfg
        x = state
        path = []
        ckpt = cfg.ode_checkpoint and torch.is_grad_enabled()
        for k in range(cfg.ode_steps):
            if ckpt:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(self._sub_step, x, cond, g, k, p_map, prev, col, use_reentrant=False)
            else:
                x = self._sub_step(x, cond, g, k, p_map, prev, col)
            path.append(x)
        self.last_path = path[:-1]
        return x

    def _sub_step(self, x: Field, cond: Field, g: GraphView, k: int, p_map: Optional[Callable], prev, col) -> Field:
        """One sub-step ``k`` of ``integrate`` (explicit Euler or Heun)."""
        cfg = self.vnet.cfg
        ck, ci = self.sub_cond(cond, g, k), (col if k == 0 else None)
        f1 = self.rate(x, ck, g, p_map, prev=self._prev_at(prev, k), col=ci)
        if cfg.ode_solver == "heun":
            x1 = Field(x.s + f1.s, x.v + f1.v)
            f2 = self.rate(x1, self.sub_cond(cond, g, k + 1), g, p_map, prev=self._prev_at(prev, k + 1))
            return Field(x.s + 0.5 * (f1.s + f2.s), x.v + 0.5 * (f1.v + f2.v))
        return Field(x.s + f1.s, x.v + f1.v)

    def physics_only(self, state: Field, cond: Field, g: GraphView) -> Field:
        """Forecast from the physics drift alone (V-Net switched off): diagnostic of the guide's tendency."""
        cfg = self.vnet.cfg
        x = state
        for k in range(cfg.ode_steps):
            ck = self.sub_cond(cond, g, k)
            drift = self.physics_drift(x, ck, g, self.physics(x, ck, g), use_gain=False)
            x = Field(x.s + drift.s, x.v + drift.v)
        return x

    def rollout(self, state: Field, conds: List[Field], g: GraphView, prev: Optional[Field] = None,
                nudge: Optional[Field] = None, col: Optional[Field] = None) -> List[Field]:
        """Autoregressive deterministic rollout; physics features are recomputed from each new state, the previous
        input state serves as ``prev`` of the next step (stage14), the nudging increment applies to the first
        step and is then carried as the added term (stage14b, ``nudge_carry``) or dropped, and the column inputs
        (stage15a) exist at the initial time only."""
        out = []
        for c in conds:
            new, _ = self(state, c, g, prev=prev, nudge=nudge, col=col)
            prev, state, nudge, col = self.next_prev(state), new, self.carried_nudge(), None
            out.append(state)
        return out

    def _prev_at(self, prev, k: int) -> Optional[Field]:
        """stage18b: the previous-state input of sub-step ``k``: the k-th entry of a path list when the V-Net follows
        the path (``prev_path``), the lead's first entry otherwise; a single state is used for every sub-step."""
        if not isinstance(prev, list):
            return prev
        return prev[min(k, len(prev) - 1)] if self.vnet.cfg.prev_path else prev[0]

    def next_prev(self, start: Field):
        """The ``prev`` of the next lead after a forecast started at ``start``: ``start`` itself, or with
        ``prev_path`` (stage18b, ODE) the path of the lead just integrated, [start, X_{+6 h}, ..., X_{+lead - 6 h}]."""
        cfg = self.vnet.cfg
        if cfg.prev_path and cfg.integrator == "ode" and self.last_path is not None:
            return [start] + list(self.last_path)
        return start
