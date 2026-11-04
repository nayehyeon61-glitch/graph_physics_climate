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
    # stage13: continuous neural-operator layer on the grid after the mesh V-Net ("" = off, "fft" = Fourier
    # neural operator, "sht" = spherical neural operator via torch_harmonics); residual on the delta, zero-initialised
    neural_op: str = ""
    no_hidden: int = 16
    no_layers: int = 2
    no_modes: Tuple[int, int] = (16, 32)


class DynamicsVNet(nn.Module):
    def __init__(self, spec: VariableSpec, cfg: VNetConfig = VNetConfig(), guide_dims: Tuple[int, int] = (48, 12)):
        super().__init__()
        self.spec, self.cfg = spec, cfg
        cs, cv, nb, hd = cfg.cs, cfg.cv, cfg.n_basis, cfg.hidden
        self.lift = GridLift(spec.n_s + spec.n_cond_s, spec.n_v + spec.n_cond_v, cs, cv, nb, hd)
        self.gnn = nn.ModuleList([GaugeStack(cs, cv, d, nb, hd) for d in cfg.depth])
        self.phys_ops = None
        if cfg.use_guide and cfg.fusion == "pigt":
            pcs, pcv = guide_dims
            ops_dims = None
            if cfg.pigt_ops:
                self.phys_ops = PhysicsOperators(spec.n_s, spec.n_v, spec.n_cond_s)
                ops_dims = self.phys_ops.out_dims
            self.fusion = nn.ModuleList([
                PhysicsIntegration(cs, cv, pcs, pcv, ops_dims, cfg.pigt_layers, cfg.heads, cfg.ds, cfg.dv, nb)
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
        self.cont_op = None
        if cfg.neural_op:
            from gpclimate.nn.neural_operator import ContinuousOperatorBlock
            self.cont_op = ContinuousOperatorBlock(spec.n_s, spec.n_v, spec.n_cond_s, spec.n_cond_v, cfg.no_hidden,
                                                   cfg.no_layers, tuple(cfg.no_modes), cfg.neural_op)
        with torch.no_grad():  # start from persistence (delta = 0)
            self.head_out.lin_s.weight.zero_()
            self.head_out.lin_s.bias.zero_()
            if self.head_out.w_re is not None:
                self.head_out.w_re.zero_()
                self.head_out.w_im.zero_()

    def _fuse(self, k: int, h: Field, physics: Optional[Dict[str, Field]], key: str, edges,
              ops: Optional[Field] = None) -> Field:
        if self.fusion is None or physics is None:
            return h
        if self.cfg.fusion == "pigt":
            return self.fusion[k](h, physics[key], ops, edges)
        return self.fusion[k](h, physics[key], edges)

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

    def forward(self, state: Field, cond: Field, physics: Optional[Dict[str, Field]], g: GraphView) -> Field:
        B = state.s.shape[0]
        cond = broadcast_batch(cond, B)
        x = Field(torch.cat([state.s, cond.s], -1), torch.cat([state.v, cond.v], -1))
        grid_emb, m0 = self.lift(x, g)
        O = self.operator_tokens(state, cond, g) if (physics is not None and self.cfg.fusion == "pigt") else [None] * 3
        H0 = self.gnn[0](m0, g.mesh_edges[0])
        F0 = self._fuse(0, H0, physics, "P0", g.attn_edges[0], O[0])
        H1 = self.gnn[1](self.down01(F0, g.down[0]), g.mesh_edges[1])
        F1 = self._fuse(1, H1, physics, "P1", g.attn_edges[1], O[1])
        H2 = self.gnn[2](self.down12(F1, g.down[1]), g.mesh_edges[2])
        F2 = self._fuse(2, H2, physics, "P2", g.attn_edges[2], O[2])
        H1u = self.gnn[3](self.merge1(self.up21(F2, g.up[1]), H1), g.mesh_edges[1])
        F1u = self._fuse(3, H1u, physics, "P1u", g.attn_edges[1], O[1])
        H0u = self.gnn[4](self.merge0(self.up10(F1u, g.up[0]), H0), g.mesh_edges[0])
        F0u = self._fuse(4, H0u, physics, "P0u", g.attn_edges[0], O[0])
        out = self.head_out(self.head_dec(F0u, g.mesh2grid, extra=grid_emb))
        if self.cont_op is not None:  # stage13: Lie guide -> discrete mesh GNN -> continuous operator on the grid
            H, W = g.extras["grid_hw"]
            out = out + self.cont_op(x, out, H, W, g.extras["grid_lat"], g.extras.get("grid_gauge"))
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

    def forward(self, state: Field, cond: Field, g: GraphView) -> Tuple[Field, Field]:
        """Deterministic forecast and its change in delta-scale units ``d = (X_bar - X_t) / delta_scale``."""
        if self.vnet.cfg.integrator != "ode":
            P = self.physics(state, cond, g)
            d = self.vnet(state, cond, P, g)
            forecast = Field(state.s + d.s * self.delta_scale_s, state.v + d.v * self.delta_scale_v)
            return forecast, d
        forecast = self.integrate(state, cond, g)
        d = Field((forecast.s - state.s) / self.delta_scale_s, (forecast.v - state.v) / self.delta_scale_v)
        return forecast, d

    # ---------------------------------------------------------------- stage S8: graph neural ODE dynamics
    def physics_drift(self, x: Field, cond: Field, g: GraphView, P: Optional[Dict[str, Field]]) -> Field:
        """Change of the normalised state over one sub-step from the frozen guide's 6 h tendency."""
        assert self.guide is not None, "ode_physics needs the physics guide"
        T = self.guide.physics_tendency(P, x, g)
        assert T is not None, "ode_physics needs a guide with a tendency head or a closure head"
        steps = self.lead_steps / self.vnet.cfg.ode_steps  # data steps (6 h) per sub-step
        return Field(T.s * self.tend_scale_s * steps, T.v * self.tend_scale_v * steps)

    def rate(self, x: Field, cond: Field, g: GraphView, p_map: Optional[Callable] = None) -> Field:
        """Increment of one sub-step: [physics drift] + learned correction (V-Net output in lead-change units / K).
        Physics features are recomputed from the current sub-step state, so the guide and the learned dynamics stay
        coupled along the whole integration path.  ``p_map`` (diagnostics only) transforms the physics features
        of every sub-step, e.g. zeroing or shuffling them to measure the guide dependence of the ODE dynamics."""
        cfg = self.vnet.cfg
        P = self.physics(x, cond, g)
        if p_map is not None and P is not None:
            P = p_map(P)
        d = self.vnet(x, cond, P, g)
        h = 1.0 / cfg.ode_steps
        f = Field(d.s * self.delta_scale_s * h, d.v * self.delta_scale_v * h)
        if cfg.ode_physics:
            drift = self.physics_drift(x, cond, g, P)
            f = Field(f.s + drift.s, f.v + drift.v)
        return f

    def integrate(self, state: Field, cond: Field, g: GraphView, p_map: Optional[Callable] = None) -> Field:
        """Fixed-step integration over the lead (explicit Euler or Heun); conditions are held fixed within the lead."""
        cfg = self.vnet.cfg
        x = state
        for _ in range(cfg.ode_steps):
            f1 = self.rate(x, cond, g, p_map)
            if cfg.ode_solver == "heun":
                x1 = Field(x.s + f1.s, x.v + f1.v)
                f2 = self.rate(x1, cond, g, p_map)
                x = Field(x.s + 0.5 * (f1.s + f2.s), x.v + 0.5 * (f1.v + f2.v))
            else:
                x = Field(x.s + f1.s, x.v + f1.v)
        return x

    def physics_only(self, state: Field, cond: Field, g: GraphView) -> Field:
        """Forecast from the physics drift alone (V-Net switched off): diagnostic of the guide's tendency."""
        cfg = self.vnet.cfg
        x = state
        for _ in range(cfg.ode_steps):
            drift = self.physics_drift(x, cond, g, self.physics(x, cond, g))
            x = Field(x.s + drift.s, x.v + drift.v)
        return x

    def rollout(self, state: Field, conds: List[Field], g: GraphView) -> List[Field]:
        """Autoregressive deterministic rollout; physics features are recomputed from each new state."""
        out = []
        for c in conds:
            state, _ = self(state, c, g)
            out.append(state)
        return out
