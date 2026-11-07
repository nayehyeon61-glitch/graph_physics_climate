"""Stage A - Lie-equivariant Physical GNN (the physics guide E_theta).

``{P0, P1, P2, P1*, P0*} = E_theta(X~_t, T_g(c_t), G, sigma)``  (design doc section 3.1)

* input: (noisy) current state, conditions (static + known forcing) and the
  perturbation level ``sigma`` (given to the model as a scalar condition);
* a gauge-equivariant V-shaped encoder/decoder over the 3-level icosahedral
  hierarchy (Fine -> Medium -> Coarse -> Medium -> Fine) produces per-resolution
  physics features; ``*`` marks decoder-side features (not a time index);
* Stage-A heads (training only): reconstruction of the clean state ``T_g(X_t)``,
  physics diagnostics (vorticity / divergence of every wind pair) and, with
  ``tend_head``, the short-range (one data step, 6 h) tendency of the clean state;
* with ``physics_ops`` the finest-mesh embedding additionally receives fixed,
  exactly equivariant primitive-equation terms (``gpclimate.nn.physics_ops``:
  gradients, divergence / curl, advection, Coriolis, absolute vorticity, ...)
  evaluated on the input, so dynamics enter the guide by construction instead of
  having to be rediscovered from neighbour differences.

Scalars (T, z, q, ...) enter as frequency-0 channels, horizontal winds as
frequency-1 tangent vectors; every layer is SO(2) gauge equivariant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from gpclimate.data.variables import VariableSpec
from gpclimate.geometry.graphs import GraphView
from gpclimate.nn.gauge_conv import GaugeStack, GaugeTransfer, Merge, transport_interpolate
from gpclimate.nn.connection_ops import ConnectionCurvatureOperators
from gpclimate.nn.physics_ops import PhysicsOperators
from gpclimate.nn.physics_ops import EARTH_RADIUS
from gpclimate.nn.primitive2d import DEFAULT_TERMS, ReducedPrimitiveTendencies
from gpclimate.nn.semi_lagrangian import SphereSemiLagrangian
from gpclimate.nn.simplicial import SimplicialBlock
from gpclimate.nn.so2 import EqFFN, EqLinear, EqNorm, Field

LEVEL_KEYS = ("P0", "P1", "P2", "P1u", "P0u")


@dataclass
class GuideConfig:
    cs: int = 48
    cv: int = 12
    depth: Tuple[int, int, int, int, int] = (2, 2, 2, 2, 2)  # enc0, enc1, bottleneck, dec1, dec0
    n_basis: int = 8
    hidden: int = 64
    physics_ops: bool = False  # fixed equivariant differential operators as extra mesh-0 inputs
    tend_head: bool = False  # Stage-A head predicting the one-step tendency of the clean state
    # stage S5a (layer-wise physical constraints): the tendency is  F(X) + S_theta  with F the fixed resolved
    # dynamics of the reduced primitive equations (gpclimate.nn.primitive2d) and S_theta a learned closure read
    # out on mesh 0; the Stage-A target of the closure head is the observed tendency minus F.
    closure_tend: bool = False
    resolved_terms: Tuple[str, ...] = DEFAULT_TERMS
    resolved_coef: Optional[dict] = None  # fixed calibration coefficients (scripts/resolved_fraction.py --fit)
    # stage11a: exact connection / curvature operators of the flow (DEC circulation, divergence, strain,
    # Okubo-Weiss, Helmholtz potentials, Lie brackets of the wind levels) as extra mesh-0 inputs
    curvature_ops: bool = False
    # stage11b: simplicial vertex-edge-face message passing on mesh 0 (0 = off, 1 = block on the mesh-0 input,
    # 2 = blocks on the mesh-0 input and on the decoder output P0u)
    simplicial: int = 0
    # stage17c (Lie transport guide): a flow head gives, per level group (850 hPa: t850 q700 u/v850; 500 hPa: z500
    # u/v500; surface: t2m msl u/v10), an equivariant correction to that level's wind; the corrected flow carries the
    # state one 6 h step by semi-Lagrangian transport with parallel-transported vectors (gpclimate.nn.semi_lagrangian)
    # and the tendency head becomes the source term: G(X) = T_u(X) + S_theta(X) dt.  Needs tend_head.
    flow_head: bool = False


# stage17c level groups: the wind that carries each state channel
FLOW_WINDS = ("u850", "u500", "u10")
FLOW_SCALAR_GROUP = {"z500": 1, "t850": 0, "q700": 0, "t2m": 2, "msl": 2}
DT_DATA = 6 * 3600.0


class GridLift(nn.Module):
    """Grid inputs -> grid embedding -> finest-mesh field (encoder adapter, grid-to-graph)."""

    def __init__(self, n_in_s: int, n_in_v: int, cs: int, cv: int, n_basis: int, hidden: int):
        super().__init__()
        self.embed = EqLinear(n_in_s, n_in_v, cs, cv)
        self.norm = EqNorm(cs, cv)
        self.ffn = EqFFN(cs, cv)
        self.enc = GaugeTransfer(cs, cv, cs, cv, n_basis, hidden)

    def forward(self, x: Field, g: GraphView) -> Tuple[Field, Field]:
        e = self.embed(x)
        e = e + self.ffn(self.norm(e))
        return e, self.enc(e, g.grid2mesh)


def broadcast_batch(f: Field, B: int) -> Field:
    if f.s.shape[0] == B:
        return f
    return Field(f.s.expand(B, *f.s.shape[1:]), f.v.expand(B, *f.v.shape[1:]))


class LiePhysicalGNN(nn.Module):
    def __init__(self, spec: VariableSpec, cfg: GuideConfig = GuideConfig()):
        super().__init__()
        self.spec, self.cfg = spec, cfg
        cs, cv, nb, hd = cfg.cs, cfg.cv, cfg.n_basis, cfg.hidden
        self.lift = GridLift(spec.n_s + spec.n_cond_s + 1, spec.n_v + spec.n_cond_v, cs, cv, nb, hd)
        d = cfg.depth
        self.enc0 = GaugeStack(cs, cv, d[0], nb, hd)
        self.down01 = GaugeTransfer(cs, cv, cs, cv, nb, hd)
        self.enc1 = GaugeStack(cs, cv, d[1], nb, hd)
        self.down12 = GaugeTransfer(cs, cv, cs, cv, nb, hd)
        self.mid = GaugeStack(cs, cv, d[2], nb, hd)
        self.up21 = GaugeTransfer(cs, cv, cs, cv, nb, hd)
        self.merge1 = Merge(cs, cv, cs, cv, cs, cv)
        self.dec1 = GaugeStack(cs, cv, d[3], nb, hd)
        self.up10 = GaugeTransfer(cs, cv, cs, cv, nb, hd)
        self.merge0 = Merge(cs, cv, cs, cv, cs, cv)
        self.dec0 = GaugeStack(cs, cv, d[4], nb, hd)
        # fixed physics operators on the finest mesh (no parameters) + learned equivariant embedding
        self.phys_ops: Optional[PhysicsOperators] = None
        if cfg.physics_ops:
            self.phys_ops = PhysicsOperators(spec.n_s, spec.n_v, spec.n_cond_s)
            ps, pv = self.phys_ops.out_dims
            self.phys_norm = EqNorm(ps, pv)
            self.phys_embed = EqLinear(ps, pv, cs, cv)
        self.curv_ops: Optional[ConnectionCurvatureOperators] = None
        if cfg.curvature_ops:
            self.curv_ops = ConnectionCurvatureOperators(spec.n_v)
            qs, qv = self.curv_ops.out_dims
            self.curv_norm = EqNorm(qs, qv)
            self.curv_embed = EqLinear(qs, qv, cs, cv)
        # Stage-A heads (graph-to-grid adapter + linear read-outs)
        self.head_dec = GaugeTransfer(cs, cv, cs, cv, nb, hd)
        self.simp_in = SimplicialBlock(cs, cv) if cfg.simplicial >= 1 else None
        self.simp_out = SimplicialBlock(cs, cv) if cfg.simplicial >= 2 else None
        self.rec_out = EqLinear(cs, cv, spec.n_s, spec.n_v)
        self.aux_out = EqLinear(cs, cv, len(spec.aux_names), 0)
        self.tend_out = EqLinear(cs, cv, spec.n_s, spec.n_v) if cfg.tend_head else None
        # closure head (mesh 0) and the fixed resolved dynamics it completes
        self.resolved: Optional[ReducedPrimitiveTendencies] = None
        self.closure_out: Optional[EqLinear] = None
        if cfg.closure_tend:
            self.resolved = ReducedPrimitiveTendencies(spec, tuple(cfg.resolved_terms))
            self.resolved.set_coefficients(cfg.resolved_coef)
            self.closure_out = EqLinear(cs, cv, spec.n_s, spec.n_v)
        # stage17c flow head (zero-initialised: the flow starts as the level winds themselves)
        self.flow_out: Optional[EqLinear] = None
        self._sl: Optional[SphereSemiLagrangian] = None
        if cfg.flow_head:
            if not cfg.tend_head:
                raise ValueError("flow_head needs tend_head (the tendency head becomes the source term)")
            winds = [a for a, _ in spec.vectors]
            missing = [n for n in FLOW_WINDS if n not in winds] + [n for n in spec.scalars if n not in FLOW_SCALAR_GROUP]
            if missing:
                raise ValueError(f"flow_head supports the base11 layout only; no flow group for {missing}")
            self._flow_vec = [winds.index(n) for n in FLOW_WINDS]
            self._flow_groups = [([i for i, n in enumerate(spec.scalars) if FLOW_SCALAR_GROUP[n] == k], [self._flow_vec[k]])
                                 for k in range(len(FLOW_WINDS))]
            self.flow_out = EqLinear(cs, cv, 0, len(FLOW_WINDS))
            with torch.no_grad():
                self.flow_out.w_re.zero_()
                self.flow_out.w_im.zero_()
            self.register_buffer("flow_v_scale", torch.ones(spec.n_v), persistent=False)  # m/s per normalised unit

    @property
    def feature_dims(self) -> Tuple[int, int]:
        return self.cfg.cs, self.cfg.cv

    def physics_terms(self, state: Field, cond: Field, g: GraphView) -> Field:
        """Raw operator outputs on mesh 0 (before normalisation / embedding), for diagnostics."""
        assert self.phys_ops is not None
        B = state.s.shape[0]
        cond = broadcast_batch(cond, B)
        raw = transport_interpolate(Field(torch.cat([state.s, cond.s], -1), state.v), g.grid2mesh)
        n_s = self.spec.n_s
        return self.phys_ops(raw.s[..., :n_s], raw.v, raw.s[..., n_s:], g.mesh_sinlat[0], g.mesh_edges[0])

    def curvature_terms(self, state: Field, g: GraphView) -> Field:
        """Raw connection / curvature operator outputs on mesh 0 (stage11a), for the encoder and diagnostics."""
        assert self.curv_ops is not None
        raw = transport_interpolate(Field(state.s[..., :0], state.v), g.grid2mesh)
        return self.curv_ops(raw.v, g.mesh_sinlat[0], g.mesh_edges[0], g.extras["complexes"][0])

    def forward(self, state: Field, cond: Field, sigma: torch.Tensor, g: GraphView) -> Dict[str, Field]:
        B, G = state.s.shape[:2]
        cond = broadcast_batch(cond, B)
        sig = sigma.to(state.s).reshape(B, 1, 1).expand(B, G, 1)
        x = Field(torch.cat([state.s, cond.s, sig], -1), torch.cat([state.v, cond.v], -1))
        _, m0 = self.lift(x, g)
        if self.phys_ops is not None:
            ops = self.phys_embed(self.phys_norm(self.physics_terms(state, cond, g)))
            m0 = Field(m0.s + ops.s, m0.v + ops.v)
        if self.curv_ops is not None:
            cur = self.curv_embed(self.curv_norm(self.curvature_terms(state, g)))
            m0 = Field(m0.s + cur.s, m0.v + cur.v)
        if self.simp_in is not None:  # stage11b
            m0 = self.simp_in(m0, g.extras["complexes"][0], g.mesh_sinlat[0])
        P0 = self.enc0(m0, g.mesh_edges[0])
        P1 = self.enc1(self.down01(P0, g.down[0]), g.mesh_edges[1])
        P2 = self.mid(self.down12(P1, g.down[1]), g.mesh_edges[2])
        P1u = self.dec1(self.merge1(self.up21(P2, g.up[1]), P1), g.mesh_edges[1])
        P0u = self.dec0(self.merge0(self.up10(P1u, g.up[0]), P0), g.mesh_edges[0])
        if self.simp_out is not None:
            P0u = self.simp_out(P0u, g.extras["complexes"][0], g.mesh_sinlat[0])
        return {"P0": P0, "P1": P1, "P2": P2, "P1u": P1u, "P0u": P0u}

    def heads(self, feats: Dict[str, Field], g: GraphView) -> Tuple[Field, torch.Tensor]:
        d = self.head_dec(feats["P0u"], g.mesh2grid)
        return self.rec_out(d), self.aux_out(d).s

    # ---------------------------------------------------------------- stage S5a: resolved dynamics + closure (mesh 0)
    def closure(self, feats: Dict[str, Field]) -> Field:
        """Learned closure tendency S_theta on mesh 0 (normalised 6 h tendency units)."""
        assert self.closure_out is not None
        return self.closure_out(feats["P0u"])

    def resolved_tendency(self, state_norm: Field, g: GraphView) -> Field:
        """Fixed resolved dynamics F(X) on mesh 0 (normalised 6 h tendency units), from the *clean* grid state."""
        assert self.resolved is not None
        return self.resolved.mesh_forward(state_norm, g)

    def tendency_mesh(self, feats: Dict[str, Field], state_norm: Field, g: GraphView) -> Field:
        """Full physics tendency F(X) + S_theta on mesh 0."""
        return self.resolved_tendency(state_norm, g) + self.closure(feats)

    def physics_tendency(self, feats: Dict[str, Field], state_norm: Field, g: GraphView) -> Optional[Field]:
        """One-data-step (6 h) tendency of the clean state on the *grid*, in normalised tendency units -- the
        physics drift of the stage S8 graph neural ODE: the Stage-A tendency head when present, otherwise the
        closure-structured ``F(X) + S_theta`` transported from mesh 0 to the grid; ``None`` without either head."""
        if self.closure_out is not None:
            return transport_interpolate(self.tendency_mesh(feats, state_norm, g), g.mesh2grid)
        if self.tend_out is not None:
            return self.tend_out(self.head_dec(feats["P0u"], g.mesh2grid))
        return None

    # ---------------------------------------------------------------- stage17c: Lie transport guide
    def flow(self, feats: Dict[str, Field], state_norm: Field, g: GraphView) -> torch.Tensor:
        """Carrying flow per level group [B, G, 3] (complex, m/s, grid node frames): level wind + learned correction."""
        assert self.flow_out is not None
        corr = self.flow_out(self.head_dec(feats["P0u"], g.mesh2grid)).v
        return (state_norm.v[..., self._flow_vec] + corr) * self.flow_v_scale[self._flow_vec]

    def transport(self, feats: Dict[str, Field], state_norm: Field, g: GraphView, steps: float = 1.0) -> Field:
        """The (normalised) state carried by the guide's flow over ``steps`` data steps of 6 h: semi-Lagrangian
        transport of every channel by its level group's flow, vectors parallel-transported (gauge-equivariant: the
        computation runs in the canonical east/north frame and the result is rotated back into the node frames)."""
        H, W = g.extras["grid_hw"]
        if self._sl is None or (self._sl.H, self._sl.W) != (H, W):
            self._sl = SphereSemiLagrangian(g.extras["grid_lat"], g.extras["grid_lon"])
        sl = self._sl.to(state_norm.s.device)
        rot = torch.exp(1j * g.extras["grid_gauge"].to(state_norm.s))[:, None]  # node frame -> canonical
        flow = self.flow(feats, state_norm, g) * rot
        xv = state_norm.v * rot
        B = state_norm.s.shape[0]

        def img(t: torch.Tensor) -> torch.Tensor:
            return t.transpose(1, 2).reshape(B, t.shape[-1], H, W)

        out_s, out_v = torch.zeros_like(state_norm.s), torch.zeros_like(xv)
        disp = flow * (steps * DT_DATA / EARTH_RADIUS)  # radians of arc
        for k, (si, vi) in enumerate(self._flow_groups):
            ts, tv = sl(img(state_norm.s[..., si]), img(xv[..., vi]), disp[..., k].reshape(B, H, W))
            out_s[..., si] = ts.flatten(2).transpose(1, 2)
            out_v[..., vi] = tv.flatten(2).transpose(1, 2)
        return Field(out_s, out_v * rot.conj())

    def heads_all(self, feats: Dict[str, Field], g: GraphView) -> Tuple[Field, torch.Tensor, Optional[Field]]:
        """(reconstruction, aux diagnostics, one-step tendency or None)."""
        d = self.head_dec(feats["P0u"], g.mesh2grid)
        tend = self.tend_out(d) if self.tend_out is not None else None
        return self.rec_out(d), self.aux_out(d).s, tend
