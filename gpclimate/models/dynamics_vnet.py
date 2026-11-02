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
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from gpclimate.data.variables import VariableSpec
from gpclimate.geometry.graphs import GraphView
from gpclimate.models.lie_physical_gnn import GridLift, GuideConfig, LiePhysicalGNN, broadcast_batch
from gpclimate.nn.attention import PhysicsFusion
from gpclimate.nn.gauge_conv import GaugeStack, GaugeTransfer, Merge
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


class DynamicsVNet(nn.Module):
    def __init__(self, spec: VariableSpec, cfg: VNetConfig = VNetConfig(), guide_dims: Tuple[int, int] = (48, 12)):
        super().__init__()
        self.spec, self.cfg = spec, cfg
        cs, cv, nb, hd = cfg.cs, cfg.cv, cfg.n_basis, cfg.hidden
        self.lift = GridLift(spec.n_s + spec.n_cond_s, spec.n_v + spec.n_cond_v, cs, cv, nb, hd)
        self.gnn = nn.ModuleList([GaugeStack(cs, cv, d, nb, hd) for d in cfg.depth])
        if cfg.use_guide:
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
        with torch.no_grad():  # start from persistence (delta = 0)
            self.head_out.lin_s.weight.zero_()
            self.head_out.lin_s.bias.zero_()
            if self.head_out.w_re is not None:
                self.head_out.w_re.zero_()
                self.head_out.w_im.zero_()

    def _fuse(self, k: int, h: Field, physics: Optional[Dict[str, Field]], key: str, edges) -> Field:
        if self.fusion is None or physics is None:
            return h
        return self.fusion[k](h, physics[key], edges)

    def forward(self, state: Field, cond: Field, physics: Optional[Dict[str, Field]], g: GraphView) -> Field:
        B = state.s.shape[0]
        cond = broadcast_batch(cond, B)
        x = Field(torch.cat([state.s, cond.s], -1), torch.cat([state.v, cond.v], -1))
        grid_emb, m0 = self.lift(x, g)
        H0 = self.gnn[0](m0, g.mesh_edges[0])
        F0 = self._fuse(0, H0, physics, "P0", g.attn_edges[0])
        H1 = self.gnn[1](self.down01(F0, g.down[0]), g.mesh_edges[1])
        F1 = self._fuse(1, H1, physics, "P1", g.attn_edges[1])
        H2 = self.gnn[2](self.down12(F1, g.down[1]), g.mesh_edges[2])
        F2 = self._fuse(2, H2, physics, "P2", g.attn_edges[2])
        H1u = self.gnn[3](self.merge1(self.up21(F2, g.up[1]), H1), g.mesh_edges[1])
        F1u = self._fuse(3, H1u, physics, "P1u", g.attn_edges[1])
        H0u = self.gnn[4](self.merge0(self.up10(F1u, g.up[0]), H0), g.mesh_edges[0])
        F0u = self._fuse(4, H0u, physics, "P0u", g.attn_edges[0])
        return self.head_out(self.head_dec(F0u, g.mesh2grid, extra=grid_emb))


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
        P = self.physics(state, cond, g)
        d = self.vnet(state, cond, P, g)
        forecast = Field(state.s + d.s * self.delta_scale_s, state.v + d.v * self.delta_scale_v)
        return forecast, d

    def rollout(self, state: Field, conds: List[Field], g: GraphView) -> List[Field]:
        """Autoregressive deterministic rollout; physics features are recomputed from each new state."""
        out = []
        for c in conds:
            state, _ = self(state, c, g)
            out.append(state)
        return out
