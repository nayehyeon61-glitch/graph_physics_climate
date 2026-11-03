"""stage11b: simplicial (vertex-edge-face) message passing on the gauge-aware mesh complex.

The block realises the "extended connection" of the design doc ``docs/design/Connection_Curvature_Graph_KO.md``:
vertex features (equivariant Field) are restricted to the edges as gauge-invariant 1-forms (d0 of the scalars,
flat of the vectors), the edges are restricted to the faces as 2-forms (d1 = circulation), learned MLPs act on the
invariant edge / face cochains, and the result comes back through the transposed coboundaries (d1T, d0T) and the
equivariant sharp map.  Because every edge / face quantity is a gauge invariant and the only equivariant step is
``sharp`` (which returns vectors in the vertex gauges), the block is exactly SO(2)-gauge equivariant.  d1 d0 = 0
makes the gradient part of the vertex scalars invisible to the faces: only the rotational content reaches the
2-forms, which is the structural prior of synoptic-scale flow.  The restriction maps are fixed (geometry), only the
edge / face MLPs and the vertex read-out are learned (MPSN / CWN style, Bodnar et al. 2021).
"""

from __future__ import annotations

import torch
from torch import nn

from gpclimate.geometry.connection import MeshComplex
from gpclimate.nn.so2 import EqLinear, EqNorm, Field


def _mlp(n_in: int, n_out: int, hidden: int) -> nn.Module:
    return nn.Sequential(nn.LayerNorm(n_in), nn.Linear(n_in, hidden), nn.GELU(), nn.Linear(hidden, n_out))


class SimplicialBlock(nn.Module):
    """One vertex -> edge -> face -> edge -> vertex round trip with a residual vertex update.

    Edge input (gauge invariant, [B, E, cs + cv + 2]): d0 of the vertex scalars, flat of the vertex vectors,
    edge length and dual/primal length ratio.  Face input ([B, F, ce + 2]): d1 of the edge features (circulation =
    curvature of the edge 1-forms), face area and the planetary term sin(lat)·area (Coriolis 2-form).  The vertex
    read-out takes d0T of the updated edge features as scalars and sharp of them as vectors, so the new vertex
    Field is (invariant scalars, equivariant vectors).  The read-out is zero-initialised: the block starts as the
    identity and the rest of the network keeps its pretrained behaviour at step 0.
    """

    def __init__(self, cs: int, cv: int, ce: int = 16, cf: int = 16, hidden: int = 32):
        super().__init__()
        self.cs, self.cv, self.ce, self.cf = cs, cv, ce, cf
        self.edge_in = _mlp(cs + cv + 2, ce, hidden)
        self.face_in = _mlp(ce + 2, cf, hidden)
        self.edge_out = _mlp(ce + cf, ce, hidden)
        self.v_norm = EqNorm(ce, ce)
        self.v_out = EqLinear(ce, ce, cs, cv)
        with torch.no_grad():  # identity at initialisation
            for p in self.v_out.parameters():
                p.zero_()

    # ------------------------------------------------------------------ cochain features
    def edge_features(self, x: Field, cx: MeshComplex) -> torch.Tensor:
        """Gauge-invariant edge input [B, E, cs + cv + 2]."""
        B = x.s.shape[0]
        geo = torch.stack([cx.length, cx.s1], -1)[None].expand(B, -1, -1)
        return torch.cat([cx.d0(x.s), cx.flat(x.v), geo], -1)

    def face_features(self, h_e: torch.Tensor, cx: MeshComplex, sinlat: torch.Tensor) -> torch.Tensor:
        """Gauge-invariant face input [B, F, ce + 2]: circulation of the edge features + geometric 2-forms."""
        B = h_e.shape[0]
        sin_f = sinlat[cx.faces].mean(1)
        geo = torch.stack([cx.face_area, sin_f * cx.face_area], -1)[None].expand(B, -1, -1)
        return torch.cat([cx.d1(h_e), geo], -1)

    def forward(self, x: Field, cx: MeshComplex, sinlat: torch.Tensor) -> Field:
        h_e = self.edge_in(self.edge_features(x, cx))  # [B, E, ce]   1-forms
        h_f = self.face_in(self.face_features(h_e, cx, sinlat))  # [B, F, cf]   2-forms
        h_e = self.edge_out(torch.cat([h_e, cx.d1T(h_f)], -1))  # faces -> edges (coboundary transpose)
        vert = Field(cx.d0T(h_e), cx.sharp(h_e))  # edges -> vertices: invariant scalars, equivariant vectors
        upd = self.v_out(self.v_norm(vert))
        return Field(x.s + upd.s, x.v + upd.v)
