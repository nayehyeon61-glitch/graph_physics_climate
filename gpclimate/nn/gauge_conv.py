"""SO(2) gauge-equivariant message passing on spherical graphs.

Message for an edge ``j -> i`` (cf. Gauge Equivariant Mesh CNNs, de Haan et al. 2021):

1. transport the sender's vectors into the receiver's gauge: ``exp(i alpha_ij) v_j``;
2. rotate everything into the *edge frame* (receiver gauge rotated so that axis 1
   points at the sender): multiply by ``exp(-i phi_ij)``.  All coordinates are
   now gauge invariant (``phi' = phi - theta_i`` cancels the receiver rotation);
3. an arbitrary MLP maps those invariants (+ a radial basis of the geodesic
   distance) to output scalars and output vectors *in the edge frame*;
4. rotate output vectors back with ``exp(+i phi_ij)`` and aggregate.

Steps 2-4 are the general solution of the steerable kernel constraint
``K(phi - theta) = rho_out(-theta) K(phi) rho_in(theta)`` with a nonlinear core,
so the layer is exactly equivariant by construction (not only because of the
transport, cf. design doc section 3.2).  Scalars can produce vectors (e.g. a
discrete gradient ``sum_j f(s_j - s_i) exp(i phi_ij)``) and vice versa
(divergence / curl like contractions).

Library reuse: ``torch_geometric.nn.MessagePassing`` (gather / scatter / mean
aggregation), ``torch_geometric.nn.MLP`` (edge MLP), ``e3nn.math.soft_one_hot_linspace``
(radial basis) and ``e3nn.nn.FullyConnectedNet`` (radial weights, as in e3nn convolutions).
"""

from __future__ import annotations

from typing import Optional

import torch
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import FullyConnectedNet
from torch import nn
from torch_geometric.nn import MLP, MessagePassing
from torch_geometric.utils import scatter

from gpclimate.geometry.graphs import EdgeSet
from gpclimate.nn.so2 import EqFFN, EqLinear, EqNorm, Field, cat_fields, pack, phase, unpack


def radial_embedding(dist: torch.Tensor, cutoff: float, n_basis: int) -> torch.Tensor:
    """e3nn Gaussian radial basis of the geodesic distance, ~unit second moment."""
    return soft_one_hot_linspace(dist, 0.0, cutoff, n_basis, basis="gaussian", cutoff=False)


def transport_interpolate(x: Field, edges: EdgeSet, n_dst: Optional[int] = None) -> Field:
    """Fixed geometric transfer: sum_j w_ij (s_j, exp(i alpha_ij) v_j).

    For ``down`` edges ``w`` are normalised area weights (area-weighted pooling),
    for ``up``/``mesh2grid`` inverse-distance weights.  Exactly equivariant.
    """
    if edges.weight is None:
        raise ValueError("edge set has no interpolation weights")
    n_dst = edges.n_dst if n_dst is None else n_dst
    w = edges.weight
    s = x.s.index_select(-2, edges.src) * w[:, None]
    v = x.v.index_select(-2, edges.src) * (phase(edges.alpha) * w)[:, None]
    s = scatter(s, edges.dst, dim=-2, dim_size=n_dst, reduce="sum")
    v = torch.view_as_complex(
        scatter(torch.view_as_real(v), edges.dst, dim=-3, dim_size=n_dst, reduce="sum").contiguous())
    return Field(s, v)


class GaugeConv(MessagePassing):
    def __init__(self, cs_src: int, cv_src: int, cs_dst: int, cv_dst: int, cs_out: int, cv_out: int,
                 n_basis: int = 8, hidden: int = 64, aggr: str = "mean"):
        super().__init__(aggr=aggr, node_dim=-2)
        self.dims = (cs_src, cv_src, cs_dst, cv_dst, cs_out, cv_out)
        self.n_basis = n_basis
        in_dim = cs_src + cs_dst + 2 * cv_src + 2 * cv_dst + n_basis
        self.mlp = MLP([in_dim, hidden, hidden, cs_out + 2 * cv_out], act="silu", norm=None)
        self.radial = FullyConnectedNet([n_basis, hidden, cs_out + cv_out], act=torch.nn.functional.silu)

    def forward(self, x_src: Field, x_dst: Field, edges: EdgeSet) -> Field:
        cs_src, cv_src, cs_dst, cv_dst, cs_out, cv_out = self.dims
        e = edges if bool(edges.directional.all()) else edges.subset(edges.directional)
        emb = radial_embedding(e.dist, e.cutoff, self.n_basis)
        out = self.propagate(e.edge_index, x=(pack(x_src), pack(x_dst)), size=(e.n_src, e.n_dst),
                             alpha=e.alpha, phi=e.phi, emb=emb)
        return unpack(out, cs_out, cv_out)

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, alpha: torch.Tensor, phi: torch.Tensor,
                emb: torch.Tensor) -> torch.Tensor:
        cs_src, cv_src, cs_dst, cv_dst, cs_out, cv_out = self.dims
        fi = unpack(x_i, cs_dst, cv_dst)
        fj = unpack(x_j, cs_src, cv_src)
        vi = fi.v * phase(-phi)[:, None]  # receiver vectors in the edge frame
        vj = fj.v * phase(alpha - phi)[:, None]  # transported sender vectors in the edge frame
        lead = x_i.shape[:-2]
        h = torch.cat([fi.s, fj.s, torch.view_as_real(vi).flatten(-2), torch.view_as_real(vj).flatten(-2),
                       emb.expand(*lead, *emb.shape)], -1)
        o = self.mlp(h)
        r = self.radial(emb)
        ms = o[..., :cs_out] * r[:, :cs_out]
        mv = torch.complex(o[..., cs_out:cs_out + cv_out], o[..., cs_out + cv_out:])
        mv = mv * (phase(phi)[:, None] * r[:, cs_out:])  # back to the receiver gauge
        return pack(Field(ms, mv))


class GaugeBlock(nn.Module):
    """Pre-norm residual block: x + conv(norm x); x + FFN(norm x). One GNN layer of a level."""

    def __init__(self, cs: int, cv: int, n_basis: int = 8, hidden: int = 64):
        super().__init__()
        self.n1 = EqNorm(cs, cv)
        self.conv = GaugeConv(cs, cv, cs, cv, cs, cv, n_basis, hidden)
        self.n2 = EqNorm(cs, cv)
        self.ffn = EqFFN(cs, cv)

    def forward(self, f: Field, edges: EdgeSet) -> Field:
        h = self.n1(f)
        f = f + self.conv(h, h, edges)
        return f + self.ffn(self.n2(f))


class GaugeStack(nn.Module):
    def __init__(self, cs: int, cv: int, depth: int, n_basis: int = 8, hidden: int = 64):
        super().__init__()
        self.blocks = nn.ModuleList([GaugeBlock(cs, cv, n_basis, hidden) for _ in range(depth)])

    def forward(self, f: Field, edges: EdgeSet) -> Field:
        for b in self.blocks:
            f = b(f, edges)
        return f


class GaugeTransfer(nn.Module):
    """src-node field -> dst-node field (pool, unpool, grid->mesh encoder, mesh->grid decoder).

    ``out = EqLinear(transport_interpolate(x)) + GaugeConv(x -> receiver)`` followed by an
    equivariant FFN.  The receiver state is the interpolated field, optionally concatenated
    with an extra dst-side field (e.g. the grid embedding of the current state in the decoder).
    """

    def __init__(self, cs_in: int, cv_in: int, cs_out: int, cv_out: int, n_basis: int = 8, hidden: int = 64,
                 extra_cs: int = 0, extra_cv: int = 0):
        super().__init__()
        self.pre = EqLinear(cs_in, cv_in, cs_out, cv_out)
        self.extra = (extra_cs, extra_cv)
        self.conv = GaugeConv(cs_in, cv_in, cs_out + extra_cs, cv_out + extra_cv, cs_out, cv_out, n_basis, hidden)
        self.norm = EqNorm(cs_out, cv_out)
        self.ffn = EqFFN(cs_out, cv_out)

    def forward(self, x: Field, edges: EdgeSet, extra: Optional[Field] = None) -> Field:
        base = self.pre(transport_interpolate(x, edges))
        recv = base if extra is None else cat_fields([base, extra])
        f = base + self.conv(x, recv, edges)
        return f + self.ffn(self.norm(f))


class Merge(nn.Module):
    """merge(up, skip): concatenate channels, equivariant linear projection."""

    def __init__(self, cs_a: int, cv_a: int, cs_b: int, cv_b: int, cs_out: int, cv_out: int):
        super().__init__()
        self.lin = EqLinear(cs_a + cs_b, cv_a + cv_b, cs_out, cv_out)

    def forward(self, a: Field, b: Field) -> Field:
        return self.lin(cat_fields([a, b]))
