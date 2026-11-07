"""Equivariant graph cross-attention used to inject physics features into the dynamics stream.

``F_l = H_l + beta_l * CrossAttention_eq(Q = H_l, K = P_l, V = P_l)``   (design doc eq. 4.2)

Structure follows ``graphtransformer/layers/graph_transformer_edge_layer.py``
(Dwivedi & Bresson, "A Generalization of Transformer Networks to Graphs"):
Q/K/V projections, scaled dot-product scores modulated element-wise by an
explicit edge projection ``proj_e``, ``clamp(-5, 5)``, normalisation over the
incoming edges, output projection ``O``, residual + norm + FFN.  That code is
written for DGL; here it is re-expressed with ``torch_geometric.nn.MessagePassing``
and ``torch_geometric.utils.softmax``.

Equivariant extensions

* K and V vector channels are parallel-transported into the receiver gauge
  (``exp(i alpha_ij)``) before they are compared / aggregated;
* the content score uses ``s_q . s_k + Re<v_q, v_k>`` (gauge invariant);
* a location kernel in the style of LieTransformer (Hutchinson et al. 2021,
  ``eqv_transformer.kernels.AttentionKernel`` with ``location_featurisation="mlp"``
  and ``location_feature_combination="sum"``) adds a per-head bias computed by a
  multi-head MLP built from ``eqv_transformer.multihead_neural.MultiheadLinear``
  and LieConv's ``Swish`` on invariant edge geometry (geodesic distance, self flag);
* gates ``beta_l`` are real per-channel scalars (invariant).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from gpclimate.external import repos
from gpclimate.geometry.graphs import EdgeSet
from gpclimate.nn.gauge_conv import radial_embedding
from gpclimate.nn.so2 import EqFFN, EqLinear, EqNorm, Field, cat_fields, pack, phase, unpack


class EqCrossAttention(MessagePassing):
    def __init__(self, cs_q: int, cv_q: int, cs_kv: int, cv_kv: int, heads: int = 4, ds: int = 16, dv: int = 4,
                 n_basis: int = 8, loc_hidden: int = 16):
        super().__init__(aggr="add", node_dim=-2)
        self.h, self.ds, self.dv, self.n_basis = heads, ds, dv, n_basis
        self.cs_q, self.cv_q = cs_q, cv_q
        self.q = EqLinear(cs_q, cv_q, heads * ds, heads * dv, bias=False)
        self.k = EqLinear(cs_kv, cv_kv, heads * ds, heads * dv, bias=False)
        self.v = EqLinear(cs_kv, cv_kv, heads * ds, heads * dv, bias=True)
        MultiheadLinear = repos.lietransformer_MultiheadLinear()
        Swish = repos.lieconv_Swish()
        self.loc = nn.Sequential(MultiheadLinear(heads, n_basis + 1, loc_hidden), Swish(),
                                 MultiheadLinear(heads, loc_hidden, 1))
        self.proj_e = nn.Linear(n_basis + 1, heads * (ds + dv))
        nn.init.normal_(self.proj_e.weight, std=0.02)
        nn.init.ones_(self.proj_e.bias)
        self.o = EqLinear(heads * ds, heads * dv, cs_q, cv_q)
        self.scale = 1.0 / math.sqrt(ds + 2 * dv)
        self.last_attention = None

    def forward(self, h: Field, p: Field, edges: EdgeSet) -> Field:
        H, ds, dv = self.h, self.ds, self.dv
        q, k, v = self.q(h), self.k(p), self.v(p)
        is_self = edges.is_self if edges.is_self is not None else torch.zeros_like(edges.dist, dtype=torch.bool)
        emb = torch.cat([radial_embedding(edges.dist, edges.cutoff, self.n_basis), is_self.float()[:, None]], -1)
        out = self.propagate(edges.edge_index, q=pack(q), kv=pack(cat_fields([k, v])), alpha=edges.alpha,
                             emb=emb, size=(edges.n_src, edges.n_dst))
        return self.o(unpack(out, H * ds, H * dv))

    def message(self, q_i, kv_j, alpha, emb, index, ptr, size_i):
        H, ds, dv = self.h, self.ds, self.dv
        fq = unpack(q_i, H * ds, H * dv)
        fkv = unpack(kv_j, 2 * H * ds, 2 * H * dv)
        k_s, v_s = fkv.s[..., :H * ds], fkv.s[..., H * ds:]
        kv_v = fkv.v * phase(alpha)[:, None]  # transport sender vectors into the receiver gauge
        k_v, v_v = kv_v[..., :H * dv], kv_v[..., H * dv:]

        e = self.proj_e(emb)  # [E, H*(ds+dv)] invariant edge modulation (graphtransformer imp_exp_attn)
        e_s = e[:, :H * ds].unflatten(-1, (H, ds))
        e_v = e[:, H * ds:].unflatten(-1, (H, dv))
        qs, ks = fq.s.unflatten(-1, (H, ds)), k_s.unflatten(-1, (H, ds))
        qv, kv = fq.v.unflatten(-1, (H, dv)), k_v.unflatten(-1, (H, dv))
        content = ((qs * ks * e_s).sum(-1) + ((qv.conj() * kv).real * e_v).sum(-1)) * self.scale  # [B,E,H]
        loc = self.loc(emb[:, None, :].expand(-1, H, -1)).squeeze(-1)  # [E,H] LieTransformer location kernel
        score = (content + loc).clamp(-5.0, 5.0)
        a = softmax(score, index, ptr, size_i, dim=self.node_dim)
        self.last_attention = a.detach()
        out_s = (v_s.unflatten(-1, (H, ds)) * a[..., None]).flatten(-2)
        out_v = (v_v.unflatten(-1, (H, dv)) * a[..., None]).flatten(-2)
        return pack(Field(out_s, out_v))


class PhysicsFusion(nn.Module):
    """Per-level fusion block: F = H + beta * Attn(norm H, norm P);  F = F + FFN(norm F)."""

    def __init__(self, cs_h: int, cv_h: int, cs_p: int, cv_p: int, heads: int = 4, ds: int = 16, dv: int = 4,
                 n_basis: int = 8, gate_init: float = 0.1):
        super().__init__()
        self.norm_h = EqNorm(cs_h, cv_h)
        self.norm_p = EqNorm(cs_p, cv_p)
        self.attn = EqCrossAttention(cs_h, cv_h, cs_p, cv_p, heads, ds, dv, n_basis)
        self.beta_s = nn.Parameter(torch.full((cs_h,), float(gate_init)))
        self.beta_v = nn.Parameter(torch.full((cv_h,), float(gate_init)))
        self.norm_f = EqNorm(cs_h, cv_h)
        self.ffn = EqFFN(cs_h, cv_h)

    def forward(self, h: Field, p: Field, edges: EdgeSet) -> Field:
        a = self.attn(self.norm_h(h), self.norm_p(p), edges)
        f = h + a.scale(self.beta_s, self.beta_v)
        return f + self.ffn(self.norm_f(f))
