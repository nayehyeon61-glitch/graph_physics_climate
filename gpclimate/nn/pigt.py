"""Physics-Integrated Graph Transformer (PIGT) fusion block.

Design: ``docs/design/Physics_Integrated_Graph_Transformer_KO.md``.  Per resolution level the
dynamics stream ``H`` attends jointly over three token sources living on the same mesh nodes

* ``H``  the dynamics features themselves (1-ring + self),
* ``P``  the frozen physics-guide features of that level,
* ``O``  fixed, parameter-free physics operator terms (``gpclimate.nn.physics_ops``),

with one softmax over (neighbour x source), and the physics stream ``P`` is refined in turn
by attending over ``[P, H]`` (bidirectional).  Compared with the gated one-way cross-attention
of ``gpclimate.nn.attention.PhysicsFusion`` there is no scalar gate; the residual branches start
small through a down-scaled output projection, and the attention mass that ``H`` gives to every
source is recorded per layer (``last_mass``) as a built-in contribution diagnostic.

Equivariance: K/V vectors are parallel-transported into the receiver gauge, scores are built
from invariants only (scalar products, |v|-type terms, edge geometry, per-source bias), and all
projections are ``EqLinear`` -- the same ingredients as ``EqCrossAttention``.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter, softmax

from gpclimate.external import repos
from gpclimate.geometry.graphs import EdgeSet
from gpclimate.nn.gauge_conv import radial_embedding
from gpclimate.nn.so2 import EqFFN, EqLinear, EqNorm, Field, cat_fields, pack, phase, unpack


def _scale_linear(lin: EqLinear, factor: float) -> None:
    with torch.no_grad():
        if lin.lin_s is not None:
            lin.lin_s.weight.mul_(factor)
            if lin.lin_s.bias is not None:
                lin.lin_s.bias.zero_()
        if lin.w_re is not None:
            lin.w_re.mul_(factor)
            lin.w_im.mul_(factor)


class EqMultiSourceAttention(MessagePassing):
    """Receiver ``i`` attends over ``{tokens_t[j] : j in N(i) U {i}, t in sources}`` with one softmax."""

    def __init__(self, cs_q: int, cv_q: int, src_dims: Sequence[Tuple[int, int]], heads: int = 4, ds: int = 16,
                 dv: int = 4, n_basis: int = 8, loc_hidden: int = 16, out_scale: float = 0.1):
        super().__init__(aggr="add", node_dim=-2)
        self.h, self.ds, self.dv, self.n_basis = heads, ds, dv, n_basis
        self.n_src = len(src_dims)
        self.q = EqLinear(cs_q, cv_q, heads * ds, heads * dv, bias=False)
        self.k = nn.ModuleList([EqLinear(cs, cv, heads * ds, heads * dv, bias=False) for cs, cv in src_dims])
        self.v = nn.ModuleList([EqLinear(cs, cv, heads * ds, heads * dv, bias=True) for cs, cv in src_dims])
        self.type_bias = nn.Parameter(torch.zeros(self.n_src, heads))  # invariant per-source logit offset
        MultiheadLinear = repos.lietransformer_MultiheadLinear()
        Swish = repos.lieconv_Swish()
        self.loc = nn.Sequential(MultiheadLinear(heads, n_basis + 1, loc_hidden), Swish(),
                                 MultiheadLinear(heads, loc_hidden, 1))
        self.proj_e = nn.Linear(n_basis + 1, heads * (ds + dv))
        nn.init.normal_(self.proj_e.weight, std=0.02)
        nn.init.ones_(self.proj_e.bias)
        self.o = EqLinear(heads * ds, heads * dv, cs_q, cv_q)
        _scale_linear(self.o, out_scale)  # residual branch starts small (replaces the scalar gate)
        self.scale = 1.0 / math.sqrt(ds + 2 * dv)
        self.last_mass: Optional[torch.Tensor] = None  # [n_src] mean attention mass per source

    def forward(self, h: Field, sources: Sequence[Field], edges: EdgeSet) -> Field:
        assert len(sources) == self.n_src
        H, ds, dv = self.h, self.ds, self.dv
        N, E = edges.n_dst, edges.src.numel()
        q = self.q(h)
        kv = cat_fields([self.k[t](s) for t, s in enumerate(sources)] + [self.v[t](s) for t, s in enumerate(sources)])
        # tokens of all sources stacked along the node axis: node j of source t -> index t*N + j
        T = self.n_src
        src = torch.cat([edges.src + t * N for t in range(T)])
        dst = edges.dst.repeat(T)
        alpha = edges.alpha.repeat(T)
        is_self = edges.is_self if edges.is_self is not None else torch.zeros_like(edges.dist, dtype=torch.bool)
        emb = torch.cat([radial_embedding(edges.dist, edges.cutoff, self.n_basis), is_self.float()[:, None]], -1)
        emb = emb.repeat(T, 1)
        tid = torch.arange(T, device=edges.src.device).repeat_interleave(E)
        # per-source K/V live in separate channel blocks of ``kv``; token (t, j) takes the block of its source
        ks = torch.stack([kv.s[..., t * H * ds:(t + 1) * H * ds] for t in range(T)], -3)  # [B,T,N,H*ds]
        vs = torch.stack([kv.s[..., (T + t) * H * ds:(T + t + 1) * H * ds] for t in range(T)], -3)
        kvv = torch.stack([kv.v[..., t * H * dv:(t + 1) * H * dv] for t in range(T)], -3)
        vvv = torch.stack([kv.v[..., (T + t) * H * dv:(T + t + 1) * H * dv] for t in range(T)], -3)
        tokens = Field(torch.cat([ks, vs], -1).flatten(-3, -2), torch.cat([kvv, vvv], -1).flatten(-3, -2))  # [B,T*N,.]
        out = self.propagate(torch.stack([src, dst]), q=pack(q), kv=pack(tokens), alpha=alpha, emb=emb, tid=tid,
                             size=(T * N, N))
        return self.o(unpack(out, H * ds, H * dv))

    def message(self, q_i, kv_j, alpha, emb, tid, index, ptr, size_i):
        H, ds, dv = self.h, self.ds, self.dv
        fq = unpack(q_i, H * ds, H * dv)
        fkv = unpack(kv_j, 2 * H * ds, 2 * H * dv)
        k_s, v_s = fkv.s[..., :H * ds], fkv.s[..., H * ds:]
        kv_v = fkv.v * phase(alpha)[:, None]  # transport sender vectors into the receiver gauge
        k_v, v_v = kv_v[..., :H * dv], kv_v[..., H * dv:]
        e = self.proj_e(emb)
        e_s = e[:, :H * ds].unflatten(-1, (H, ds))
        e_v = e[:, H * ds:].unflatten(-1, (H, dv))
        qs, ks = fq.s.unflatten(-1, (H, ds)), k_s.unflatten(-1, (H, ds))
        qv, kvec = fq.v.unflatten(-1, (H, dv)), k_v.unflatten(-1, (H, dv))
        content = ((qs * ks * e_s).sum(-1) + ((qv.conj() * kvec).real * e_v).sum(-1)) * self.scale  # [B,E',H]
        loc = self.loc(emb[:, None, :].expand(-1, H, -1)).squeeze(-1)  # [E',H]
        score = (content + loc + self.type_bias[tid]).clamp(-5.0, 5.0)
        a = softmax(score, index, ptr, size_i, dim=self.node_dim)
        with torch.no_grad():  # average attention mass per source per receiver (sums to 1 over sources)
            m = a.detach().mean(-1)
            m = m.mean(0) if m.dim() == 2 else m
            self.last_mass = scatter(m, tid, 0, dim_size=self.n_src, reduce="sum") / size_i
        out_s = (v_s.unflatten(-1, (H, ds)) * a[..., None]).flatten(-2)
        out_v = (v_v.unflatten(-1, (H, dv)) * a[..., None]).flatten(-2)
        return pack(Field(out_s, out_v))


class PIGTLayer(nn.Module):
    """H <- H + Attn(H; [H, P, O]); H <- H + FFN(H);  P <- P + Attn(P; [P, H]); P <- P + FFN(P)."""

    def __init__(self, cs: int, cv: int, cs_p: int, cv_p: int, ops_dims: Optional[Tuple[int, int]], heads: int, ds: int,
                 dv: int, n_basis: int, update_p: bool = True):
        super().__init__()
        self.with_ops = ops_dims is not None
        self.update_p = update_p
        self.norm_h, self.norm_p = EqNorm(cs, cv), EqNorm(cs_p, cv_p)
        src = [(cs, cv), (cs_p, cv_p)]
        if self.with_ops:
            self.norm_o = EqNorm(*ops_dims)
            src.append(tuple(ops_dims))
        self.attn_h = EqMultiSourceAttention(cs, cv, src, heads, ds, dv, n_basis)
        self.norm_h2, self.ffn_h = EqNorm(cs, cv), EqFFN(cs, cv)
        if update_p:
            self.norm_hp = EqNorm(cs, cv)
            self.attn_p = EqMultiSourceAttention(cs_p, cv_p, [(cs_p, cv_p), (cs, cv)], heads, ds, dv, n_basis)
            self.norm_p2, self.ffn_p = EqNorm(cs_p, cv_p), EqFFN(cs_p, cv_p)

    def forward(self, h: Field, p: Field, o: Optional[Field], edges: EdgeSet) -> Tuple[Field, Field]:
        hn, pn = self.norm_h(h), self.norm_p(p)
        srcs = [hn, pn] + ([self.norm_o(o)] if self.with_ops else [])
        h = h + self.attn_h(hn, srcs, edges)
        h = h + self.ffn_h(self.norm_h2(h))
        if self.update_p:
            p = p + self.attn_p(pn, [pn, self.norm_hp(h)], edges)
            p = p + self.ffn_p(self.norm_p2(p))
        return h, p

    @property
    def source_mass(self) -> Optional[List[float]]:
        return None if self.attn_h.last_mass is None else self.attn_h.last_mass.tolist()


class PhysicsIntegration(nn.Module):
    """Stack of PIGT layers for one resolution level (drop-in for ``PhysicsFusion``)."""

    def __init__(self, cs: int, cv: int, cs_p: int, cv_p: int, ops_dims: Optional[Tuple[int, int]], n_layers: int = 2,
                 heads: int = 4, ds: int = 16, dv: int = 4, n_basis: int = 8, update_p: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([PIGTLayer(cs, cv, cs_p, cv_p, ops_dims, heads, ds, dv, n_basis,
                                               update_p=update_p and (i < n_layers - 1 or n_layers == 1))
                                     for i in range(n_layers)])
        self.source_names = ["H", "P"] + (["O"] if ops_dims is not None else [])

    def forward(self, h: Field, p: Field, o: Optional[Field], edges: EdgeSet) -> Field:
        for layer in self.layers:
            h, p = layer(h, p, o, edges)
        return h

    @property
    def source_mass(self) -> List[Optional[List[float]]]:
        return [layer.source_mass for layer in self.layers]
