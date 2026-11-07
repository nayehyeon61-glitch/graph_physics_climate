"""Fixed, exactly SO(2)-equivariant differential operators on the spherical graph.

These are the first-order building blocks of the primitive equations, evaluated
with a weighted least-squares stencil on the 1-ring of every mesh node:

* gradient of a scalar          ``grad s``            (vector, frequency 1)
* divergence / curl of a vector ``div v``, ``curl v``   (invariant scalars)
* scalar advection              ``u . grad s``        (scalar)
* vector self-advection         ``(u . grad) u``      (vector)
* Coriolis term                 ``f k x u = i f u``   (vector), ``f = sin(lat)``
* absolute vorticity            ``zeta_a = curl u / (2 Omega) + sin(lat)`` and its advection,
  stretching ``zeta_a div u``   (barotropic vorticity equation terms)
* graph Laplacian of scalars

The least-squares gradient at node ``i`` solves ``min_g sum_j w_ij |g . d_ij - (s_j - s_i)|^2``
with ``d_ij`` the log-map vector from ``i`` to neighbour ``j`` in the gauge of ``i``
(``dist_ij * exp(i phi_ij)``) and ``w_ij = 1/|d_ij|^2``.  For vectors the neighbour
value is parallel-transported first (``exp(i alpha_ij) v_j``) and the 2x2 Jacobian
is fitted the same way; its trace and antisymmetric part are gauge invariant and
``J u`` transforms as a vector.  Every output is therefore equivariant by
construction (no learned parameters), so the physics guide receives the
dynamical terms instead of having to rediscover them from neighbour differences.
Lengths are measured in units of the mean edge length of the level, so all
outputs are O(1) in normalised units; the vorticity/Coriolis ratio uses the
physical wind scale when it is known.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch_geometric.utils import scatter

from gpclimate.geometry.graphs import EdgeSet
from gpclimate.nn.so2 import Field, phase

EARTH_RADIUS = 6.371e6
OMEGA = 7.292e-5


class Stencil:
    """Per-edge vectors and per-node inverse moment matrices for one EdgeSet."""

    def __init__(self, edges: EdgeSet):
        e = edges if bool(edges.directional.all()) else edges.subset(edges.directional)
        self.src, self.dst, self.n = e.src, e.dst, e.n_dst
        self.h = float(e.dist.mean())
        dvec = (e.dist / self.h)[:, None] * torch.view_as_real(phase(e.phi))  # [E, 2] log map, unit = mean edge
        self.d = dvec
        self.w = 1.0 / (dvec * dvec).sum(-1).clamp(min=1e-6)  # [E]
        self.alpha = phase(e.alpha)  # transport src gauge -> dst gauge
        M = scatter(self.w[:, None, None] * dvec[:, :, None] * dvec[:, None, :], self.dst, 0, dim_size=self.n,
                    reduce="sum")  # [N, 2, 2]
        self.Minv = torch.linalg.inv(M + 1e-6 * torch.eye(2))
        self.wsum = scatter(self.w, self.dst, 0, dim_size=self.n, reduce="sum").clamp(min=1e-6)

    def grad(self, s: torch.Tensor) -> torch.Tensor:
        """s [B, N, C] -> complex gradient [B, N, C]."""
        diff = s.index_select(1, self.src) - s.index_select(1, self.dst)  # [B, E, C]
        b = scatter(diff[..., None] * (self.w[:, None] * self.d)[None, :, None, :], self.dst, 1, dim_size=self.n,
                    reduce="sum")  # [B, N, C, 2]
        g = torch.einsum("nij,bncj->bnci", self.Minv, b)
        return torch.view_as_complex(g.contiguous())

    def laplacian(self, s: torch.Tensor) -> torch.Tensor:
        diff = s.index_select(1, self.src) - s.index_select(1, self.dst)
        return scatter(diff * self.w[None, :, None], self.dst, 1, dim_size=self.n, reduce="sum") / self.wsum[None, :, None]

    def jacobian(self, v: torch.Tensor) -> torch.Tensor:
        """v complex [B, N, C] -> Jacobian [B, N, C, 2, 2] in the receiver gauge (rows: component, cols: direction)."""
        vj = v.index_select(1, self.src) * self.alpha[None, :, None]
        delta = torch.view_as_real(vj - v.index_select(1, self.dst))  # [B, E, C, 2]
        Cm = scatter(delta[..., :, None] * (self.w[:, None] * self.d)[None, :, None, None, :], self.dst, 1,
                     dim_size=self.n, reduce="sum")  # [B, N, C, 2, 2]  sum w Delta d^T
        return torch.einsum("bncij,njk->bncik", Cm, self.Minv)


def _dot(u: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """u . g for complex 2-vectors (invariant)."""
    return (u.conj() * g).real


class PhysicsOperators(nn.Module):
    """Derived physics fields on a mesh level from (state scalars, state vectors, condition scalars).

    Output channel counts (n_s scalars, n_v vectors, n_c condition scalars):
      scalars: div(n_v) + curl(n_v) + u.grad s (n_v*n_s) + u.grad c (n_v*n_c) + u.grad zeta_a (n_v)
               + zeta_a div u (n_v) + lap s (n_s)
      vectors: grad s (n_s) + grad c (n_c) + grad zeta_a (n_v) + (u.grad)u (n_v) + i f u (n_v)
    """

    def __init__(self, n_s: int, n_v: int, n_c: int, v_scale_mps: Optional[torch.Tensor] = None):
        super().__init__()
        self.n_s, self.n_v, self.n_c = n_s, n_v, n_c
        # physical scale (m/s) of each normalised wind channel; a buffer so checkpoints carry it
        self.register_buffer("v_scale", torch.ones(n_v) if v_scale_mps is None else torch.as_tensor(v_scale_mps).float())

    @property
    def out_dims(self) -> Tuple[int, int]:
        n_s, n_v, n_c = self.n_s, self.n_v, self.n_c
        return (2 * n_v + n_v * n_s + n_v * n_c + 2 * n_v + n_s, n_s + n_c + 3 * n_v)

    @property
    def scalar_names(self) -> Tuple[str, ...]:
        names = [f"div{k}" for k in range(self.n_v)] + [f"curl{k}" for k in range(self.n_v)]
        names += [f"adv{k}_s{j}" for k in range(self.n_v) for j in range(self.n_s)]
        names += [f"adv{k}_c{j}" for k in range(self.n_v) for j in range(self.n_c)]
        names += [f"adv{k}_zeta" for k in range(self.n_v)] + [f"stretch{k}" for k in range(self.n_v)]
        names += [f"lap_s{j}" for j in range(self.n_s)]
        return tuple(names)

    def forward(self, s: torch.Tensor, v: torch.Tensor, c: torch.Tensor, sinlat: torch.Tensor, edges: EdgeSet) -> Field:
        st = Stencil(edges)
        B, N = s.shape[:2]
        f = sinlat.to(s.dtype)[None, :, None].expand(B, N, 1)  # Coriolis parameter / (2 Omega)
        c = c.expand(B, N, -1) if c.shape[0] != B else c
        grad_s = st.grad(s)  # [B,N,n_s] complex
        grad_c = st.grad(c) if self.n_c > 0 else s.new_zeros(B, N, 0).to(torch.complex64)
        J = st.jacobian(v)  # [B,N,n_v,2,2]
        div = J[..., 0, 0] + J[..., 1, 1]
        curl = J[..., 1, 0] - J[..., 0, 1]
        # curl in physical s^-1 is curl_norm * v_scale / (R * h); absolute vorticity in units of 2 Omega
        kappa = self.v_scale / (EARTH_RADIUS * st.h * 2 * OMEGA)
        zeta_a = curl * kappa.to(curl.dtype) + f  # [B,N,n_v]
        grad_zeta = st.grad(zeta_a)
        u = v  # [B,N,n_v] complex
        adv_s = _dot(u[..., :, None], grad_s[..., None, :]).flatten(-2)  # [B,N,n_v*n_s]
        adv_c = _dot(u[..., :, None], grad_c[..., None, :]).flatten(-2) if self.n_c > 0 else s.new_zeros(B, N, 0)
        adv_zeta = _dot(u, grad_zeta)  # [B,N,n_v]
        stretch = zeta_a * div
        lap = st.laplacian(s)
        ur = torch.view_as_real(u)  # [B,N,n_v,2]
        self_adv = torch.view_as_complex(torch.einsum("bncij,bncj->bnci", J, ur).contiguous())
        coriolis = 1j * f * u
        scal = torch.cat([div, curl, adv_s, adv_c, adv_zeta, stretch, lap], -1)
        vec = torch.cat([grad_s, grad_c, grad_zeta, self_adv, coriolis], -1)
        return Field(scal, vec)
