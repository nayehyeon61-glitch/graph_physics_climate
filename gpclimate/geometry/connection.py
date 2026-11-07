"""Connection and curvature on the icosahedral simplicial complex -- the torch side of ``dec.py`` (stage11).

Two Lie structures meet on every mesh level:

* the **geometric connection**: the SO(2) gauge transport ``alpha_ij`` of ``SphereGraphs`` (Levi-Civita).  Its
  curvature is the holonomy around a face, which equals the face area (Gauss-Bonnet, ``tests/test_dec.py``):
  constant on the sphere, so it carries no information beyond the geometry -- it is what makes the gauge
  equivariance of the layers exact;
* the **dynamical (flow) connection**: a tangent-vector field ``u`` (wind) seen as an edge 1-form ``u^flat``
  (``flat``).  Its exterior derivative ``d1 u^flat`` is the circulation around every face = the curvature 2-form
  of the flow (relative vorticity x area); the codifferential ``delta1 u^flat`` is the divergence; both are exact
  cochain operations (``d1 d0 = 0``), unlike the least-squares stencils of ``nn.physics_ops``.  Solving the
  Poisson problems ``Lap psi = zeta`` and ``Lap chi = div`` gives the (non-local) Helmholtz potentials.

``MeshComplex`` stores the oriented edges / faces, the directions of every edge in the gauge frames of its two
end points (so ``flat`` and ``sharp`` are gauge invariant / equivariant), the diagonal Hodge stars and the
geometric holonomy, all in units of the mean edge length ``h`` like ``nn.physics_ops.Stencil`` (so outputs are
O(1) in normalised units).  ``gauge_transformed`` applies a per-node frame rotation (equivariance tests).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

import numpy as np
import torch
from torch_geometric.utils import scatter

from gpclimate.geometry import dec


def _phase(a: torch.Tensor) -> torch.Tensor:
    return torch.polar(torch.ones_like(a), a)


@dataclass
class MeshComplex:
    """Simplicial 2-complex of one mesh level with gauge-aware edge directions (all tensors on one device)."""

    level: int
    n_v: int
    edges: torch.Tensor  # [E, 2] long, i < j (oriented i -> j)
    faces: torch.Tensor  # [F, 3] long
    face_edges: torch.Tensor  # [F, 3] long
    signs: torch.Tensor  # [F, 3] float, orientation of each boundary edge in the face loop a->b->c->a
    dir_i: torch.Tensor  # [E] complex: direction of j seen from i, in the gauge of i
    dir_j: torch.Tensor  # [E] complex: direction of i seen from j, in the gauge of j
    length: torch.Tensor  # [E] geodesic length / h
    s0: torch.Tensor  # [V] vertex area / h^2
    s1: torch.Tensor  # [E] dual length / primal length
    s2: torch.Tensor  # [F] h^2 / face area
    face_area: torch.Tensor  # [F] / h^2
    holonomy: torch.Tensor  # [F] holonomy of the geometric SO(2) connection (rad; = +- face area on the sphere)
    h: float  # mean edge length [rad]
    sharp_inv: Optional[torch.Tensor] = None  # [V, 2, 2] inverse moment matrices of the vertex reconstruction
    lap_pinv: Optional[torch.Tensor] = None  # cached pseudo-inverse of d0^T S1 d0 (Poisson solves)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_level(cls, graphs, level: int) -> "MeshComplex":
        nodes, faces, e = graphs.mesh[level], graphs.mesh_faces[level], graphs.mesh_edges[level]
        d0, d1, edges = dec.coboundaries(nodes.n, faces)
        _, face_edges, signs = dec.oriented_edges(faces)
        stars = dec.hodge_stars(nodes.pos, faces, edges, face_edges, nodes.area)
        hol, _ = dec.holonomy(e.src.numpy(), e.dst.numpy(), e.alpha.double().numpy(), faces, nodes.pos)
        # directed-edge lookup (src, dst) -> phi = direction of src seen from dst, in the gauge of dst
        src, dst, phi = e.src.numpy().astype(np.int64), e.dst.numpy().astype(np.int64), e.phi.double().numpy()
        key = src * nodes.n + dst
        order = np.argsort(key)
        key_s = key[order]

        def lookup(s, d):
            k = s.astype(np.int64) * nodes.n + d
            pos = np.searchsorted(key_s, k)
            assert np.all(key_s[pos] == k), "mesh edge set must contain both directions of every edge"
            return phi[order[pos]]

        i, j = edges[:, 0], edges[:, 1]
        h = float(stars["primal_length"].mean())
        t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt)
        cx = cls(level=level, n_v=nodes.n, edges=t(edges, torch.long), faces=t(np.asarray(faces), torch.long),
                 face_edges=t(face_edges, torch.long), signs=t(signs), dir_i=_phase(t(lookup(j, i))),
                 dir_j=_phase(t(lookup(i, j))), length=t(stars["primal_length"] / h), s0=t(nodes.area / h ** 2),
                 s1=t(stars["s1"]), s2=t(stars["face_area"] * 0 + h ** 2 / stars["face_area"]),
                 face_area=t(stars["face_area"] / h ** 2), holonomy=t(hol), h=h)
        cx.sharp_inv = cx._build_sharp()
        return cx

    @property
    def n_e(self) -> int:
        return self.edges.shape[0]

    @property
    def n_f(self) -> int:
        return self.faces.shape[0]

    def _build_sharp(self) -> torch.Tensor:
        """Per-vertex 2x2 moment matrices of the least-squares reconstruction of a vector from the edge 1-forms."""
        w = 1.0 / (self.length ** 2)
        di, dj = torch.view_as_real(self.dir_i), torch.view_as_real(self.dir_j)  # [E, 2]
        M = scatter(w[:, None, None] * di[:, :, None] * di[:, None, :], self.edges[:, 0], 0, dim_size=self.n_v,
                    reduce="sum")
        M = M + scatter(w[:, None, None] * dj[:, :, None] * dj[:, None, :], self.edges[:, 1], 0, dim_size=self.n_v,
                        reduce="sum")
        return torch.linalg.inv(M + 1e-6 * torch.eye(2))

    def gauge_transformed(self, theta: torch.Tensor) -> "MeshComplex":
        """The same complex after rotating every vertex frame by ``theta`` [V] (directions turn by -theta)."""
        th = theta.to(self.dir_i.real.dtype)
        cx = replace(self, dir_i=self.dir_i * _phase(-th[self.edges[:, 0]]),
                     dir_j=self.dir_j * _phase(-th[self.edges[:, 1]]), lap_pinv=self.lap_pinv)
        cx.sharp_inv = cx._build_sharp()
        return cx

    def to(self, device) -> "MeshComplex":
        mv = lambda x: None if x is None else x.to(device)
        return replace(self, edges=mv(self.edges), faces=mv(self.faces), face_edges=mv(self.face_edges),
                       signs=mv(self.signs), dir_i=mv(self.dir_i), dir_j=mv(self.dir_j), length=mv(self.length),
                       s0=mv(self.s0), s1=mv(self.s1), s2=mv(self.s2), face_area=mv(self.face_area),
                       holonomy=mv(self.holonomy), sharp_inv=mv(self.sharp_inv), lap_pinv=mv(self.lap_pinv))

    # ------------------------------------------------------------------ cochain calculus  ([B, cells, C] tensors)
    def d0(self, s: torch.Tensor) -> torch.Tensor:
        """0-form -> 1-form: (d0 s)_ij = s_j - s_i."""
        return s.index_select(1, self.edges[:, 1]) - s.index_select(1, self.edges[:, 0])

    def d0T(self, w: torch.Tensor) -> torch.Tensor:
        return (scatter(w, self.edges[:, 1], 1, dim_size=self.n_v, reduce="sum")
                - scatter(w, self.edges[:, 0], 1, dim_size=self.n_v, reduce="sum"))

    def d1(self, w: torch.Tensor) -> torch.Tensor:
        """1-form -> 2-form: signed sum of the three boundary edges of every face (circulation)."""
        B, C = w.shape[0], w.shape[-1]
        return (w.index_select(1, self.face_edges.reshape(-1)).reshape(B, -1, 3, C) * self.signs[None, :, :, None]).sum(2)

    def d1T(self, c: torch.Tensor) -> torch.Tensor:
        B, C = c.shape[0], c.shape[-1]
        vals = (c[:, :, None, :] * self.signs[None, :, :, None]).reshape(B, -1, C)
        return scatter(vals, self.face_edges.reshape(-1), 1, dim_size=self.n_e, reduce="sum")

    def delta1(self, w: torch.Tensor) -> torch.Tensor:
        """Codifferential of a 1-form, delta1 = S0^-1 d0^T S1 (= minus the vertex divergence, so that L0 = delta1 d0
        is the positive Laplacian)."""
        return self.d0T(w * self.s1[None, :, None]) / self.s0[None, :, None]

    def laplacian0(self, s: torch.Tensor) -> torch.Tensor:
        """Positive Hodge Laplacian of a 0-form, L0 = delta1 d0 (= -Lap)."""
        return self.delta1(self.d0(s))

    # ------------------------------------------------------------------ vectors <-> 1-forms (gauge aware)
    def flat(self, v: torch.Tensor) -> torch.Tensor:
        """Tangent-vector field (complex [B, V, C], components in the vertex gauges) -> edge 1-form [B, E, C]:
        average of the tangential components at the two end points times the edge length.  Gauge invariant."""
        ci = (self.dir_i.conj()[None, :, None] * v.index_select(1, self.edges[:, 0])).real
        cj = (self.dir_j.conj()[None, :, None] * v.index_select(1, self.edges[:, 1])).real
        return 0.5 * self.length[None, :, None] * (ci - cj)

    def sharp(self, w: torch.Tensor) -> torch.Tensor:
        """Edge 1-form -> vertex vector field (least squares over the incident edges; left inverse of ``flat`` for
        linear fields).  Equivariant: the result is expressed in the vertex gauge."""
        val = w / self.length[None, :, None]  # component along the edge
        wt = (1.0 / self.length ** 2)[:, None]
        di, dj = torch.view_as_real(self.dir_i), torch.view_as_real(self.dir_j)
        b = scatter(val[..., None] * (wt * di)[None, :, None, :], self.edges[:, 0], 1, dim_size=self.n_v, reduce="sum")
        b = b + scatter(-val[..., None] * (wt * dj)[None, :, None, :], self.edges[:, 1], 1, dim_size=self.n_v,
                        reduce="sum")
        g = torch.einsum("nij,bncj->bnci", self.sharp_inv, b)
        return torch.view_as_complex(g.contiguous())

    # ------------------------------------------------------------------ curvature of the flow connection
    def curvature(self, v: torch.Tensor) -> torch.Tensor:
        """Circulation of ``v`` around every face = curvature 2-form of the flow connection [B, F, C]."""
        return self.d1(self.flat(v))

    def face_to_vertex(self, c: torch.Tensor, density: bool = True) -> torch.Tensor:
        """2-form on faces -> vertex value: sum over the incident faces divided by their total area (``density``)
        or by their number."""
        B, C = c.shape[0], c.shape[-1]
        idx = self.faces.reshape(-1)
        num = scatter(c[:, :, None, :].expand(B, -1, 3, C).reshape(B, -1, C), idx, 1, dim_size=self.n_v, reduce="sum")
        wf = self.face_area if density else torch.ones_like(self.face_area)
        den = scatter(wf[:, None].expand(-1, 3).reshape(-1), idx, 0, dim_size=self.n_v, reduce="sum")
        return num / den[None, :, None]

    def vorticity(self, v: torch.Tensor) -> torch.Tensor:
        """Relative vorticity at the vertices from the exact circulation (Stokes over the incident faces)."""
        return self.face_to_vertex(self.curvature(v))

    def divergence(self, v: torch.Tensor) -> torch.Tensor:
        """Divergence at the vertices from the exact codifferential (outflow of the dual cell / its area)."""
        return -self.delta1(self.flat(v))

    # ------------------------------------------------------------------ Poisson solves (Helmholtz potentials)
    def poisson_pinv(self) -> torch.Tensor:
        """Pseudo-inverse of A0 = d0^T S1 d0 (symmetric PSD, kernel = constants), cached, float32 [V, V]."""
        if self.lap_pinv is None:
            D0 = torch.zeros(self.n_e, self.n_v, dtype=torch.float64, device=self.edges.device)
            ar = torch.arange(self.n_e, device=self.edges.device)
            D0[ar, self.edges[:, 1]] = 1.0
            D0[ar, self.edges[:, 0]] = -1.0
            A0 = D0.T @ (self.s1.double()[:, None] * D0)
            self.lap_pinv = torch.linalg.pinv(A0, hermitian=True, rtol=1e-8).float()
        return self.lap_pinv

    def poisson(self, rhs: torch.Tensor) -> torch.Tensor:
        """Zero-mean solution x of ``Lap x = rhs`` (Lap = -L0) for vertex fields rhs [B, V, C]."""
        return -torch.einsum("mn,bnc->bmc", self.poisson_pinv(), rhs * self.s0[None, :, None])

    def potentials(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Streamfunction psi (Lap psi = zeta) and velocity potential chi (Lap chi = div), both [B, V, C]."""
        return self.poisson(self.vorticity(v)), self.poisson(self.divergence(v))
