"""Discrete exterior calculus (DEC) on the icosahedral simplicial complex.

Every mesh level of ``SphereGraphs`` is a triangulated sphere, i.e. a simplicial 2-complex with vertices, edges and
triangular faces.  This module exposes the topological and metric operators that the "homology" augmentation of the
five-layer plan (docs/design/Five_Layer_Geometric_Stochastic_GNN_Plan_KO.md, section 8) needs:

* coboundary (incidence) matrices ``d0: C^0 -> C^1`` and ``d1: C^1 -> C^2`` with ``d1 @ d0 = 0`` exactly;
* Betti numbers from their ranks -- on S^2 they are (1, 0, 1): ``L1`` has no kernel, so an edge 1-form (a tangent
  vector field) is gradient + curl with **no harmonic component**; the harmonic "hole" term of the proposal only
  exists on domains with holes;
* diagonal Hodge stars from primal geodesic lengths, dual (circumcentric) lengths, face areas and vertex areas, the
  Hodge Laplacians ``L0, L1, L2`` and a Helmholtz-Hodge decomposition of 1-forms (exact ``div curl = 0``,
  ``curl grad = 0`` identities, unlike the 1-ring least-squares stencils of ``gpclimate.nn.physics_ops``);
* the holonomy of the SO(2) gauge connection around every face, which by Gauss-Bonnet must equal the face's
  spherical area (the connection is the Levi-Civita connection) and sums to 4*pi over the sphere.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from gpclimate.geometry.sphere import spherical_triangle_area


def oriented_edges(faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unique undirected edges ``(i < j)`` of a triangle mesh and, per face, the edge index and orientation sign of
    its three boundary edges ``(a,b), (b,c), (c,a)`` (+1 when the loop traverses the edge from the smaller to the
    larger vertex index)."""
    faces = np.asarray(faces, dtype=np.int64)
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    loops = np.stack([np.stack([a, b], 1), np.stack([b, c], 1), np.stack([c, a], 1)], 1)  # [F, 3, 2]
    und = np.sort(loops, axis=-1).reshape(-1, 2)
    edges, inv = np.unique(und, axis=0, return_inverse=True)
    face_edges = inv.reshape(len(faces), 3)
    signs = np.where(loops[..., 0] < loops[..., 1], 1, -1).astype(np.int64)
    return edges, face_edges, signs


def coboundaries(n_vertices: int, faces: np.ndarray) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """``d0`` [E, V] (edge i<j: -1 at i, +1 at j) and ``d1`` [F, E] (oriented face boundary); ``d1 @ d0 == 0``."""
    edges, face_edges, signs = oriented_edges(faces)
    E, F = len(edges), len(faces)
    rows = np.repeat(np.arange(E), 2)
    cols = edges.reshape(-1)
    vals = np.tile(np.array([-1.0, 1.0]), E)
    d0 = sp.csr_matrix((vals, (rows, cols)), shape=(E, n_vertices))
    d1 = sp.csr_matrix((signs.reshape(-1).astype(float), (np.repeat(np.arange(F), 3), face_edges.reshape(-1))),
                       shape=(F, E))
    return d0, d1, edges


def betti_numbers(d0: sp.spmatrix, d1: sp.spmatrix) -> Tuple[int, int, int]:
    """(b0, b1, b2) from the ranks of the coboundaries (dense SVD; use on a coarse level)."""
    r0 = np.linalg.matrix_rank(d0.toarray())
    r1 = np.linalg.matrix_rank(d1.toarray())
    V, E, F = d0.shape[1], d0.shape[0], d1.shape[0]
    return int(V - r0), int(E - r0 - r1), int(F - r1)


def _circumcenters(v: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    n = np.cross(b - a, c - a)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    # the circumcentre of a spherical triangle on the unit sphere is the (outward) unit normal of the plane
    n *= np.sign(np.sum(n * (a + b + c), -1, keepdims=True))
    return n


def hodge_stars(vertices: np.ndarray, faces: np.ndarray, edges: np.ndarray, face_edges: np.ndarray,
                vertex_area: np.ndarray) -> Dict[str, np.ndarray]:
    """Diagonal Hodge stars: ``s0`` = dual (vertex) area, ``s1`` = dual length / primal length,
    ``s2`` = 1 / face area (unit sphere, radians)."""
    v = np.asarray(vertices, np.float64)
    prim = np.arccos(np.clip(np.sum(v[edges[:, 0]] * v[edges[:, 1]], -1), -1, 1))
    cc = _circumcenters(v, faces)
    # dual edge = geodesic between the circumcentres of the two faces sharing the edge
    owner = -np.ones((len(edges), 2), dtype=np.int64)
    for f in range(len(faces)):
        for e in face_edges[f]:
            owner[e, 0 if owner[e, 0] < 0 else 1] = f
    assert (owner >= 0).all(), "open mesh: every edge must have two faces"
    dual = np.arccos(np.clip(np.sum(cc[owner[:, 0]] * cc[owner[:, 1]], -1), -1, 1))
    fa = spherical_triangle_area(v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]])
    return {"s0": np.asarray(vertex_area, np.float64), "s1": dual / prim, "s2": 1.0 / fa, "face_area": fa,
            "primal_length": prim, "dual_length": dual}


def hodge_laplacians(d0: sp.spmatrix, d1: sp.spmatrix, stars: Dict[str, np.ndarray]):
    """Metric Hodge Laplacians ``L0`` (vertices), ``L1`` (edges), ``L2`` (faces), with codifferentials
    ``delta1 = s0^-1 d0^T s1`` and ``delta2 = s1^-1 d1^T s2``."""
    S0, S1, S2 = (sp.diags(stars[k]) for k in ("s0", "s1", "s2"))
    S0i, S1i = sp.diags(1.0 / stars["s0"]), sp.diags(1.0 / stars["s1"])
    delta1 = S0i @ d0.T @ S1  # C^1 -> C^0
    delta2 = S1i @ d1.T @ S2  # C^2 -> C^1
    L0 = delta1 @ d0
    L1 = d0 @ delta1 + delta2 @ d1
    L2 = d1 @ delta2
    return L0.tocsr(), L1.tocsr(), L2.tocsr(), delta1.tocsr(), delta2.tocsr()


def helmholtz_hodge(u: np.ndarray, d0: sp.spmatrix, d1: sp.spmatrix, stars: Dict[str, np.ndarray],
                    tol: float = 1e-10) -> Dict[str, np.ndarray]:
    """Split an edge 1-form ``u`` into ``d0 chi`` (gradient), ``delta2 psi`` (curl) and the harmonic remainder.

    ``chi`` solves ``L0 chi = delta1 u`` and ``psi`` solves ``L2 psi = d1 u`` (both singular by one constant; the
    conjugate gradient iteration stays in the orthogonal complement).  On S^2 the remainder is zero up to solver
    tolerance because ``b1 = 0``.
    """
    # the Hodge Laplacians are self-adjoint in the star inner products, not symmetric as matrices, so solve the
    # symmetrised (star-weighted) normal equations with conjugate gradients instead
    S1, S1i = sp.diags(stars["s1"]), sp.diags(1.0 / stars["s1"])
    A0 = (d0.T @ S1 @ d0).tocsr()  # = S0 L0, symmetric PSD, kernel = constants
    b0 = d0.T @ (stars["s1"] * u)
    chi, _ = spla.cg(A0, b0, rtol=tol, maxiter=20000) if np.linalg.norm(b0) > 0 else (np.zeros(d0.shape[1]), 0)
    A2 = (d1 @ S1i @ d1.T).tocsr()  # phi = S2 psi: A2 phi = d1 u, symmetric PSD
    b2 = d1 @ u
    phi, _ = spla.cg(A2, b2, rtol=tol, maxiter=20000) if np.linalg.norm(b2) > 0 else (np.zeros(d1.shape[0]), 0)
    psi = phi / stars["s2"]
    grad, curl = d0 @ chi, S1i @ (d1.T @ phi)  # delta2 psi = S1^-1 d1^T S2 psi = S1^-1 d1^T phi
    return {"chi": chi, "psi": psi, "grad": grad, "curl": curl, "harmonic": u - grad - curl}


def holonomy(src: np.ndarray, dst: np.ndarray, alpha: np.ndarray, faces: np.ndarray,
             vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Holonomy angle of the SO(2) connection around each oriented face loop ``a -> b -> c -> a`` (wrapped to
    (-pi, pi]) and the face's spherical area.  ``alpha`` follows ``sphere.edge_geometry``: for a directed edge
    ``j -> i`` the transported coordinates in frame ``i`` are ``exp(i alpha) z_j``.  Gauss-Bonnet: the holonomy of
    the Levi-Civita connection equals +-(area) with one sign for all consistently oriented faces."""
    table = {(int(s), int(d)): float(a) for s, d, a in zip(src, dst, alpha)}
    faces = np.asarray(faces, dtype=np.int64)
    hol = np.empty(len(faces))
    for f, (a, b, c) in enumerate(faces):
        hol[f] = table[(a, b)] + table[(b, c)] + table[(c, a)]
    hol = (hol + np.pi) % (2 * np.pi) - np.pi
    v = np.asarray(vertices, np.float64)
    area = spherical_triangle_area(v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]])
    return hol, area
