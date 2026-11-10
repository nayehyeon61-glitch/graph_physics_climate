"""stage18 graph tendency core: finite-volume face graph of the lat-lon grid and a conservative learned flux.

The PG-Flow tendency core (nayehyeon61-glitch/physical_guide_flowmatching, ``docs/era5_15day_ko.md``,
``SphericalTransport2D`` + ``SphericalFluxSourceClosure2D``) writes the rate of every channel as

    dq/dt = -div(V q) + K lap q - div(J_theta) + S_theta,      V re-evaluated at every right-hand-side call,

with the learned closure split into a flux divergence (changes nothing in the area integral) and a source.  Here the
same core lives on a graph: the cells of the 64x32 grid are the nodes, the cell faces are the edges, and

    div_i(J) = ( sum_{faces f leaving i} J_f l_f  -  sum_{faces f entering i} J_f l_f ) / A_i

with face lengths ``l`` and cell areas ``A`` of the sphere, so ``sum_i A_i div_i(J) = 0`` exactly for any face flux
(polar caps are closed: no face crosses a pole).  The flux of a face is read from the two cells' embeddings by an
antisymmetric map  J(i -> j) = psi(f_i(n), f_j(n)) - psi(f_j(-n), f_i(-n))  where ``f(n)`` holds the scalar features
and the components of the vector features along and across the face normal ``n`` (pointing from i to j), so the flux
is gauge invariant and J(j -> i) = -J(i -> j) by construction: message passing on the face graph *is* a
conservative finite-volume flux.  The transport part of the core (-div(V q) with the guide's flow) is done by the
semi-Lagrangian step of stage17c (``gpclimate.nn.semi_lagrangian``), because an explicit flux form at a 6 h sub-step
violates the CFL limit by an order of magnitude near the poles of a 5.6 deg grid (PG-Flow sub-cycles at minutes).

stage18d Laplace block (PhyMPGN, Zeng et al. ICLR 2025, github.com/intell-sci-comput/PhyMPGN): the diffusion term
K lap q of the PG-Flow core, missing in stage18, on the same covering control volumes.  PhyMPGN's discrete
Laplace-Beltrami  lap f_i = (1/d_i) sum_j w_ij (f_j - f_i)  uses the area d_i of the polygon that covers node i
(circumcentres around it) and the cotangent weights w_ij = (cot a + cot b) / 2 = dual length / primal length.  On this
orthogonal grid the covering polygon is the cell (area A_i) and the weight is face length / centre distance
(``weight = l / d``), so ``laplacian`` is that operator.  The learned part is the diffusivity K >= 0 per cell and
channel (PhyMPGN's learnable correction of the operator itself is the antisymmetric flux J_theta here).  An explicit
6 h diffusion step is unstable near the poles (east faces of a polar cell are 30 km apart), so the face number
nu = K w dt / min(A_i, A_j) is capped smoothly at ``nu_max`` (0.12): with four faces per cell the update is a convex
combination of the cell and its neighbours (no new extrema) and the area integral is conserved exactly.

stage18e edge-preserving diffusion (Kovac & Smith 2009, arXiv 0911.1928, "Regression on a graph"): their fit
minimises  sum_i (y_i - f_i)^2 + lambda sum_edges |f_j - f_i|,  an L1 (total-variation) penalty on the edge differences
whose minimisers are piecewise constant: jumps survive, noise is removed.  The gradient flow of the eps-regularised
penalty  sum_f sqrt((q_j - q_i)^2 + eps^2)  is a diffusion with the face diffusivity  K eps / sqrt(dq^2 + eps^2):
linear diffusion where |dq| << eps and a bounded flux  K eps sign(dq)  across a jump.  ``diffusion(..., edge_eps)``
multiplies the learned K by this factor (``edge_factor`` <= 1, so the stability cap and the conservation are unchanged);
eps is a per-variable scale (the median neighbour difference of the training states, times a learned factor).

stage18h energy-based partition (감자 2026-11-10): the power diagram of stage18g (``deformed``) with weights from the
energy of the current state instead of static fields, a_i = -sum_c alpha_c log(e_c,i / mean e_c + eps): with alpha > 0
the cells shrink where the energy is high (MPAS SCVT density function / r-adaptive equidistribution, here with the
generators fixed).  The partition is re-drawn every sub-step and differs per sample (``a`` [B, N]), so it is used with
``fixed_areas`` (cell changes accumulated in the grid cells: exact conservation of the grid integral).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from torch_geometric.utils import scatter

from gpclimate.nn.physics_ops import EARTH_RADIUS
from gpclimate.nn.so2 import Field


class GridFaceGraph(nn.Module):
    """Faces of the regular lat-lon grid (cells ordered latitude-major, ``index = i_lat * W + j_lon``)."""

    def __init__(self, lat_deg: torch.Tensor, lon_deg: torch.Tensor):
        super().__init__()
        lat = torch.deg2rad(torch.as_tensor(lat_deg, dtype=torch.float64))
        lon = torch.as_tensor(lon_deg, dtype=torch.float64)
        H, W = len(lat), len(lon)
        if H < 2 or not bool((lat[1:] > lat[:-1]).all()):
            raise ValueError("latitudes must be ascending")
        dlon = 2 * math.pi / W
        if abs(float(torch.deg2rad(lon[1] - lon[0])) - dlon) > 1e-6:
            raise ValueError("longitudes must be equally spaced and cover the full circle")
        self.H, self.W = H, W
        # cell bounds: midpoints between centres, the outermost bounds at the poles
        mid = 0.5 * (lat[1:] + lat[:-1])
        bounds = torch.cat([lat.new_tensor([-math.pi / 2]), mid, lat.new_tensor([math.pi / 2])])
        area_row = EARTH_RADIUS ** 2 * dlon * (bounds[1:].sin() - bounds[:-1].sin())  # [H]
        area = area_row[:, None].expand(H, W).reshape(-1)
        idx = torch.arange(H * W).view(H, W)
        # east faces: (i, j) -> (i, j + 1) (periodic), length = meridional extent of the cell row
        e_src, e_dst = idx.reshape(-1), torch.roll(idx, -1, dims=1).reshape(-1)
        e_len = (EARTH_RADIUS * (bounds[1:] - bounds[:-1]))[:, None].expand(H, W).reshape(-1)
        # north faces: (i, j) -> (i + 1, j), length = zonal extent at the shared bound
        n_src, n_dst = idx[:-1].reshape(-1), idx[1:].reshape(-1)
        n_len = (EARTH_RADIUS * dlon * mid.cos())[:, None].expand(H - 1, W).reshape(-1)
        self.register_buffer("src", torch.cat([e_src, n_src]), persistent=False)
        self.register_buffer("dst", torch.cat([e_dst, n_dst]), persistent=False)
        length = torch.cat([e_len, n_len])
        # face normal (from src to dst) in the canonical east/north frame: east faces 1, north faces i
        normal = torch.cat([torch.ones(len(e_src), dtype=torch.complex128), torch.full((len(n_src),), 1j)])
        # stage18d: centre distances (east: along the latitude circle, north: along the meridian) and the cotangent
        # (covering) weights l / d of the finite-volume Laplace-Beltrami operator
        e_dist = (EARTH_RADIUS * lat.cos() * dlon)[:, None].expand(H, W).reshape(-1)
        n_dist = (EARTH_RADIUS * (lat[1:] - lat[:-1]))[:, None].expand(H - 1, W).reshape(-1)
        dist = torch.cat([e_dist, n_dist])
        src_all, dst_all = torch.cat([e_src, n_src]), torch.cat([e_dst, n_dst])
        self.register_buffer("weight", (length / dist).float(), persistent=False)
        self.register_buffer("face_mass", torch.minimum(area[src_all], area[dst_all]).float(), persistent=False)
        self.register_buffer("length", length.float(), persistent=False)
        self.register_buffer("area", area.float(), persistent=False)
        self.register_buffer("normal", normal.to(torch.complex64), persistent=False)
        self.mean_length = float(length.mean())
        # stage18f / stage18g: per-cell centre spacings (zonal dx, meridional extent dy), face latitude cosine and type
        self.n_east = len(e_src)
        self.register_buffer("dx", e_dist.float(), persistent=False)
        self.register_buffer("dy", e_len.float(), persistent=False)
        face_cos = torch.cat([lat.cos()[:, None].expand(H, W).reshape(-1), mid.cos()[:, None].expand(H - 1, W).reshape(-1)])
        self.register_buffer("face_cos", face_cos.float(), persistent=False)
        self.register_buffer("face_type", torch.cat([torch.ones(len(e_src)), torch.zeros(len(n_src))]), persistent=False)

    @property
    def n_cells(self) -> int:
        return self.H * self.W

    @staticmethod
    def _bc(t: torch.Tensor) -> torch.Tensor:
        """A geometry tensor [N] / [F] (shared) or [B, N] / [B, F] (per sample, stage18h) -> broadcastable to [B, ., C]."""
        return t[None, :, None] if t.dim() == 1 else t[:, :, None]

    def divergence(self, J: torch.Tensor) -> torch.Tensor:
        """Face flux ``J`` [B, F, C] (per unit length, from src to dst) -> cell divergence [B, N, C]."""
        fl = J * self._bc(self.length)
        out = scatter(fl, self.src, 1, dim_size=self.n_cells, reduce="sum")
        out = out - scatter(fl, self.dst, 1, dim_size=self.n_cells, reduce="sum")
        return out / self._bc(self.area)

    def _exchange(self, F: torch.Tensor) -> torch.Tensor:
        """Face exchange ``F`` [B, F, C] (amount moved from dst into src) -> cell change [B, N, C] (area-conserving)."""
        out = scatter(F, self.src, 1, dim_size=self.n_cells, reduce="sum")
        out = out - scatter(F, self.dst, 1, dim_size=self.n_cells, reduce="sum")
        return out / self._bc(self.area)

    def laplacian(self, q: torch.Tensor) -> torch.Tensor:
        """stage18d: finite-volume Laplace-Beltrami of ``q`` [B, N, C] (1/m^2):  (1/A_i) sum_f (l_f/d_f)(q_j - q_i)."""
        dq = q.index_select(1, self.dst) - q.index_select(1, self.src)
        return self._exchange(self._bc(self.weight) * dq)

    def diffusion(self, q: torch.Tensor, K: torch.Tensor, dt: float, nu_max: float = 0.12,
                  edge_eps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """stage18d: change of ``q`` [B, N, C] by one explicit diffusion step of ``dt`` seconds with the face
        diffusivity ``K`` [B, F, C] (m^2/s, >= 0).  For small K it is  dt K lap q;  the face number
        nu = K (l/d) dt / min(A_i, A_j) is capped smoothly at ``nu_max`` (stable, monotone, area integral conserved).
        stage18e: with ``edge_eps`` [C] (> 0, units of q) the face diffusivity is multiplied by the edge-preserving
        factor ``edge_factor``: linear diffusion across differences << eps, a bounded flux across fronts (the
        eps-regularised gradient flow of the total variation sum_f |q_j - q_i|, Kovac & Smith 2009)."""
        dq = q.index_select(1, self.dst) - q.index_select(1, self.src)
        if edge_eps is not None:
            K = K * self.edge_factor(dq, edge_eps)
        x = K * self._bc(self.weight * dt / self.face_mass)
        nu = nu_max * torch.tanh(x / nu_max)
        return self._exchange(nu * self._bc(self.face_mass) * dq)

    @staticmethod
    def edge_factor(dq: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """stage18e: eps / sqrt(dq^2 + eps^2) in (0, 1] for the face differences ``dq`` [B, F, C] and the edge scale
        ``eps`` [C]:  1 - O(dq^2 / eps^2) for small differences, eps / |dq| across a jump (flux eps sign dq)."""
        return eps / torch.sqrt(dq * dq + eps * eps)

    def face_mean(self, c: torch.Tensor) -> torch.Tensor:
        """Symmetric face value of a cell field [B, N, C] -> [B, F, C]."""
        return 0.5 * (c.index_select(1, self.src) + c.index_select(1, self.dst))

    def face_normals(self, gauge: torch.Tensor) -> torch.Tensor:
        """Face normals in the node frames of the src cells [F] (complex): canonical normal times exp(-i gauge)."""
        return self.normal * torch.exp(-1j * gauge[self.src].to(torch.float32))

    def deformed(self, a: torch.Tensor, max_shift: float = 0.25, fixed_areas: bool = False) -> "GridFaceGraph":
        """stage18g (theta_g): the power diagram of the grid points with the cell weights ``a`` [N] (dimensionless),
        keeping the Delaunay connectivity.  The bisector of i and j moves from the midpoint towards the lighter cell
        by  delta_f = max_shift tanh(a_i - a_j) d_f  along the face normal (power distance |x - x_i|^2 - w_i, shift
        (w_i - w_j) / 2 d_ij, here bounded); the cell areas (zonal and meridional stretch, first order in the shifts,
        total area kept), the face lengths (mean stretch of the two cells), the covering weights l / d (centres fixed)
        and the face masses are recomputed, so every operator of the returned graph conserves the integral with
        respect to the new areas.  ``a = 0`` gives the grid itself.

        stage18h: ``a`` may be [B, N] (a partition per sample, re-drawn every sub-step from the energy of the state).
        A partition that changes in time cannot conserve the integral over its own, changing areas from one sub-step to
        the next, so with ``fixed_areas`` the cell changes are accumulated in the fixed grid cells (the change of the
        new cell's content is handed back to the generator's grid cell, ALE style) and the face masses of the stability
        cap stay those of the grid: only the face lengths and the covering weights l / d of the new partition act, and
        sum_i A_i dq_i = 0 exactly for the grid areas A.  The new areas are kept in ``deformation`` for diagnostics."""
        N, ne = self.n_cells, self.n_east
        d = a.dim() - 1  # cell / face axis
        shift = max_shift * torch.tanh(a.index_select(d, self.src) - a.index_select(d, self.dst))  # delta_f / d_f
        delta = shift * self.length / self.weight  # metres, towards dst (the src cell grows)
        de, dn = delta[..., :ne], delta[..., ne:]
        sx = (scatter(de, self.src[:ne], d, dim_size=N, reduce="sum")
              - scatter(de, self.dst[:ne], d, dim_size=N, reduce="sum")) / self.dx
        sy = (scatter(dn, self.src[ne:], d, dim_size=N, reduce="sum")
              - scatter(dn, self.dst[ne:], d, dim_size=N, reduce="sum")) / self.dy
        area = self.area * (1 + sx) * (1 + sy)
        area = area * (self.area.sum() / area.sum(-1, keepdim=True))
        s_e = 0.5 * (sy.index_select(d, self.src[:ne]) + sy.index_select(d, self.dst[:ne]))  # east faces: meridional
        s_n = 0.5 * (sx.index_select(d, self.src[ne:]) + sx.index_select(d, self.dst[ne:]))  # north faces: zonal
        length = self.length * (1 + torch.cat([s_e, s_n], -1))
        g = GridFaceGraph.__new__(GridFaceGraph)
        nn.Module.__init__(g)
        g.H, g.W, g.n_east, g.mean_length = self.H, self.W, ne, self.mean_length
        for name in ("src", "dst", "normal", "dx", "dy", "face_cos", "face_type"):
            setattr(g, name, getattr(self, name))
        g.length = length
        g.weight = length / (self.length / self.weight)
        if fixed_areas:
            g.area, g.face_mass = self.area, self.face_mass
        else:
            g.area = area
            g.face_mass = torch.minimum(area.index_select(d, self.src), area.index_select(d, self.dst))
        # MPAS / SCVT diagnostic (stage22 reference): a centroidal Voronoi cell has its generator at the mass centroid;
        # here the generators stay and the faces move, so the centroid moves by half the sum of the two opposite face
        # shifts (east faces: src = i is its east face, dst = i its west face, both shifted eastwards by delta)
        cxs = 0.5 * (scatter(de, self.src[:ne], d, dim_size=N, reduce="sum")
                     + scatter(de, self.dst[:ne], d, dim_size=N, reduce="sum")) / self.dx
        cys = 0.5 * (scatter(dn, self.src[ne:], d, dim_size=N, reduce="sum")
                     + scatter(dn, self.dst[ne:], d, dim_size=N, reduce="sum")) / self.dy
        g.deformation = {"shift": shift, "sx": sx, "sy": sy, "area_ratio": area / self.area,
                         "centroid": torch.sqrt(cxs ** 2 + cys ** 2)}  # centroid offset / cell size
        return g


class EdgeConductance(nn.Module):
    """stage18f (theta_p): symmetric (kappa_ij = kappa_ji), gauge-invariant log-multiplier of the face diffusivity
    from the two cells' embeddings and physical edge features p_f, zero at initialisation (multiplier 1).  The
    dual-space physics of the user's boxed law  du_i/dt = (1/|V_i|) sum_j (|F_ij| / d_ij) kappa_theta(u_i, u_j, p_ij)
    (u_j - u_i):  the geometry |F_ij| / d_ij / |V_i| is the face graph's, kappa_theta is this head times the cell
    diffusivity of the Laplace block."""
    N_FIXED = 2  # face latitude cosine, face type

    def __init__(self, cs: int, cv: int, n_out: int, n_static: int, n_flow: int, hidden: int = 64):
        super().__init__()
        self.n_static, self.n_flow, self.n_out = n_static, n_flow, n_out
        self.n_phys = n_static + 2 * n_flow + n_out + self.N_FIXED
        self.n_in = 2 * (cs + 2 * cv) + self.n_phys
        self.psi = nn.Sequential(nn.Linear(self.n_in, hidden), nn.SiLU(), nn.Linear(hidden, n_out))
        nn.init.zeros_(self.psi[-1].weight)
        nn.init.zeros_(self.psi[-1].bias)

    def phys_features(self, faces: GridFaceGraph, q: torch.Tensor, flow: torch.Tensor, static: torch.Tensor,
                      gauge: torch.Tensor) -> torch.Tensor:
        """p_f [B, F, n_phys]: |difference| of the static fields [B, N, n_static] (terrain, land-sea), the flow
        [B, N, n_flow] (complex, m/s, node frames) across and along the face (|mean of the two cells' projections| per
        level group, in 10 m/s), |q_j - q_i| of the diffused channels [B, N, n_out], the face latitude cosine and the
        face type (1 east, 0 north).  Every entry is invariant under i <-> j."""
        src, dst = faces.src, faces.dst
        rot = torch.exp(-1j * gauge.to(torch.float32))
        proj = 0.5 * (flow.index_select(1, src) * (faces.normal * rot[src]).conj()[None, :, None]
                      + flow.index_select(1, dst) * (faces.normal * rot[dst]).conj()[None, :, None]) / 10.0
        B, F = proj.shape[:2]
        fixed = torch.stack([faces.face_cos, faces.face_type], -1)[None].expand(B, F, self.N_FIXED)
        return torch.cat([(static.index_select(1, dst) - static.index_select(1, src)).abs(), proj.real.abs(),
                          proj.imag.abs(), (q.index_select(1, dst) - q.index_select(1, src)).abs(), fixed], -1)

    def forward(self, emb: Field, faces: GridFaceGraph, gauge: torch.Tensor, phys: torch.Tensor) -> torch.Tensor:
        """log kappa multiplier [B, F, n_out] in [-4, 4]; invariant under (i <-> j, n -> -n)."""
        rot = torch.exp(-1j * gauge.to(torch.float32))
        fi = FaceFluxHead._side(emb, faces.src, faces.normal * rot[faces.src])
        fj = FaceFluxHead._side(emb, faces.dst, faces.normal * rot[faces.dst])
        cs = emb.s.shape[-1]
        same, diff = fi + fj, fi - fj  # scalars: sum and |difference|; vector projections flip with n: |.| of both
        feats = torch.cat([same[..., :cs], diff[..., :cs].abs(), same[..., cs:].abs(), diff[..., cs:].abs(), phys], -1)
        return self.psi(feats).clamp(-4.0, 4.0)


class FaceFluxHead(nn.Module):
    """Antisymmetric, gauge-invariant face flux from per-cell embeddings (zero at initialisation)."""

    def __init__(self, cs: int, cv: int, n_out: int, hidden: int = 64):
        super().__init__()
        self.n_in = 2 * (cs + 2 * cv)
        self.psi = nn.Sequential(nn.Linear(self.n_in, hidden), nn.SiLU(), nn.Linear(hidden, n_out))
        nn.init.zeros_(self.psi[-1].weight)
        nn.init.zeros_(self.psi[-1].bias)

    @staticmethod
    def _side(emb: Field, idx: torch.Tensor, n_node: torch.Tensor) -> torch.Tensor:
        """f(n) of the cells ``idx``: scalars, then the vector features along / across ``n`` (in that cell's frame)."""
        s = emb.s.index_select(1, idx)
        v = emb.v.index_select(1, idx) * n_node.conj()[None, :, None]
        return torch.cat([s, v.real, v.imag], -1)

    def forward(self, emb: Field, faces: GridFaceGraph, gauge: torch.Tensor) -> torch.Tensor:
        """``emb`` [B, N, cs] / [B, N, cv] -> J [B, F, n_out], J(src -> dst) = -J(dst -> src)."""
        rot = torch.exp(-1j * gauge.to(torch.float32))  # canonical -> node frame
        n_src = faces.normal * rot[faces.src]
        n_dst = faces.normal * rot[faces.dst]  # same normal (src -> dst) seen from the dst cell
        fi, fj = self._side(emb, faces.src, n_src), self._side(emb, faces.dst, n_dst)
        cs = emb.s.shape[-1]

        def flip(f):  # f(-n): the vector projections change sign
            return torch.cat([f[..., :cs], -f[..., cs:]], -1)

        return self.psi(torch.cat([fi, fj], -1)) - self.psi(torch.cat([flip(fj), flip(fi)], -1))
