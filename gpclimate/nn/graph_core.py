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
"""

from __future__ import annotations

import math

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
        self.register_buffer("length", length.float(), persistent=False)
        self.register_buffer("area", area.float(), persistent=False)
        self.register_buffer("normal", normal.to(torch.complex64), persistent=False)
        self.mean_length = float(length.mean())

    @property
    def n_cells(self) -> int:
        return self.H * self.W

    def divergence(self, J: torch.Tensor) -> torch.Tensor:
        """Face flux ``J`` [B, F, C] (per unit length, from src to dst) -> cell divergence [B, N, C]."""
        fl = J * self.length[None, :, None]
        out = scatter(fl, self.src, 1, dim_size=self.n_cells, reduce="sum")
        out = out - scatter(fl, self.dst, 1, dim_size=self.n_cells, reduce="sum")
        return out / self.area[None, :, None]

    def face_normals(self, gauge: torch.Tensor) -> torch.Tensor:
        """Face normals in the node frames of the src cells [F] (complex): canonical normal times exp(-i gauge)."""
        return self.normal * torch.exp(-1j * gauge[self.src].to(torch.float32))


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
