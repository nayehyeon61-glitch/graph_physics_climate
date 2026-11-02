"""Multi-resolution spherical graphs: lat/lon grid + nested icosahedral meshes.

The icosahedral hierarchy and the face->edge conversion come from physicsnemo's
GraphCast utilities (``icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere``
and ``faces_to_edges``).  On top of them we attach per-node gauges and per-edge
gauge data (transport angle ``alpha``, direction ``phi``, geodesic distance).

Edge sets (all ``src -> dst``):

* ``mesh_edges[l]``  - 1-ring of mesh level ``l`` (message passing)
* ``attn_edges[l]``  - 1-ring + self loops of level ``l`` (physics cross-attention)
* ``down[l]``        - level ``l`` -> ``l+1`` (area-weighted pooling, with transport)
* ``up[l]``          - level ``l+1`` -> ``l`` (inverse-distance unpooling, with transport)
* ``grid2mesh``      - lat/lon grid -> finest mesh (encoder adapter)
* ``mesh2grid``      - finest mesh -> lat/lon grid (decoder adapter)

Level 0 is the finest mesh, as in the design document (Fine -> Medium -> Coarse).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence

import numpy as np
import torch
from scipy.spatial import cKDTree

from gpclimate.external import repos
from gpclimate.geometry import sphere

DIR_EPS = 1e-7  # edges shorter than this have no well-defined direction


@dataclass
class NodeSet:
    pos: np.ndarray  # [N,3] float64 unit vectors
    e1: np.ndarray  # [N,3] gauge axis 1
    e2: np.ndarray  # [N,3] gauge axis 2 (e1 x e2 = pos)
    area: np.ndarray  # [N] steradians
    lat: np.ndarray
    lon: np.ndarray

    @property
    def n(self) -> int:
        return len(self.pos)

    def with_gauge(self, theta: np.ndarray) -> "NodeSet":
        e1, e2 = sphere.rotate_frames(self.e1, self.e2, theta)
        return replace(self, e1=e1, e2=e2)


@dataclass
class EdgeSet:
    src: torch.Tensor  # [E] long
    dst: torch.Tensor  # [E] long
    alpha: torch.Tensor  # [E] transport angle src-gauge -> dst-gauge
    phi: torch.Tensor  # [E] direction of src seen from dst (dst gauge); 0 where undefined
    dist: torch.Tensor  # [E] geodesic distance [rad]
    n_src: int
    n_dst: int
    cutoff: float  # radial-basis cutoff for this edge type
    weight: Optional[torch.Tensor] = None  # [E] interpolation weights (sum to 1 per dst)
    is_self: Optional[torch.Tensor] = None  # [E] bool self loops (attention)

    @property
    def edge_index(self) -> torch.Tensor:
        return torch.stack([self.src, self.dst], 0)

    @property
    def directional(self) -> torch.Tensor:
        return self.dist > DIR_EPS

    def subset(self, mask: torch.Tensor) -> "EdgeSet":
        def sel(t):
            return None if t is None else t[mask]

        return replace(self, src=self.src[mask], dst=self.dst[mask], alpha=self.alpha[mask],
                       phi=self.phi[mask], dist=self.dist[mask], weight=sel(self.weight),
                       is_self=sel(self.is_self))

    def gauge_transformed(self, theta_src: torch.Tensor, theta_dst: torch.Tensor) -> "EdgeSet":
        """Edge data after rotating node gauges by theta (closed form, see sphere.py)."""
        ts, td = theta_src[self.src], theta_dst[self.dst]
        return replace(self, alpha=self.alpha - td + ts, phi=self.phi - td)

    def to(self, device) -> "EdgeSet":
        def mv(t):
            return None if t is None else t.to(device)

        return replace(self, src=mv(self.src), dst=mv(self.dst), alpha=mv(self.alpha), phi=mv(self.phi),
                       dist=mv(self.dist), weight=mv(self.weight), is_self=mv(self.is_self))


def _make_edges(src_nodes: NodeSet, dst_nodes: NodeSet, src: np.ndarray, dst: np.ndarray,
                cutoff: float, weight: Optional[np.ndarray] = None,
                is_self: Optional[np.ndarray] = None) -> EdgeSet:
    alpha, phi, dist = sphere.edge_geometry(src_nodes.pos, src_nodes.e1, src_nodes.e2,
                                            dst_nodes.pos, dst_nodes.e1, dst_nodes.e2, src, dst)
    phi = np.where(dist > DIR_EPS, phi, 0.0)
    f = torch.float32
    return EdgeSet(
        src=torch.as_tensor(src, dtype=torch.long), dst=torch.as_tensor(dst, dtype=torch.long),
        alpha=torch.as_tensor(alpha, dtype=f), phi=torch.as_tensor(phi, dtype=f),
        dist=torch.as_tensor(dist, dtype=f), n_src=src_nodes.n, n_dst=dst_nodes.n, cutoff=float(cutoff),
        weight=None if weight is None else torch.as_tensor(weight, dtype=f),
        is_self=None if is_self is None else torch.as_tensor(is_self, dtype=torch.bool),
    )


def _normalize_per_dst(w: np.ndarray, dst: np.ndarray, n_dst: int) -> np.ndarray:
    tot = np.zeros(n_dst)
    np.add.at(tot, dst, w)
    return w / tot[dst]


def _radius_or_knn(src_pos: np.ndarray, dst_pos: np.ndarray, radius_rad: float, k: int):
    """dst <- src pairs within a geodesic radius, unioned with the k nearest src nodes."""
    tree = cKDTree(src_pos)
    chord = 2.0 * np.sin(radius_rad / 2.0)
    pairs = set()
    for i, nb in enumerate(tree.query_ball_point(dst_pos, r=chord)):
        pairs.update((j, i) for j in nb)
    if k > 0:
        _, nn = tree.query(dst_pos, k=k)
        nn = np.asarray(nn).reshape(len(dst_pos), -1)
        for i in range(len(dst_pos)):
            pairs.update((int(j), i) for j in nn[i])
    arr = np.array(sorted(pairs), dtype=np.int64)
    return arr[:, 0], arr[:, 1]


def _knn(src_pos: np.ndarray, dst_pos: np.ndarray, k: int):
    tree = cKDTree(src_pos)
    d, nn = tree.query(dst_pos, k=k)
    nn = np.asarray(nn).reshape(len(dst_pos), k)
    dst = np.repeat(np.arange(len(dst_pos)), k)
    return nn.ravel().astype(np.int64), dst


def _geodesic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arccos(np.clip(np.sum(a * b, -1), -1.0, 1.0))


class SphereGraphs:
    """Grid + 3-level icosahedral hierarchy with gauges and all edge sets.

    Parameters
    ----------
    lat_deg, lon_deg:
        1-D grid coordinates (any order; poles allowed, e.g. the 121x240 KAI grid).
    mesh_splits:
        icosahedral refinement of each level, finest first, e.g. ``(4, 3, 2)`` =
        2562 / 642 / 162 nodes.
    """

    def __init__(self, lat_deg: Sequence[float], lon_deg: Sequence[float],
                 mesh_splits: Sequence[int] = (4, 3, 2), g2m_k: int = 4, down_k: int = 7):
        mesh_splits = tuple(int(s) for s in mesh_splits)
        if list(mesh_splits) != sorted(mesh_splits, reverse=True):
            raise ValueError("mesh_splits must be finest first, strictly decreasing")
        self.mesh_splits = mesh_splits
        self.lat = np.asarray(lat_deg, np.float64)
        self.lon = np.asarray(lon_deg, np.float64)
        self.n_lat, self.n_lon = len(self.lat), len(self.lon)

        # ---- grid nodes (row-major: lat slow, lon fast); gauge = (east, north) = ERA5 u/v frame
        LAT, LON = np.meshgrid(self.lat, self.lon, indexing="ij")
        gpos = sphere.latlon_to_xyz(LAT.ravel(), LON.ravel())
        ge, gn = sphere.east_north_frames(LAT.ravel(), LON.ravel())
        garea = sphere.latlon_cell_areas(self.lat, self.n_lon).ravel()
        self.grid = NodeSet(gpos, ge, gn, garea, LAT.ravel(), LON.ravel())

        # ---- icosahedral levels from physicsnemo (GraphCast) utilities
        ico = repos.physicsnemo_icosahedral_mesh()
        hierarchy = ico.get_hierarchy_of_triangular_meshes_for_sphere(splits=max(mesh_splits))
        self.mesh: List[NodeSet] = []
        self.mesh_faces: List[np.ndarray] = []
        self.mesh_edge_len: List[float] = []
        self.mesh_edges: List[EdgeSet] = []
        self.attn_edges: List[EdgeSet] = []
        for s in mesh_splits:
            m = hierarchy[s]
            v = np.asarray(m.vertices, np.float64)
            v /= np.linalg.norm(v, axis=-1, keepdims=True)
            lat, lon = sphere.xyz_to_latlon(v)
            e1, e2 = sphere.east_north_frames(lat, lon)
            nodes = NodeSet(v, e1, e2, sphere.vertex_areas(v, m.faces), lat, lon)
            snd, rcv = ico.faces_to_edges(np.asarray(m.faces))
            pairs = np.unique(np.stack([snd, rcv], 1), axis=0)
            elen = _geodesic(v[pairs[:, 0]], v[pairs[:, 1]])
            self.mesh.append(nodes)
            self.mesh_faces.append(np.asarray(m.faces))
            self.mesh_edge_len.append(float(elen.mean()))
            cut = 1.5 * float(elen.max())
            self.mesh_edges.append(_make_edges(nodes, nodes, pairs[:, 0], pairs[:, 1], cut))
            loops = np.arange(nodes.n)
            a_src = np.concatenate([pairs[:, 0], loops])
            a_dst = np.concatenate([pairs[:, 1], loops])
            is_self = np.concatenate([np.zeros(len(pairs), bool), np.ones(nodes.n, bool)])
            self.attn_edges.append(_make_edges(nodes, nodes, a_src, a_dst, cut, is_self=is_self))

        # ---- pooling / unpooling between consecutive levels (area weights + transport)
        self.down: List[EdgeSet] = []
        self.up: List[EdgeSet] = []
        for l in range(len(mesh_splits) - 1):
            fine, coarse = self.mesh[l], self.mesh[l + 1]
            hc = self.mesh_edge_len[l + 1]
            src, dst = _radius_or_knn(fine.pos, coarse.pos, 0.6 * hc, down_k)
            d = _geodesic(fine.pos[src], coarse.pos[dst])
            w = fine.area[src] * np.exp(-((d / (0.5 * hc)) ** 2))
            self.down.append(_make_edges(fine, coarse, src, dst, 1.5 * float(d.max()),
                                         weight=_normalize_per_dst(w, dst, coarse.n)))
            src, dst = _knn(coarse.pos, fine.pos, 3)
            d = _geodesic(coarse.pos[src], fine.pos[dst])
            w = 1.0 / (d + 1e-3 * hc)
            self.up.append(_make_edges(coarse, fine, src, dst, 1.5 * float(d.max()),
                                       weight=_normalize_per_dst(w, dst, fine.n)))

        # ---- grid <-> finest mesh adapters
        m0, h0 = self.mesh[0], self.mesh_edge_len[0]
        src, dst = _radius_or_knn(self.grid.pos, m0.pos, 0.6 * h0, g2m_k)
        d = _geodesic(self.grid.pos[src], m0.pos[dst])
        w = self.grid.area[src] * np.exp(-((d / (0.5 * h0)) ** 2))
        self.grid2mesh = _make_edges(self.grid, m0, src, dst, 1.5 * float(d.max()),
                                     weight=_normalize_per_dst(w, dst, m0.n))
        src, dst = _knn(m0.pos, self.grid.pos, 3)
        d = _geodesic(m0.pos[src], self.grid.pos[dst])
        w = 1.0 / (d + 1e-3 * h0)
        self.mesh2grid = _make_edges(m0, self.grid, src, dst, 1.5 * float(d.max()),
                                     weight=_normalize_per_dst(w, dst, self.grid.n))

        self.grid_area_weight = torch.as_tensor(self.grid.area / self.grid.area.mean(), dtype=torch.float32)

    # ------------------------------------------------------------------ helpers
    @property
    def n_levels(self) -> int:
        return len(self.mesh)

    def summary(self) -> dict:
        return {
            "grid": [self.n_lat, self.n_lon],
            "mesh_nodes": [m.n for m in self.mesh],
            "mesh_edges": [int(e.src.numel()) for e in self.mesh_edges],
            "mean_edge_deg": [float(np.rad2deg(h)) for h in self.mesh_edge_len],
            "grid2mesh_edges": int(self.grid2mesh.src.numel()),
            "mesh2grid_edges": int(self.mesh2grid.src.numel()),
            "down_edges": [int(e.src.numel()) for e in self.down],
            "up_edges": [int(e.src.numel()) for e in self.up],
        }

    def edge_sets(self) -> dict:
        out = {"grid2mesh": self.grid2mesh, "mesh2grid": self.mesh2grid}
        for l in range(self.n_levels):
            out[f"mesh{l}"] = self.mesh_edges[l]
            out[f"attn{l}"] = self.attn_edges[l]
        for l in range(self.n_levels - 1):
            out[f"down{l}"] = self.down[l]
            out[f"up{l}"] = self.up[l]
        return out

    def gauge_transformed(self, theta_mesh: Sequence[torch.Tensor],
                          theta_grid: Optional[torch.Tensor] = None) -> "GraphView":
        """Edge sets after per-node gauge rotations (used for equivariance tests and T_g augmentation)."""
        tg = torch.zeros(self.grid.n) if theta_grid is None else theta_grid
        tm = list(theta_mesh)
        v = GraphView.from_graphs(self)
        v.grid2mesh = self.grid2mesh.gauge_transformed(tg, tm[0])
        v.mesh2grid = self.mesh2grid.gauge_transformed(tm[0], tg)
        v.mesh_edges = [e.gauge_transformed(tm[l], tm[l]) for l, e in enumerate(self.mesh_edges)]
        v.attn_edges = [e.gauge_transformed(tm[l], tm[l]) for l, e in enumerate(self.attn_edges)]
        v.down = [e.gauge_transformed(tm[l], tm[l + 1]) for l, e in enumerate(self.down)]
        v.up = [e.gauge_transformed(tm[l + 1], tm[l]) for l, e in enumerate(self.up)]
        return v

    def view(self, device=None) -> "GraphView":
        v = GraphView.from_graphs(self)
        return v.to(device) if device is not None else v


@dataclass
class GraphView:
    """The torch-side part of SphereGraphs that the networks consume."""

    grid2mesh: EdgeSet
    mesh2grid: EdgeSet
    mesh_edges: List[EdgeSet]
    attn_edges: List[EdgeSet]
    down: List[EdgeSet]
    up: List[EdgeSet]
    n_grid: int
    n_mesh: List[int]
    grid_area_weight: torch.Tensor
    extras: dict = field(default_factory=dict)

    @classmethod
    def from_graphs(cls, g: SphereGraphs) -> "GraphView":
        return cls(g.grid2mesh, g.mesh2grid, list(g.mesh_edges), list(g.attn_edges), list(g.down), list(g.up),
                   g.grid.n, [m.n for m in g.mesh], g.grid_area_weight)

    def to(self, device) -> "GraphView":
        return GraphView(self.grid2mesh.to(device), self.mesh2grid.to(device),
                         [e.to(device) for e in self.mesh_edges], [e.to(device) for e in self.attn_edges],
                         [e.to(device) for e in self.down], [e.to(device) for e in self.up],
                         self.n_grid, list(self.n_mesh), self.grid_area_weight.to(device), dict(self.extras))
