"""Differential geometry of the unit sphere needed for SO(2) gauge equivariance.

Conventions
-----------
* Every node ``i`` carries a unit position ``n_i`` and a right-handed tangent
  frame (gauge) ``(e1_i, e2_i)`` with ``e1 x e2 = n``.
* A tangent vector with frame coordinates ``(x, y)`` is stored as the complex
  number ``z = x + i y``.  Rotating the frame by ``theta`` (``e1' = cos e1 + sin e2``)
  changes the coordinates to ``exp(-i theta) z`` (a frequency-1 SO(2) irrep).
* For an edge ``j -> i`` (source ``j``, destination ``i``):
    - ``alpha_ij``: Levi-Civita transport along the geodesic expressed in the two
      gauges, so that ``exp(i alpha_ij) z_j`` are the coordinates in frame ``i``.
      Under gauge change ``alpha' = alpha - theta_i + theta_j``.
    - ``phi_ij``: direction of ``j`` seen from ``i`` (log map at ``i``) in frame ``i``.
      Under gauge change ``phi' = phi - theta_i``.
    - ``dist_ij``: geodesic distance (radians), gauge invariant.

The SO(2) group operations (log of the 2x2 transport matrix, lifting of a 2D
log-map vector to (angle, radius)) use ``lie_conv.lieGroups.SO2`` from the
LieConv fork.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from gpclimate.external import repos


def latlon_to_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Unit vectors for latitude/longitude in degrees (physicsnemo ``latlon2xyz`` convention)."""
    gu = repos.physicsnemo_graph_utils()
    latlon = torch.as_tensor(np.stack([np.asarray(lat_deg, np.float64).ravel(),
                                       np.asarray(lon_deg, np.float64).ravel()], -1))
    xyz = gu.latlon2xyz(latlon, radius=1, unit="deg").numpy()
    return xyz.reshape(np.shape(lat_deg) + (3,))


def xyz_to_latlon(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gu = repos.physicsnemo_graph_utils()
    v = np.asarray(xyz, np.float64).reshape(-1, 3)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    v[:, 2] = np.clip(v[:, 2], -1.0, 1.0)
    ll = gu.xyz2latlon(torch.as_tensor(v), radius=1, unit="deg").numpy()
    shp = np.shape(xyz)[:-1]
    return ll[:, 0].reshape(shp), ll[:, 1].reshape(shp)


def east_north_frames(lat_deg: np.ndarray, lon_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Local (east, north) frame from lat/lon; well defined even at the poles because
    it is parameterised by the grid point's own longitude (the ERA5 u/v convention)."""
    lat = np.deg2rad(np.asarray(lat_deg, np.float64))
    lon = np.deg2rad(np.asarray(lon_deg, np.float64))
    east = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], -1)
    north = np.stack([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)], -1)
    return east, north


def rotate_frames(e1: np.ndarray, e2: np.ndarray, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Gauge transformation: rotate each frame by ``theta`` about its normal."""
    c, s = np.cos(theta)[..., None], np.sin(theta)[..., None]
    return c * e1 + s * e2, -s * e1 + c * e2


def _skew(k: np.ndarray) -> np.ndarray:
    z = np.zeros(k.shape[:-1])
    return np.stack([
        np.stack([z, -k[..., 2], k[..., 1]], -1),
        np.stack([k[..., 2], z, -k[..., 0]], -1),
        np.stack([-k[..., 1], k[..., 0], z], -1),
    ], -2)


def transport_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrices [E,3,3] mapping T_a S^2 onto T_b S^2 along the geodesic a->b
    (Levi-Civita parallel transport). Rodrigues formula, stable for a ~= b."""
    k = np.cross(a, b)
    c = np.sum(a * b, -1)
    if np.any(c < -1 + 1e-9):
        raise ValueError("parallel transport between antipodal points is undefined")
    K = _skew(k)
    eye = np.broadcast_to(np.eye(3), K.shape)
    return eye + K + (K @ K) / (1.0 + c)[..., None, None]


def edge_geometry(
    pos_src: np.ndarray, e1_src: np.ndarray, e2_src: np.ndarray,
    pos_dst: np.ndarray, e1_dst: np.ndarray, e2_dst: np.ndarray,
    src: np.ndarray, dst: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (alpha, phi, dist) for edges src[k] -> dst[k] (float64 numpy)."""
    SO2 = repos.lieconv_SO2()()
    a, b = pos_src[src], pos_dst[dst]
    R = transport_rotation(a, b)
    t1 = np.einsum("eij,ej->ei", R, e1_src[src])
    t2 = np.einsum("eij,ej->ei", R, e2_src[src])
    f1, f2 = e1_dst[dst], e2_dst[dst]
    # 2x2 matrix of the transport in gauge coordinates; columns = images of the source basis
    M = np.stack([
        np.stack([np.sum(t1 * f1, -1), np.sum(t2 * f1, -1)], -1),
        np.stack([np.sum(t1 * f2, -1), np.sum(t2 * f2, -1)], -1),
    ], -2)
    alpha = SO2.log(torch.as_tensor(M))[..., 0].numpy()
    # log map of the source point at the destination, in destination coordinates
    cosd = np.clip(np.sum(a * b, -1), -1.0, 1.0)
    d = a - cosd[:, None] * b
    xy = np.stack([np.sum(d * f1, -1), np.sum(d * f2, -1)], -1)
    phi, _r = SO2.lifted_elems(torch.as_tensor(xy)[None])  # (1,E,1) angle, (1,E,1) radius
    phi = phi[0, :, 0].numpy()
    dist = np.arccos(cosd)
    return alpha, phi, dist


def spherical_triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Area (spherical excess) of unit-sphere triangles (Van Oosterom & Strackee)."""
    num = np.abs(np.einsum("...i,...i->...", a, np.cross(b, c)))
    den = 1.0 + np.sum(a * b, -1) + np.sum(b * c, -1) + np.sum(c * a, -1)
    return 2.0 * np.arctan2(num, den)


def vertex_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Barycentric dual areas (one third of the adjacent spherical faces); sums to 4*pi."""
    fa = spherical_triangle_area(vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]])
    area = np.zeros(len(vertices))
    for k in range(3):
        np.add.at(area, faces[:, k], fa / 3.0)
    return area


def latlon_cell_areas(lat_deg: np.ndarray, n_lon: int) -> np.ndarray:
    """Area of equiangular lat/lon cells (handles grids with/without pole rows); sums to 4*pi."""
    lat = np.deg2rad(np.asarray(lat_deg, np.float64))
    edges = np.concatenate([[-np.pi / 2], 0.5 * (lat[1:] + lat[:-1]), [np.pi / 2]])
    if lat[0] > lat[-1]:
        edges = np.concatenate([[np.pi / 2], 0.5 * (lat[1:] + lat[:-1]), [-np.pi / 2]])
    band = np.abs(np.sin(edges[1:]) - np.sin(edges[:-1])) * 2 * np.pi
    return np.repeat(band[:, None] / n_lon, n_lon, axis=1)
