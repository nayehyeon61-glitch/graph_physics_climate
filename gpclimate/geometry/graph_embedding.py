"""stage19b GPINN graph embedding: eigenvectors of a terrain / coastline weighted Laplacian of the grid's face graph.

GPINN (arXiv 2306.09792) adds the leading non-trivial eigenvectors of the graph Laplacian (Fiedler vector and the next
ones) to the PINN inputs, so the network sees the *topology* of the domain, which local coordinates do not give.  On a
uniform sphere these eigenvectors are spherical harmonics -- smooth functions of position that the guide already has
through its coordinates -- so the edges are weighted by how strongly two neighbouring cells are physically connected:

    w_f = (l_f / d_f) * exp(-|h_i - h_j| / h0) * (c if the land-sea mask changes across the face else 1)

(face length over centre distance = the finite-volume Laplacian, damped across steep orography and coastlines).  The
generalised problem  L v = lambda A v  (A = cell areas) is solved and the k eigenvectors after the constant one are
returned, sign-fixed (largest entry positive) and scaled to unit area-weighted RMS.  The low modes then separate
continents, ocean basins and mountain barriers -- a global positional encoding of the Earth's surface graph.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from gpclimate.nn.graph_core import GridFaceGraph
from gpclimate.nn.physics_ops import EARTH_RADIUS


def terrain_laplacian_eigvecs(lat_deg, lon_deg, orog_m, lsm, k: int = 8, h0: float = 1000.0,
                              coast: float = 0.25) -> torch.Tensor:
    """[H * W, k] eigenvectors (cells latitude-major, latitudes ascending as in ``GridFaceGraph``)."""
    lat = np.asarray(lat_deg, dtype=np.float64)
    orog, lsm = np.asarray(orog_m, dtype=np.float64), np.asarray(lsm, dtype=np.float64)
    if lat[0] > lat[-1]:  # the face graph wants ascending latitudes
        lat, orog, lsm = lat[::-1], orog[::-1], lsm[::-1]
    fg = GridFaceGraph(torch.as_tensor(lat.copy()), torch.as_tensor(np.asarray(lon_deg, dtype=np.float64)))
    H, W = fg.H, fg.W
    src, dst = fg.src.numpy(), fg.dst.numpy()
    length = fg.length.double().numpy()
    latr = np.deg2rad(lat)
    # centre distances: east faces along the latitude circle, north faces along the meridian
    d_east = EARTH_RADIUS * np.cos(latr)[:, None].repeat(W, 1).reshape(-1) * (2 * math.pi / W)
    d_north = EARTH_RADIUS * np.diff(latr)[:, None].repeat(W, 1).reshape(-1)
    dist = np.concatenate([d_east, d_north])
    h, m = orog.reshape(-1), lsm.reshape(-1)
    w = length / dist * np.exp(-np.abs(h[src] - h[dst]) / h0)
    w = w * np.where(np.abs(m[src] - m[dst]) > 0.5, coast, 1.0)
    N = H * W
    Lm = np.zeros((N, N))
    np.add.at(Lm, (src, dst), -w)
    np.add.at(Lm, (dst, src), -w)
    np.add.at(Lm, (src, src), w)
    np.add.at(Lm, (dst, dst), w)
    area = fg.area.double().numpy()
    a_isqrt = 1.0 / np.sqrt(area)
    S = Lm * a_isqrt[:, None] * a_isqrt[None, :]  # A^-1/2 L A^-1/2, symmetric
    _, vecs = np.linalg.eigh(S)
    V = vecs[:, 1:k + 1] * a_isqrt[:, None]  # back to L v = lambda A v
    for j in range(V.shape[1]):
        col = V[:, j]
        if col[np.argmax(np.abs(col))] < 0:
            V[:, j] = -col
        V[:, j] /= math.sqrt(float((V[:, j] ** 2 * area).sum() / area.sum()))
    if np.asarray(lat_deg)[0] > np.asarray(lat_deg)[-1]:  # back to the caller's latitude order
        V = V.reshape(H, W, -1)[::-1].reshape(N, -1)
    return torch.as_tensor(np.ascontiguousarray(V), dtype=torch.float32)
