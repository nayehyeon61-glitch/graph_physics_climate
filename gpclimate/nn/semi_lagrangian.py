"""Semi-Lagrangian transport on the sphere with parallel transport of vectors (stage17c, Lie transport guide).

A flow is given as a tangent displacement per arrival grid point over one time step, ``d = dx e_east + dy e_north``
(radians on the unit sphere, complex ``dx + i dy``).  The departure point is the point the great circle through the
arrival point with tangent ``-d`` reaches after the arc length ``|d|``:

    q = p cos|d| - (d / |d|) sin|d|

Scalars are interpolated at ``q`` (bilinear on the regular lat-lon grid, periodic in longitude, clamped at the
outermost latitude rows).  Vectors are interpolated in the departure frame and then parallel-transported along the
great circle to the arrival frame: the angle between the vector and the geodesic tangent is preserved, so the complex
components rotate by ``exp(i (theta_a - theta_d))`` with ``theta`` the bearing (from east, counter-clockwise) of the
geodesic at each end.  The update is unconditionally stable (no CFL limit), which is what the explicit 12 h drift of
stage8b lacked.  Everything is differentiable w.r.t. the transported fields and the displacement.

Vectors are expected in the canonical geographic (east, north) frame; callers rotate gauge-transformed fields back
first (``Field.gauge(-grid_gauge)``), as the stage13 neural operator does.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class SphereSemiLagrangian(nn.Module):
    def __init__(self, lat_deg: torch.Tensor, lon_deg: torch.Tensor):
        super().__init__()
        lat = torch.deg2rad(torch.as_tensor(lat_deg, dtype=torch.float64))
        lon = torch.deg2rad(torch.as_tensor(lon_deg, dtype=torch.float64))
        if not bool((lat[1:] > lat[:-1]).all()):
            raise ValueError("latitudes must be ascending")
        dlon = torch.diff(lon)
        if not torch.allclose(dlon, dlon[0].expand_as(dlon)) or abs(float(dlon[0]) * len(lon) - 2 * torch.pi) > 1e-6:
            raise ValueError("longitudes must be equally spaced and cover the full circle")
        dlat = torch.diff(lat)
        if not torch.allclose(dlat, dlat[0].expand_as(dlat), atol=1e-9):
            raise ValueError("latitudes must be equally spaced")
        self.H, self.W = len(lat), len(lon)
        self.lat0, self.dlat = float(lat[0]), float(dlat[0])
        self.lon0, self.dlon = float(lon[0]), float(dlon[0])
        LAT, LON = torch.meshgrid(lat, lon, indexing="ij")
        p = torch.stack([LAT.cos() * LON.cos(), LAT.cos() * LON.sin(), LAT.sin()], -1)
        east = torch.stack([-LON.sin(), LON.cos(), torch.zeros_like(LON)], -1)
        north = torch.stack([-LAT.sin() * LON.cos(), -LAT.sin() * LON.sin(), LAT.cos()], -1)
        self.register_buffer("p", p.float(), persistent=False)
        self.register_buffer("east", east.float(), persistent=False)
        self.register_buffer("north", north.float(), persistent=False)

    def departure(self, disp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``disp`` [B, H, W] complex (radians).  Returns departure latitude, longitude [B, H, W] (radians) and the
        parallel-transport rotation [B, H, W] (complex, |r| = 1) from the departure to the arrival frame."""
        dx, dy = disp.real, disp.imag
        D = dx[..., None] * self.east + dy[..., None] * self.north  # tangent at p, |D| = arc length
        delta = torch.sqrt(dx * dx + dy * dy + 1e-24)
        sinc = torch.sinc(delta / torch.pi)  # sin(delta) / delta, smooth at 0
        q = self.p * delta.cos()[..., None] - D * sinc[..., None]
        q = q / q.norm(dim=-1, keepdim=True)
        lat_d = torch.asin(q[..., 2].clamp(-1.0, 1.0))
        lon_d = torch.atan2(q[..., 1], q[..., 0])
        # geodesic tangent (times delta) at the departure point, heading towards p: p delta sin(delta) + D cos(delta)
        t = self.p * (delta * delta.sin())[..., None] + D * delta.cos()[..., None]
        clat, slat, clon, slon = lat_d.cos(), lat_d.sin(), lon_d.cos(), lon_d.sin()
        te = -t[..., 0] * slon + t[..., 1] * clon
        tn = -t[..., 0] * slat * clon - t[..., 1] * slat * slon + t[..., 2] * clat
        a = torch.complex(dx, dy)  # tangent at the arrival point (bearing theta_a)
        b = torch.complex(te, tn)  # tangent at the departure point (bearing theta_d)
        den = a.abs() * b.abs()
        r = torch.where(den > 1e-20, a * b.conj() / den.clamp_min(1e-20), torch.ones_like(a))
        return lat_d, lon_d, r

    def sample(self, img: torch.Tensor, lat_d: torch.Tensor, lon_d: torch.Tensor) -> torch.Tensor:
        """Bilinear values of ``img`` [B, C, H, W] (real) at the departure points [B, H, W]."""
        B = img.shape[0]
        padded = torch.cat([img[..., -1:], img, img[..., :1]], -1)  # periodic longitude
        j = torch.remainder(lon_d - self.lon0, 2 * torch.pi) / self.dlon + 1.0  # column in the padded image
        i = (lat_d - self.lat0) / self.dlat
        grid = torch.stack([2.0 * j / (self.W + 1) - 1.0, 2.0 * i / (self.H - 1) - 1.0], -1)
        return F.grid_sample(padded, grid.to(img.dtype).expand(B, -1, -1, -1), mode="bilinear",
                             padding_mode="border", align_corners=True)

    def forward(self, scal: torch.Tensor, vec: torch.Tensor, disp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transport scalars [B, Cs, H, W] (real) and vectors [B, Cv, H, W] (complex, east + i north) by one step of
        the displacement ``disp`` [B, H, W] (complex, radians)."""
        lat_d, lon_d, r = self.departure(disp)
        cs = scal.shape[1]
        img = torch.cat([scal, vec.real, vec.imag], 1)
        out = self.sample(img, lat_d, lon_d)
        s_out = out[:, :cs]
        v_out = torch.complex(out[:, cs:cs + vec.shape[1]], out[:, cs + vec.shape[1]:]) * r[:, None]
        return s_out, v_out
