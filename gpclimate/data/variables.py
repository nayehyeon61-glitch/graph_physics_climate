"""Variable bookkeeping: which channels are gauge scalars and which form tangent vectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from gpclimate.nn.so2 import Field

Pair = Tuple[str, str]


@dataclass(frozen=True)
class VariableSpec:
    """Channel roles.

    ``vectors`` are (eastward, northward) component pairs, e.g. ("u850", "v850");
    inside the model each pair is one complex tangent-vector channel ``u + i v``
    expressed in the local (east, north) gauge of the grid point.
    """

    scalars: Tuple[str, ...]
    vectors: Tuple[Pair, ...]
    static_scalars: Tuple[str, ...] = ()
    static_vectors: Tuple[Pair, ...] = ()
    forcing_scalars: Tuple[str, ...] = ()

    @property
    def n_s(self) -> int:
        return len(self.scalars)

    @property
    def n_v(self) -> int:
        return len(self.vectors)

    @property
    def n_cond_s(self) -> int:
        return len(self.static_scalars) + len(self.forcing_scalars)

    @property
    def n_cond_v(self) -> int:
        return len(self.static_vectors)

    @property
    def aux_names(self) -> List[str]:
        """Physics diagnostics used by the Stage-A auxiliary head: vorticity & divergence per wind pair."""
        out = []
        for u, _v in self.vectors:
            out += [f"vort_{u[1:] or u}", f"div_{u[1:] or u}"]
        return out

    def state_channels(self) -> List[str]:
        return list(self.scalars) + [c for p in self.vectors for c in p]


def split_state(x: torch.Tensor, spec: VariableSpec, names: Sequence[str]) -> Field:
    """[B, C, H, W] physical/normalised array with named channels -> grid Field [B, H*W, .]."""
    idx = {n: i for i, n in enumerate(names)}
    B = x.shape[0]
    s = torch.stack([x[:, idx[n]] for n in spec.scalars], 1) if spec.scalars else x.new_zeros(B, 0, *x.shape[2:])
    if spec.vectors:
        u = torch.stack([x[:, idx[a]] for a, _ in spec.vectors], 1)
        v = torch.stack([x[:, idx[b]] for _, b in spec.vectors], 1)
        z = torch.complex(u, v)
    else:
        z = torch.zeros(B, 0, *x.shape[2:], dtype=torch.complex64)
    return Field(s.flatten(2).transpose(1, 2), z.flatten(2).transpose(1, 2))


def merge_state(f: Field, spec: VariableSpec, n_lat: int, n_lon: int) -> Dict[str, torch.Tensor]:
    """grid Field -> dict of [B, H, W] channels (inverse of split_state)."""
    B = f.s.shape[0]
    out = {}
    s = f.s.transpose(1, 2).reshape(B, -1, n_lat, n_lon)
    for k, n in enumerate(spec.scalars):
        out[n] = s[:, k]
    v = f.v.transpose(1, 2).reshape(B, -1, n_lat, n_lon)
    for k, (a, b) in enumerate(spec.vectors):
        out[a] = v[:, k].real
        out[b] = v[:, k].imag
    return out


def vorticity_divergence(u: torch.Tensor, v: torch.Tensor, lat_deg: np.ndarray, lon_deg: np.ndarray,
                         radius: float = 1.0, pole_mask_deg: float = 88.0):
    """Relative vorticity and divergence on a lat/lon grid (finite differences, unit sphere by default).

    u, v: [..., n_lat, n_lon] eastward / northward components.  Rows poleward of
    ``pole_mask_deg`` are set to 0 (metric singularity of the lat/lon grid).
    """
    lat = torch.as_tensor(np.deg2rad(lat_deg), dtype=u.dtype)
    dlon = float(np.deg2rad(np.mean(np.diff(lon_deg))))
    cos = torch.cos(lat)[:, None]
    dv_dlon = (torch.roll(v, -1, -1) - torch.roll(v, 1, -1)) / (2 * dlon)
    du_dlon = (torch.roll(u, -1, -1) - torch.roll(u, 1, -1)) / (2 * dlon)
    ducos_dlat = torch.gradient(u * cos, spacing=(lat,), dim=-2)[0]
    dvcos_dlat = torch.gradient(v * cos, spacing=(lat,), dim=-2)[0]
    safe = torch.where(cos.abs() < 1e-6, torch.ones_like(cos), cos)
    vort = (dv_dlon - ducos_dlat) / (radius * safe)
    div = (du_dlon + dvcos_dlat) / (radius * safe)
    mask = (torch.abs(torch.as_tensor(lat_deg)) <= pole_mask_deg).to(u.dtype)[:, None]
    return vort * mask, div * mask


def insolation_proxy(times_ns: np.ndarray, lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """max(0, cos(solar zenith)) at the given UTC times: [T, n_lat, n_lon] (known forcing, a scalar field)."""
    t = np.asarray(times_ns).astype("datetime64[ns]")
    doy = (t - t.astype("datetime64[Y]")).astype("timedelta64[s]").astype(np.float64) / 86400.0
    hour = (t - t.astype("datetime64[D]")).astype("timedelta64[s]").astype(np.float64) / 3600.0
    decl = np.deg2rad(-23.44) * np.cos(2 * np.pi * (doy + 10.0) / 365.25)
    sub_lon = np.deg2rad(-15.0 * (hour - 12.0))
    lat = np.deg2rad(np.asarray(lat_deg))[None, :, None]
    lon = np.deg2rad(np.asarray(lon_deg))[None, None, :]
    cosz = (np.sin(lat) * np.sin(decl)[:, None, None]
            + np.cos(lat) * np.cos(decl)[:, None, None] * np.cos(lon - sub_lon[:, None, None]))
    return np.maximum(cosz, 0.0).astype(np.float32)
