"""Small ERA5 subsets from the public WeatherBench2 bucket + pair datasets and normalisation.

Source: ``gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr``
(6-hourly, 64x32 equiangular, conservative regridding; WeatherBench2 data
catalogue).  Only a few weeks are fetched for CPU smoke tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from gpclimate.data.variables import VariableSpec, insolation_proxy, split_state, vorticity_divergence
from gpclimate.nn.so2 import Field

WB2_ERA5_64x32 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"

# internal name -> (WB2 variable, pressure level or None)
WB2_SOURCES: Dict[str, Tuple[str, Optional[int]]] = {
    "z500": ("geopotential", 500),
    "t850": ("temperature", 850),
    "q700": ("specific_humidity", 700),
    "t2m": ("2m_temperature", None),
    "msl": ("mean_sea_level_pressure", None),
    "u850": ("u_component_of_wind", 850),
    "v850": ("v_component_of_wind", 850),
    "u500": ("u_component_of_wind", 500),
    "v500": ("v_component_of_wind", 500),
    "u10": ("10m_u_component_of_wind", None),
    "v10": ("10m_v_component_of_wind", None),
}
WB2_STATIC = {"orog": "geopotential_at_surface", "lsm": "land_sea_mask"}

DEFAULT_SPEC = VariableSpec(
    scalars=("z500", "t850", "q700", "t2m", "msl"),
    vectors=(("u850", "v850"), ("u500", "v500"), ("u10", "v10")),
    static_scalars=("orog", "lsm"),
    static_vectors=(("dorog_dx", "dorog_dy"),),
    forcing_scalars=("toa_t", "toa_t1", "coriolis"),
)


def fetch_wb2_subset(out_path: str, periods: Dict[str, Tuple[str, str]], url: str = WB2_ERA5_64x32) -> str:
    """Download the variables of DEFAULT_SPEC for a few time windows into one ``.npz``."""
    import xarray as xr

    ds = xr.open_zarr(url, storage_options={"token": "anon"})
    arrays: Dict[str, np.ndarray] = {}
    # group requests by WB2 variable so each (100-step, all-level) chunk is fetched once
    by_var: Dict[str, List[Tuple[str, Optional[int]]]] = {}
    for name, (var, level) in WB2_SOURCES.items():
        by_var.setdefault(var, []).append((name, level))
    for split, (start, end) in periods.items():
        sub = ds.sel(time=slice(start, end))
        arrays[f"{split}/time"] = sub["time"].values.astype("datetime64[ns]").astype(np.int64)
        for var, items in by_var.items():
            da = sub[var]
            levels = [lv for _, lv in items if lv is not None]
            if levels:
                da = da.sel(level=levels)
                vals = da.transpose("time", "level", "latitude", "longitude").sortby("latitude").values
                for name, lv in items:
                    arrays[f"{split}/{name}"] = vals[:, levels.index(lv)].astype(np.float32)
            else:
                vals = da.transpose("time", "latitude", "longitude").sortby("latitude").values
                for name, _ in items:
                    arrays[f"{split}/{name}"] = vals.astype(np.float32)
    lat = ds["latitude"].sortby("latitude").values
    arrays["lat"] = np.asarray(lat, np.float64)
    arrays["lon"] = np.asarray(ds["longitude"].values, np.float64)
    for name, var in WB2_STATIC.items():
        arrays[f"static/{name}"] = ds[var].transpose("latitude", "longitude").sortby("latitude").values.astype(np.float32)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    return out_path


def static_fields(z: Dict[str, np.ndarray], lat: np.ndarray, lon: np.ndarray) -> Dict[str, np.ndarray]:
    """Orography, land-sea mask and the orography gradient (a *vector* condition)."""
    orog = z["static/orog"] / 9.80665  # metres
    lsm = z["static/lsm"]
    latr = np.deg2rad(lat)
    dlon = np.deg2rad(np.mean(np.diff(lon)))
    ddx = (np.roll(orog, -1, -1) - np.roll(orog, 1, -1)) / (2 * dlon) / np.maximum(np.cos(latr)[:, None], 1e-3)
    ddy = np.gradient(orog, latr, axis=0)
    return {"orog": orog, "lsm": lsm, "dorog_dx": ddx, "dorog_dy": ddy}


@dataclass
class Normalizer:
    """Scalars: z-score.  Vectors: one common scale per (u, v) pair and no mean shift,
    so normalisation commutes with frame rotations (design doc section 9)."""

    s_mean: torch.Tensor
    s_std: torch.Tensor
    v_scale: torch.Tensor
    cs_mean: torch.Tensor
    cs_std: torch.Tensor
    cv_scale: torch.Tensor
    delta_s: torch.Tensor  # std of normalised one-step change
    delta_v: torch.Tensor
    aux_std: torch.Tensor

    def state(self, f: Field) -> Field:
        return Field((f.s - self.s_mean) / self.s_std, f.v / self.v_scale)

    def state_inv(self, f: Field) -> Field:
        return Field(f.s * self.s_std + self.s_mean, f.v * self.v_scale)

    def cond(self, f: Field) -> Field:
        return Field((f.s - self.cs_mean) / self.cs_std, f.v / self.cv_scale)

    def to_dict(self) -> Dict[str, list]:
        return {k: getattr(self, k).tolist() for k in self.__dataclass_fields__}


class ERA5Pairs:
    """(X_t, X_{t+lead}) pairs plus conditions, as grid Fields (physical units)."""

    def __init__(self, npz_path: str, split: str, spec: VariableSpec = DEFAULT_SPEC, lead_steps: int = 4):
        z = dict(np.load(npz_path))
        self.spec, self.lead = spec, lead_steps
        self.lat, self.lon = z["lat"], z["lon"]
        self.times = z[f"{split}/time"]
        names = spec.state_channels()
        self.state = torch.as_tensor(np.stack([z[f"{split}/{n}"] for n in names], 1))  # [T, C, H, W]
        self.names = names
        st = static_fields(z, self.lat, self.lon)
        self.static = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in st.items()}
        self.toa = torch.as_tensor(insolation_proxy(self.times, self.lat, self.lon))  # [T, H, W]
        self.coriolis = torch.as_tensor(np.sin(np.deg2rad(self.lat)), dtype=torch.float32)[:, None].expand(
            len(self.lat), len(self.lon))

    def __len__(self) -> int:
        return max(0, self.state.shape[0] - self.lead)

    def cond_field(self, idx: torch.Tensor) -> Field:
        H, W = len(self.lat), len(self.lon)
        B = len(idx)
        sc = [self.static[n].expand(B, H, W) for n in self.spec.static_scalars]
        fc = {"toa_t": self.toa[idx], "toa_t1": self.toa[idx + self.lead], "coriolis": self.coriolis.expand(B, H, W)}
        sc += [fc[n] for n in self.spec.forcing_scalars]
        s = torch.stack(sc, 1).flatten(2).transpose(1, 2)
        vs = [torch.complex(self.static[a], self.static[b]).expand(B, H, W) for a, b in self.spec.static_vectors]
        v = torch.stack(vs, 1).flatten(2).transpose(1, 2) if vs else torch.zeros(B, H * W, 0, dtype=torch.complex64)
        return Field(s, v)

    def batch(self, idx: Sequence[int]):
        idx = torch.as_tensor(idx, dtype=torch.long)
        x0 = split_state(self.state[idx], self.spec, self.names)
        x1 = split_state(self.state[idx + self.lead], self.spec, self.names)
        return x0, x1, self.cond_field(idx)

    def aux_targets(self, f_phys: Field) -> torch.Tensor:
        """Vorticity / divergence (unit-sphere FD) of every wind pair: [B, G, 2*n_v]."""
        B, G = f_phys.s.shape[:2]
        H, W = len(self.lat), len(self.lon)
        out = []
        for k in range(self.spec.n_v):
            u = f_phys.v[..., k].real.reshape(B, H, W)
            v = f_phys.v[..., k].imag.reshape(B, H, W)
            vo, dv = vorticity_divergence(u, v, self.lat, self.lon)
            out += [vo.reshape(B, G), dv.reshape(B, G)]
        return torch.stack(out, -1)


def fit_normalizer(ds: ERA5Pairs) -> Normalizer:
    idx = torch.arange(len(ds))
    x0, x1, c = ds.batch(idx)
    s_mean, s_std = x0.s.mean((0, 1)), x0.s.std((0, 1)) + 1e-6
    v_scale = torch.sqrt((x0.v.abs() ** 2).mean((0, 1)) / 2) + 1e-6
    cs_mean, cs_std = c.s.mean((0, 1)), c.s.std((0, 1)) + 1e-6
    cv_scale = torch.sqrt((c.v.abs() ** 2).mean((0, 1)) / 2) + 1e-6 if c.v.shape[-1] else torch.ones(0)
    n0 = Field((x0.s - s_mean) / s_std, x0.v / v_scale)
    n1 = Field((x1.s - s_mean) / s_std, x1.v / v_scale)
    delta_s = (n1.s - n0.s).std((0, 1)) + 1e-6
    delta_v = torch.sqrt(((n1.v - n0.v).abs() ** 2).mean((0, 1)) / 2) + 1e-6
    aux_std = ds.aux_targets(x0).std((0, 1)) + 1e-9
    return Normalizer(s_mean, s_std, v_scale, cs_mean, cs_std, cv_scale, delta_s, delta_v, aux_std)
