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

# internal name -> (WB2 variable, pressure level or None).  The first 11 entries are the base layout of
# DEFAULT_SPEC; the remaining ones complete the FourCastNet-20 channel set (stage10 feature-matrix stage).
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
    # FourCastNet (Pathak et al. 2022; NVlabs/FourCastNet config/AFNO.yaml, channels 0-19) additions
    "sp": ("surface_pressure", None),
    "u1000": ("u_component_of_wind", 1000),
    "v1000": ("v_component_of_wind", 1000),
    "z1000": ("geopotential", 1000),
    "z850": ("geopotential", 850),
    "t500": ("temperature", 500),
    "z50": ("geopotential", 50),
    "r500": ("relative_humidity", 500),
    "r850": ("relative_humidity", 850),
    "tcwv": ("total_column_water_vapour", None),
}
WB2_STATIC = {"orog": "geopotential_at_surface", "lsm": "land_sea_mask"}

# conditions shared by every state layout: statics (orography, land-sea mask, orography gradient as a vector) and
# known forcings (insolation proxy now and at the valid time, Coriolis parameter)
_CONDITIONS = dict(static_scalars=("orog", "lsm"), static_vectors=(("dorog_dx", "dorog_dy"),),
                   forcing_scalars=("toa_t", "toa_t1", "coriolis"))

DEFAULT_SPEC = VariableSpec(
    scalars=("z500", "t850", "q700", "t2m", "msl"),
    vectors=(("u850", "v850"), ("u500", "v500"), ("u10", "v10")),
    **_CONDITIONS,
)
# FourCastNet's 20 channels (u10, v10, t2m, sp, msl, t850, u1000, v1000, z1000, u850, v850, z850, u500, v500, z500,
# t500, z50, r500, r850, tcwv) in our scalar / tangent-vector split: 12 scalars + 4 wind pairs.  FCN21 adds q700 so
# that the base layout stays a subset (identical 11 comparison variables; the stage5c water constraint keeps q700).
FCN20_SPEC = VariableSpec(
    scalars=("z500", "t850", "t2m", "msl", "sp", "z1000", "z850", "t500", "z50", "r500", "r850", "tcwv"),
    vectors=(("u850", "v850"), ("u500", "v500"), ("u10", "v10"), ("u1000", "v1000")),
    **_CONDITIONS,
)
FCN21_SPEC = VariableSpec(
    scalars=("z500", "t850", "q700", "t2m", "msl", "sp", "z1000", "z850", "t500", "z50", "r500", "r850", "tcwv"),
    vectors=(("u850", "v850"), ("u500", "v500"), ("u10", "v10"), ("u1000", "v1000")),
    **_CONDITIONS,
)
# name stored in config.json (``--spec``); runs created before stage10 have no key and mean "base11"
SPECS: Dict[str, VariableSpec] = {"base11": DEFAULT_SPEC, "fcn20": FCN20_SPEC, "fcn21": FCN21_SPEC}


def fetch_wb2_subset(out_path: str, periods: Dict[str, Tuple[str, str]], url: str = WB2_ERA5_64x32,
                     names: Optional[Sequence[str]] = None) -> str:
    """Download the state channels ``names`` (default: the 11 of DEFAULT_SPEC) for a few time windows into one
    ``.npz`` (``{split}/{name}`` [T, lat, lon] float32, ``{split}/time`` int64 ns, ``lat``, ``lon``, ``static/*``)."""
    import xarray as xr

    names = list(names) if names is not None else DEFAULT_SPEC.state_channels()
    unknown = [n for n in names if n not in WB2_SOURCES]
    if unknown:
        raise KeyError(f"no WeatherBench2 source for {unknown}; known: {sorted(WB2_SOURCES)}")
    ds = xr.open_zarr(url, storage_options={"token": "anon"})
    arrays: Dict[str, np.ndarray] = {}
    # group requests by WB2 variable so each (100-step, all-level) chunk is fetched once
    by_var: Dict[str, List[Tuple[str, Optional[int]]]] = {}
    for name in names:
        var, level = WB2_SOURCES[name]
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
    delta_s: torch.Tensor  # std of normalised lead-time change
    delta_v: torch.Tensor
    aux_std: torch.Tensor
    tend_s: Optional[torch.Tensor] = None  # std of normalised one-data-step (6 h) change (guide tendency head)
    tend_v: Optional[torch.Tensor] = None

    def state(self, f: Field) -> Field:
        return Field((f.s - self.s_mean) / self.s_std, f.v / self.v_scale)

    def state_inv(self, f: Field) -> Field:
        return Field(f.s * self.s_std + self.s_mean, f.v * self.v_scale)

    def cond(self, f: Field) -> Field:
        return Field((f.s - self.cs_mean) / self.cs_std, f.v / self.cv_scale)

    def to_dict(self) -> Dict[str, list]:
        return {k: getattr(self, k).tolist() for k in self.__dataclass_fields__ if getattr(self, k) is not None}

    def tendency(self, x0: Field, x1step: Field) -> Field:
        """Normalised one-step tendency target ``(X_{t+1} - X_t)`` in physical units -> unit variance."""
        if self.tend_s is None:
            raise ValueError("normalizer has no tendency scale; call fit_tendency_scale first")
        return Field((x1step.s - x0.s) / self.s_std / self.tend_s, (x1step.v - x0.v) / self.v_scale / self.tend_v)


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

    def prev_state(self, idx: Sequence[int]) -> Field:
        """State one lead *before* ``idx`` (stage14 input: the observed change X_t - X_{t-lead}); for the first
        ``lead`` times of a split the state itself is returned (zero change)."""
        idx = torch.as_tensor(idx, dtype=torch.long)
        return split_state(self.state[torch.clamp(idx - self.lead, min=0)], self.spec, self.names)

    def next_state(self, idx: Sequence[int]) -> Field:
        """State one data step (6 h) after ``idx`` (always available since lead >= 1)."""
        idx = torch.as_tensor(idx, dtype=torch.long)
        return split_state(self.state[idx + 1], self.spec, self.names)

    def sequence(self, idx: Sequence[int], n_steps: int):
        """States ``X_{t + k lead}`` for k = 0..n_steps and conditions for k = 0..n_steps-1 (rollout training).

        Requires ``idx + n_steps * lead < T``, i.e. ``idx < len(self) - (n_steps - 1) * lead``.
        """
        idx = torch.as_tensor(idx, dtype=torch.long)
        states = [split_state(self.state[idx + k * self.lead], self.spec, self.names) for k in range(n_steps + 1)]
        conds = [self.cond_field(idx + k * self.lead) for k in range(n_steps)]
        return states, conds

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


class _Moments:
    """Running per-channel sum / sum of squares over the (sample, node) axes of ``[B, N, C]`` real tensors, so the
    normaliser statistics of a 10-year file (14,612 x 2048 x C values per quantity) are computed in chunks instead
    of materialising every pair at once (which needs ~9 GB for 10 years)."""

    def __init__(self):
        self.n, self.s, self.q = 0, None, None

    def add(self, x: torch.Tensor) -> None:
        x = x.double()
        s, q = x.sum((0, 1)), (x * x).sum((0, 1))
        self.n += x.shape[0] * x.shape[1]
        self.s = s if self.s is None else self.s + s
        self.q = q if self.q is None else self.q + q

    def mean(self) -> torch.Tensor:
        return (self.s / self.n).float()

    def std(self) -> torch.Tensor:
        """Unbiased standard deviation (same convention as ``torch.std``)."""
        m = self.s / self.n
        return ((self.q - self.n * m * m) / max(self.n - 1, 1)).clamp_min(0).sqrt().float()

    def rms_half(self) -> torch.Tensor:
        """``sqrt(mean(x) / 2)`` for ``x = |v|^2`` of a complex vector channel (per-component scale)."""
        return (self.s / self.n / 2).sqrt().float()


def _chunks(n: int, chunk: int):
    for a in range(0, n, chunk):
        yield torch.arange(a, min(n, a + chunk))


def fit_normalizer(ds: ERA5Pairs, chunk: int = 1024) -> Normalizer:
    """Normaliser statistics over all training pairs, accumulated in chunks of ``chunk`` samples (exact up to
    float64 round-off; identical to a single full-dataset pass)."""
    m_s, m_v, m_cs, m_cv, m_aux = _Moments(), _Moments(), _Moments(), _Moments(), _Moments()
    for idx in _chunks(len(ds), chunk):
        x0, _, c = ds.batch(idx)
        m_s.add(x0.s)
        m_v.add(x0.v.abs() ** 2)
        m_cs.add(c.s)
        if c.v.shape[-1]:
            m_cv.add(c.v.abs() ** 2)
        m_aux.add(ds.aux_targets(x0))
    s_mean, s_std = m_s.mean(), m_s.std() + 1e-6
    v_scale = m_v.rms_half() + 1e-6
    cs_mean, cs_std = m_cs.mean(), m_cs.std() + 1e-6
    cv_scale = m_cv.rms_half() + 1e-6 if m_cv.n else torch.ones(0)
    aux_std = m_aux.std() + 1e-9
    m_ds, m_dv = _Moments(), _Moments()
    for idx in _chunks(len(ds), chunk):
        x0, x1, _ = ds.batch(idx)
        m_ds.add((x1.s - x0.s) / s_std)  # std of the normalised lead-time change (mean subtracted)
        m_dv.add(((x1.v - x0.v) / v_scale).abs() ** 2)
    delta_s, delta_v = m_ds.std() + 1e-6, m_dv.rms_half() + 1e-6
    norm = Normalizer(s_mean, s_std, v_scale, cs_mean, cs_std, cv_scale, delta_s, delta_v, aux_std)
    return fit_tendency_scale(ds, norm, chunk)


def fit_tendency_scale(ds: ERA5Pairs, norm: Normalizer, chunk: int = 1024) -> Normalizer:
    """Fill ``tend_s / tend_v`` (std of the normalised 6 h change) in place and return ``norm``."""
    m_ts, m_tv = _Moments(), _Moments()
    for idx in _chunks(len(ds), chunk):
        x0 = split_state(ds.state[idx], ds.spec, ds.names)
        x1 = ds.next_state(idx)
        m_ts.add((x1.s - x0.s) / norm.s_std)
        m_tv.add(((x1.v - x0.v) / norm.v_scale).abs() ** 2)
    norm.tend_s = m_ts.std() + 1e-6
    norm.tend_v = m_tv.rms_half() + 1e-6
    return norm


# ---------------------------------------------------------------------------------------------------------------
# stage15a: column information as *input-only* conditions of the V-Net (not predicted, not scored).
# WeatherODE (Liu et al. 2024) feeds 48 channels (6 variables x 7 levels) and predicts 5; our stage10a/10b fed the
# FourCastNet column channels as extra *state* channels and lost accuracy to loss dilution, while z500 / msl improved.
# Here the channels of FCN21 that are not in base11 are read from a separate file and used at the initial time only:
# a rollout never has their future values (feeding X_{t+k lead} columns would leak the truth), so later steps get
# zeros plus an availability flag of 0 (see DynamicsVNet.forward).
COLUMN_SETS: Dict[str, Tuple[Tuple[str, ...], Tuple[Tuple[str, str], ...]]] = {
    "fcn10": (("sp", "z1000", "z850", "t500", "z50", "r500", "r850", "tcwv"), (("u1000", "v1000"),)),
}


class ColumnInputs:
    """Column channels of one split, normalised (scalars z-score, each wind pair one common scale, no mean shift),
    aligned with the base data set's time axis (checked)."""

    def __init__(self, npz_path: str, split: str, times: np.ndarray, which: str = "fcn10",
                 stats: Optional[Dict[str, list]] = None):
        scalars, vectors = COLUMN_SETS[which]
        self.scalars, self.vectors = scalars, vectors
        z = np.load(npz_path)  # NpzFile: read each member once (every access re-decompresses it)
        t = z[f"{split}/time"]
        if len(t) != len(times) or not np.array_equal(t, times):
            raise ValueError(f"column file {npz_path} [{split}] times do not match the state data "
                             f"({len(t)} vs {len(times)} steps)")
        s = np.stack([z[f"{split}/{n}"] for n in scalars], 1)  # [T, n_s, H, W]
        u = np.stack([z[f"{split}/{a}"] for a, _ in vectors], 1)
        v = np.stack([z[f"{split}/{b}"] for _, b in vectors], 1)
        z.close()
        if stats is None:
            stats = column_stats(s, u, v)
        self.stats = stats
        mean = np.asarray(stats["s_mean"], np.float32)[None, :, None, None]
        std = np.asarray(stats["s_std"], np.float32)[None, :, None, None]
        scale = np.asarray(stats["v_scale"], np.float32)[None, :, None, None]
        self.s = torch.as_tensor(((s - mean) / std).astype(np.float32))
        self.v = torch.complex(torch.as_tensor((u / scale).astype(np.float32)),
                               torch.as_tensor((v / scale).astype(np.float32)))

    @property
    def n_s(self) -> int:
        return len(self.scalars)

    @property
    def n_v(self) -> int:
        return len(self.vectors)

    def field(self, idx) -> Field:
        """Normalised column channels at ``idx`` as a grid Field ``[B, H*W, n]``."""
        idx = torch.as_tensor(idx, dtype=torch.long)
        return Field(self.s[idx].flatten(2).transpose(1, 2), self.v[idx].flatten(2).transpose(1, 2))


def column_stats(s: np.ndarray, u: np.ndarray, v: np.ndarray, chunk: int = 1024) -> Dict[str, list]:
    """Per-channel mean / std of the column scalars and the common RMS scale of each wind pair, in float64 chunks
    (a 10-year split is 14,612 x 2048 values per channel)."""
    n, k = len(s), s.shape[1]
    s1, s2, w2 = np.zeros(k), np.zeros(k), np.zeros(u.shape[1])
    for i in range(0, n, chunk):
        a = s[i:i + chunk].astype(np.float64)
        s1 += a.sum((0, 2, 3))
        s2 += (a ** 2).sum((0, 2, 3))
        w2 += (u[i:i + chunk].astype(np.float64) ** 2 + v[i:i + chunk].astype(np.float64) ** 2).sum((0, 2, 3))
    cnt = n * s.shape[2] * s.shape[3]
    mean = s1 / cnt
    std = np.sqrt(np.maximum(s2 / cnt - mean ** 2, 0.0)) + 1e-6
    return {"s_mean": mean.tolist(), "s_std": std.tolist(), "v_scale": (np.sqrt(w2 / cnt / 2) + 1e-6).tolist()}
