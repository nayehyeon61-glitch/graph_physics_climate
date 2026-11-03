"""The chunked normaliser statistics (needed for the 10-year file of stage S7) equal the single-pass formulas."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpclimate.data.era5_wb2 import ERA5Pairs, Field, fit_normalizer, split_state  # noqa: E402

SMOKE = Path(__file__).resolve().parents[1] / "data" / "era5_wb2_64x32_smoke.npz"


def _reference(ds):
    """Single full-dataset pass (the implementation before chunking)."""
    idx = torch.arange(len(ds))
    x0, x1, c = ds.batch(idx)
    s_mean, s_std = x0.s.mean((0, 1)), x0.s.std((0, 1)) + 1e-6
    v_scale = torch.sqrt((x0.v.abs() ** 2).mean((0, 1)) / 2) + 1e-6
    cs_mean, cs_std = c.s.mean((0, 1)), c.s.std((0, 1)) + 1e-6
    cv_scale = torch.sqrt((c.v.abs() ** 2).mean((0, 1)) / 2) + 1e-6
    n0 = Field((x0.s - s_mean) / s_std, x0.v / v_scale)
    n1 = Field((x1.s - s_mean) / s_std, x1.v / v_scale)
    delta_s = (n1.s - n0.s).std((0, 1)) + 1e-6
    delta_v = torch.sqrt(((n1.v - n0.v).abs() ** 2).mean((0, 1)) / 2) + 1e-6
    aux_std = ds.aux_targets(x0).std((0, 1)) + 1e-9
    xs = split_state(ds.state[idx], ds.spec, ds.names)
    xn = ds.next_state(idx)
    tend_s = ((xn.s - xs.s) / s_std).std((0, 1)) + 1e-6
    tend_v = torch.sqrt((((xn.v - xs.v) / v_scale).abs() ** 2).mean((0, 1)) / 2) + 1e-6
    return dict(s_mean=s_mean, s_std=s_std, v_scale=v_scale, cs_mean=cs_mean, cs_std=cs_std, cv_scale=cv_scale,
                delta_s=delta_s, delta_v=delta_v, aux_std=aux_std, tend_s=tend_s, tend_v=tend_v)


@pytest.mark.skipif(not SMOKE.exists(), reason="smoke data file not present")
def test_chunked_normalizer_matches_single_pass():
    ds = ERA5Pairs(str(SMOKE), "train")
    ref = _reference(ds)
    for chunk in (7, 1024):  # several chunks and a single chunk
        norm = fit_normalizer(ds, chunk=chunk)
        for k, v in ref.items():
            got = getattr(norm, k)
            assert got.shape == v.shape, k
            # msl ~ 1e5 Pa: the float32 single pass is itself only ~1e-6 accurate; the chunked pass accumulates in float64
            assert torch.allclose(got, v, rtol=2e-5, atol=0), f"{k} chunk={chunk}: {got} vs {v}"
