"""PG-Flow / KAI contract adapter: exact round trip and common-scale wind vectors."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpclimate.contracts.pgflow_kai import CHANNEL_NAMES, KAIAdapter, kai_grid, kai_spec  # noqa: E402


def _stats():
    rng = np.random.default_rng(0)
    mean = rng.normal(size=67)
    std = rng.uniform(0.5, 3.0, size=67)
    return mean, std


def test_channel_layout():
    assert len(CHANNEL_NAMES) == 67
    assert CHANNEL_NAMES[6] == "z500" and CHANNEL_NAMES[33] == "t850" and CHANNEL_NAMES[64] == "orog"
    spec = kai_spec()
    assert spec.n_v == 12 and spec.n_s == 41  # 66 dynamic = 41 scalars + 12 wind pairs (24) + tisr forcing
    lat, lon = kai_grid()
    assert lat[0] == -90 and lat[-1] == 90 and lon[-1] == 358.5


def test_round_trip_exact():
    mean, std = _stats()
    ad = KAIAdapter(mean, std)
    x = torch.randn(2, 67, 7, 12, dtype=torch.float64)
    state, cond = ad.to_internal(x)
    back = ad.to_kai(state, x)
    assert torch.allclose(back, x, atol=1e-5)  # float32 internal state


def test_wind_pair_uses_common_scale():
    """Physical (u, v) rotate as a vector: |u + iv| of the internal channel is |wind| / one scale."""
    mean, std = _stats()
    ad = KAIAdapter(mean, std)
    x = torch.randn(1, 67, 3, 4, dtype=torch.float64)
    state, _ = ad.to_internal(x)
    k = 6  # 500 hPa pair
    iu, iv = CHANNEL_NAMES.index("u500"), CHANNEL_NAMES.index("v500")
    u = x[0, iu] * (std[iu] + 1e-8) + mean[iu]
    v = x[0, iv] * (std[iv] + 1e-8) + mean[iv]
    speed = torch.sqrt(u ** 2 + v ** 2).flatten()
    got = state.v[0, :, k].abs().double() * ad.vec_scale[k]
    assert torch.allclose(got, speed, rtol=1e-5)
