"""stage10 physical-guide feature matrix: the FourCastNet-style channel specs and their WeatherBench2 sources, the
npz -> pairs -> normaliser -> forecaster path with 21 state channels, and the base layout staying a subset."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpclimate.data.era5_wb2 import (DEFAULT_SPEC, FCN20_SPEC, FCN21_SPEC, SPECS, WB2_SOURCES, ERA5Pairs,  # noqa: E402
                                     fit_normalizer, fit_tendency_scale)
from gpclimate.data.variables import merge_state  # noqa: E402
from gpclimate.geometry.graphs import SphereGraphs  # noqa: E402
from gpclimate.models.dynamics_vnet import LieGuidedForecaster, VNetConfig  # noqa: E402
from gpclimate.models.lie_physical_gnn import GuideConfig  # noqa: E402
from gpclimate.nn.physics_ops import PhysicsOperators  # noqa: E402

torch.manual_seed(0)
# NVlabs/FourCastNet config/AFNO.yaml: in_channels = out_channels = 0..19 in this order
FCN_CHANNELS = ["u10", "v10", "t2m", "sp", "msl", "t850", "u1000", "v1000", "z1000", "u850", "v850", "z850",
                "u500", "v500", "z500", "t500", "z50", "r500", "r850", "tcwv"]


def test_specs_cover_the_fourcastnet_channels_and_keep_the_base_layout_as_a_subset():
    assert sorted(FCN20_SPEC.state_channels()) == sorted(FCN_CHANNELS)
    assert set(FCN21_SPEC.state_channels()) == set(FCN_CHANNELS) | {"q700"}
    assert set(DEFAULT_SPEC.state_channels()) <= set(FCN21_SPEC.state_channels())
    assert FCN21_SPEC.n_s == 13 and FCN21_SPEC.n_v == 4 and len(FCN21_SPEC.state_channels()) == 21
    for name, spec in SPECS.items():
        ch = spec.state_channels()
        assert len(ch) == len(set(ch)), name
        assert all(c in WB2_SOURCES for c in ch), name  # every channel can be fetched from the WB2 bucket
        for u, v in spec.vectors:  # wind pairs are (eastward, northward) components of the same level
            assert u[0] == "u" and v[0] == "v" and u[1:] == v[1:]
            assert WB2_SOURCES[u][1] == WB2_SOURCES[v][1]
    # the conditions (statics + forcings) are unchanged, so the fixed operators see the same geometry
    assert FCN21_SPEC.n_cond_s == DEFAULT_SPEC.n_cond_s == 5 and FCN21_SPEC.n_cond_v == DEFAULT_SPEC.n_cond_v == 1
    # physics-operator matrix growth documented in docs/design/Physical_Guide_Feature_Matrix_KO.md
    assert PhysicsOperators(5, 3, 5).out_dims == (47, 19)
    assert PhysicsOperators(13, 4, 5).out_dims == (101, 30)


def _synthetic_npz(path: Path, spec, T: int = 12) -> None:
    lat = np.linspace(-87.1875, 87.1875, 32)
    lon = np.arange(64) * 5.625
    rng = np.random.default_rng(0)
    arrays = {"lat": lat, "lon": lon,
              "static/orog": (rng.random((32, 64)) * 1e4).astype(np.float32),
              "static/lsm": (rng.random((32, 64)) > 0.5).astype(np.float32)}
    t0 = np.datetime64("2019-01-01T00:00").astype("datetime64[ns]").astype(np.int64)
    for split in ("train", "calib", "test"):
        arrays[f"{split}/time"] = t0 + np.arange(T) * 6 * 3600 * 10 ** 9
        for i, n in enumerate(spec.state_channels()):
            arrays[f"{split}/{n}"] = (rng.standard_normal((T, 32, 64)) * (i + 1) + 10 * i).astype(np.float32)
    np.savez(path, **arrays)


def test_fcn21_pairs_normaliser_and_forecaster(tmp_path):
    p = tmp_path / "fcn21.npz"
    _synthetic_npz(p, FCN21_SPEC)
    ds = ERA5Pairs(str(p), "train", FCN21_SPEC, lead_steps=4)
    assert ds.state.shape == (12, 21, 32, 64) and len(ds) == 8
    norm = fit_normalizer(ds, chunk=5)
    fit_tendency_scale(ds, norm, chunk=5)
    assert norm.s_mean.shape == (13,) and norm.v_scale.shape == (4,) and norm.aux_std.shape == (8,)
    assert norm.tend_s.shape == (13,) and norm.delta_v.shape == (4,)
    x0, x1, cond = ds.batch([0, 1])
    assert x0.s.shape == (2, 2048, 13) and x0.v.shape == (2, 2048, 4)
    assert cond.s.shape == (2, 2048, 5) and cond.v.shape == (2, 2048, 1)
    # normalised scalars are z-scored per channel, vectors share one scale per pair (no mean shift)
    xn = norm.state(x0)
    assert torch.isfinite(xn.s).all() and torch.isfinite(xn.v).all()
    assert torch.allclose(norm.state_inv(xn).s, x0.s, rtol=1e-5, atol=1e-3)
    # split / merge round trip preserves every named channel
    phys = merge_state(x0, FCN21_SPEC, 32, 64)
    for i, n in enumerate(ds.names):
        assert torch.equal(phys[n], ds.state[[0, 1], i]), n
    # the base-11 spec reads the same (superset) file with identical values
    ds11 = ERA5Pairs(str(p), "train", DEFAULT_SPEC, lead_steps=4)
    assert ds11.state.shape[1] == 11
    for n in ds11.names:
        assert torch.equal(ds11.state[:, ds11.names.index(n)], ds.state[:, ds.names.index(n)]), n
    # guide (physics operators + tendency head) and V-Net run on the 21-channel state
    graphs = SphereGraphs(ds.lat, ds.lon, mesh_splits=(2, 1, 0))
    g = graphs.view()
    gcfg = GuideConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), n_basis=4, hidden=16, physics_ops=True, tend_head=True)
    vcfg = VNetConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), heads=2, ds=4, dv=2, n_basis=4, hidden=16)
    m = LieGuidedForecaster(FCN21_SPEC, gcfg, vcfg)
    cn = norm.cond(cond)
    with torch.no_grad():
        P = m.physics(xn, cn, g)
        T = m.guide.physics_tendency(P, xn, g)
        f, d = m(xn, cn, g)
    assert T.s.shape == (2, g.n_grid, 13) and T.v.shape == (2, g.n_grid, 4)
    assert f.s.shape == xn.s.shape and f.v.shape == xn.v.shape and torch.isfinite(f.s).all()
    assert torch.allclose(f.s, xn.s) and torch.allclose(f.v, xn.v)  # zero-initialised delta head = persistence
