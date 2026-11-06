"""stage15b intermediate-time loss: an auxiliary head on the V-Net decoder embedding predicts X_{t + j 6 h}, j =
1..inter_times, inside the first 24 h lead (WeatherODE-style supervision of intermediate states without extra solver
evaluations).  Checks that the head starts at persistence, does not change the forecast, that the data targets are
the right times, and the stage9b + aux pipeline end to end on a synthetic 12x24 data set."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from gpclimate.data.era5_wb2 import DEFAULT_SPEC, ERA5Pairs  # noqa: E402
from gpclimate.geometry.graphs import SphereGraphs  # noqa: E402
from gpclimate.models.dynamics_vnet import LieGuidedForecaster, VNetConfig  # noqa: E402
from gpclimate.models.lie_physical_gnn import GuideConfig  # noqa: E402
from gpclimate.nn.so2 import Field  # noqa: E402
from test_column_inputs import _write_columns  # noqa: E402
from test_train_eval_smoke import _run, _write_data  # noqa: E402


def _model(aux: int):
    torch.manual_seed(0)
    cfg = VNetConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), heads=2, ds=4, dv=2, hidden=16, n_basis=4,
                     use_guide=False, prev_state=True, inter_times=aux)
    m = LieGuidedForecaster(DEFAULT_SPEC, GuideConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), hidden=16, n_basis=4), cfg)
    with torch.no_grad():
        m.vnet.head_out.lin_s.weight.normal_(0, 0.1)
    return m.eval()


def _inputs(B=2, G=12 * 24):
    torch.manual_seed(1)
    rf = lambda n_s, n_v: Field(torch.randn(B, G, n_s), torch.randn(B, G, n_v, dtype=torch.complex64))  # noqa: E731
    return rf(5, 3), rf(5, 1)


def test_inter_head_starts_at_persistence_and_leaves_the_forecast_unchanged():
    g = SphereGraphs(np.linspace(-87.1875, 87.1875, 12), np.arange(24) * 15.0, mesh_splits=(2, 1, 0)).view()
    x0, c = _inputs()
    m3, m0 = _model(3), _model(0)
    missing, unexpected = m3.load_state_dict(m0.state_dict(), strict=False)  # same weights except the aux head
    assert not unexpected and all(k.startswith("vnet.head_inter.") for k in missing)
    with torch.no_grad():
        f3, _ = m3(x0, c, g)
        f0, _ = m0(x0, c, g)
        aux = m3.inter_forecasts(x0)
    assert torch.allclose(f3.s, f0.s) and torch.allclose(f3.v, f0.v)  # training-only head
    assert len(aux) == 3 and all(a.s.shape == x0.s.shape and a.v.shape == x0.v.shape for a in aux)
    assert all(torch.equal(a.s, x0.s) and torch.equal(a.v, x0.v) for a in aux)  # zero-initialised
    with torch.no_grad():
        m3.vnet.head_inter.lin_s.weight.normal_(0, 0.1)
        m3.vnet.head_inter.w_re.normal_(0, 0.1)
        m3(x0, c, g)
        aux = m3.inter_forecasts(x0)
    assert not torch.allclose(aux[0].s, aux[2].s) and not torch.allclose(aux[0].v, aux[1].v)
    # each time block has its own output channels: changing block j's weights moves only aux[j]
    with torch.no_grad():
        n_s = DEFAULT_SPEC.n_s
        m3.vnet.head_inter.lin_s.bias[n_s:2 * n_s] += 1.0
        m3(x0, c, g)
        aux2 = m3.inter_forecasts(x0)
    assert torch.allclose(aux2[0].s, aux[0].s) and torch.allclose(aux2[2].s, aux[2].s)
    assert not torch.allclose(aux2[1].s, aux[1].s)
    with pytest.raises(AssertionError):
        m0.inter_forecasts(x0)


def test_state_after_is_the_intermediate_time(tmp_path):
    p = tmp_path / "d.npz"
    _write_data(p)
    ds = ERA5Pairs(str(p), "train")
    idx = [0, 5]
    for j in (1, 2, 3):
        a = ds.state_after(idx, j)
        b = ds.batch([i + j for i in idx])[0]
        assert torch.equal(a.s, b.s) and torch.equal(a.v, b.v)
    assert torch.equal(ds.state_after(idx, 4).s, ds.batch(idx)[1].s)  # 4 data steps = the 24 h target


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("aux")
    base, col = d / "synthetic.npz", d / "columns.npz"
    _write_data(base)
    _write_columns(col, base)
    return base, col


def test_stage15b_pipeline(files, tmp_path):
    base, col = files
    run_dir = tmp_path / "run_s15b"
    flags = ["--fusion", "pigt", "--pigt-layers", "1", "--pigt-ops", "1", "--curvature-ops", "1", "--prev-state", "1",
             "--column-inputs", "fcn10", "--column-data", str(col), "--inter-times", "1", "--inter-weight", "1.0"]  # SMALL uses lead 2 (12 h)
    roll = ("--rollout-steps", "2", "--rollout-start", "2")
    out = _run(run_dir, base, "b_noguide", *flags, "--steps", "3", "--val-every", "2", *roll)
    assert "intermediate-time loss: 1 auxiliary targets (6 h), weight 1.0" in out
    assert " L_int " in out and " 6/12/18 h " in out  # label of the validation column
    cfg = json.load(open(run_dir / "config.json"))
    assert cfg["inter_times"] == 1 and cfg["inter_weight"] == 1.0
    rows = [json.loads(line) for line in open(run_dir / "log_vnet_noguide.jsonl")]
    assert any("l_int" in r for r in rows) and any(len(r.get("val_inter_mse", [])) == 1 for r in rows)
    _run(run_dir, base, "a", "--steps", "3", "--val-every", "2")
    _run(run_dir, base, "b", "--steps", "3", "--val-every", "2", *roll)
    _run(run_dir, base, "c", "--steps", "2", "--val-every", "2", "--noise-constraints", "1")
    _run(run_dir, base, "eval", "--eval-stride", "4", "--max-lead-steps", "2", "--n-ens-cases", "2", "--members", "2",
         "--sample-steps", "2")
    ev = json.load(open(run_dir / "eval.json"))
    assert all(np.isfinite(v) for v in ev["rmse"]["2"]["lie_guided"].values())


def test_inter_times_must_fit_inside_the_lead(files, tmp_path):
    import subprocess
    base, _ = files
    cmd = [sys.executable, "-W", "ignore", str(ROOT / "scripts/train_eval.py"), "--run", str(tmp_path / "bad"),
           "--data", str(base), "--stage", "b_noguide", "--threads", "1", "--steps", "1", "--lead", "2", "--inter-times", "2"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert r.returncode != 0 and "inter_times must be < lead" in r.stderr
