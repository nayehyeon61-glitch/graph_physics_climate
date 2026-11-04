"""stage14: the V-Net also sees the observed change over the previous lead (X_t - X_{t-lead}); the guide does not.
Checks: data helper clamps at the start of a split, forecaster equivariance with the extra input, the zero-change
fallback (prev=None == prev=state), rollout threading, and a Stage B + eval smoke run with --prev-state 1."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpclimate.data.era5_wb2 import ERA5Pairs, SPECS  # noqa: E402
from gpclimate.data.variables import VariableSpec  # noqa: E402
from gpclimate.geometry.graphs import SphereGraphs  # noqa: E402
from gpclimate.models.dynamics_vnet import LieGuidedForecaster, VNetConfig  # noqa: E402
from gpclimate.models.lie_physical_gnn import GuideConfig  # noqa: E402
from gpclimate.nn.so2 import Field  # noqa: E402
from gpclimate.training.losses import gauge_equivariance_report, random_gauges, relative_error  # noqa: E402

torch.manual_seed(0)
SPEC = VariableSpec(scalars=("z500", "t850"), vectors=(("u850", "v850"), ("u10", "v10")),
                    static_scalars=("orog",), static_vectors=(("dox", "doy"),), forcing_scalars=("toa",))


@pytest.fixture(scope="module")
def graphs():
    return SphereGraphs(np.linspace(-82.5, 82.5, 12), np.arange(24) * 15.0, mesh_splits=(2, 1, 0))


def _rand(B, N, cs, cv):
    return Field(torch.randn(B, N, cs), torch.complex(torch.randn(B, N, cv), torch.randn(B, N, cv)))


def _model(prev_state: bool):
    m = LieGuidedForecaster(SPEC, GuideConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), hidden=16),
                            VNetConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), heads=2, ds=4, dv=2, hidden=16,
                                       prev_state=prev_state))
    for p in m.vnet.head_out.parameters():
        torch.nn.init.normal_(p, std=0.1)
    return m


def test_prev_state_data_helper(tmp_path):
    from test_train_eval_smoke import _write_data
    p = tmp_path / "d.npz"
    _write_data(p, T=(16, 12, 12))
    ds = ERA5Pairs(str(p), "train", SPECS["base11"], lead_steps=2)
    x = ds.prev_state([0, 1, 5])
    x0 = ds.batch([0, 0, 3])[0]  # states at 0 (idx 0 and 1 clamp to 0) and 3 (= 5 - lead)
    assert torch.allclose(x.s, x0.s) and torch.allclose(x.v, x0.v)


def test_prev_input_equivariant_and_fallback(graphs):
    model = _model(True)
    G = graphs.grid.n
    x, xp = _rand(2, G, SPEC.n_s, SPEC.n_v), _rand(2, G, SPEC.n_s, SPEC.n_v)
    c = _rand(1, G, SPEC.n_cond_s, SPEC.n_cond_v)
    gen = torch.Generator().manual_seed(1)
    tg, tm = random_gauges(graphs, gen)
    with torch.no_grad():
        f0, _ = model(x, c, graphs.view(), prev=xp)
        f1, _ = model(x.gauge(tg), c.gauge(tg), graphs.gauge_transformed(tm, tg), prev=xp.gauge(tg))
        assert relative_error(f1, f0.gauge(tg)) < 2e-5
        f_none, _ = model(x, c, graphs.view())
        f_same, _ = model(x, c, graphs.view(), prev=x)
        assert relative_error(f_none, f_same) < 1e-7
        assert relative_error(f0, f_none) > 1e-4  # the extra input matters
        # rollout threads the previous input state
        conds = [c, c]
        r = model.rollout(x, conds, graphs.view(), prev=xp)
        s1, _ = model(x, c, graphs.view(), prev=xp)
        s2, _ = model(s1, c, graphs.view(), prev=x)
        assert relative_error(r[1], s2) < 1e-7
    # without the flag the model ignores prev entirely
    base = _model(False)
    with torch.no_grad():
        a, _ = base(x, c, graphs.view())
        b, _ = base(x, c, graphs.view(), prev=xp)
    assert relative_error(a, b) < 1e-7
    fn = lambda xx, cc, gv: {"forecast": model(xx, cc, gv)[0]}  # noqa: E731
    rep = gauge_equivariance_report(fn, x, c, graphs, {"forecast": "grid"})
    assert max(rep.values()) < 2e-5, rep


def test_stage_b_and_eval_with_prev_state(tmp_path):
    from test_train_eval_smoke import _run, _write_data
    data = tmp_path / "synthetic.npz"
    _write_data(data)
    run_dir = tmp_path / "run_prev"
    _run(run_dir, data, "a", "--steps", "3", "--val-every", "2", "--prev-state", "1")
    _run(run_dir, data, "b", "--steps", "4", "--val-every", "2", "--rollout-steps", "2", "--rollout-start", "2")
    _run(run_dir, data, "b_noguide", "--steps", "3", "--val-every", "2")
    assert json.load(open(run_dir / "config.json"))["prev_state"] == 1
    rows = [json.loads(l) for l in open(run_dir / "log_vnet.jsonl")]
    assert np.isfinite(rows[-1]["val_mse"]) and np.isfinite(rows[-1]["val_mse2"])
    sd = torch.load(run_dir / "vnet.pt")
    k = next(k for k in sd if k.startswith("lift.embed.lin_s.weight"))
    assert sd[k].shape[1] == 2 * 5 + 5  # state + change + scalar conditions
    _run(run_dir, data, "eval", "--eval-stride", "4", "--max-lead-steps", "2", "--n-ens-cases", "0")
    ev = json.load(open(run_dir / "eval.json"))
    assert {"lie_guided", "no_guide"} <= set(ev["rmse"]["1"]) and all(ev["finite"]["lie_guided"].values())
