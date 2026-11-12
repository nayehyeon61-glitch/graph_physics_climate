"""stage14b: initial-increment nudging.  The first forecast step relaxes toward the model's own error over the
previous lead, e_t = X_t - M(X_{t-lead}), with a learned state-dependent coefficient in (0, 1).
Checks: increment definition, zero increment == no nudging, equivariance with the increment, first-step-only
rollout, flag-off models ignore the argument, and a Stage B + eval smoke run with --nudge 1."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpclimate.data.variables import VariableSpec  # noqa: E402
from gpclimate.geometry.graphs import SphereGraphs  # noqa: E402
from gpclimate.models.dynamics_vnet import LieGuidedForecaster, VNetConfig  # noqa: E402
from gpclimate.models.lie_physical_gnn import GuideConfig  # noqa: E402
from gpclimate.nn.so2 import Field  # noqa: E402
from gpclimate.training.losses import random_gauges, relative_error  # noqa: E402

torch.manual_seed(0)
SPEC = VariableSpec(scalars=("z500", "t850"), vectors=(("u850", "v850"), ("u10", "v10")),
                    static_scalars=("orog",), static_vectors=(("dox", "doy"),), forcing_scalars=("toa",))


@pytest.fixture(scope="module")
def graphs():
    return SphereGraphs(np.linspace(-82.5, 82.5, 12), np.arange(24) * 15.0, mesh_splits=(2, 1, 0))


def _rand(B, N, cs, cv):
    return Field(torch.randn(B, N, cs), torch.complex(torch.randn(B, N, cv), torch.randn(B, N, cv)))


def _model(nudge: bool, prev_state: bool = False, carry: bool = True):
    m = LieGuidedForecaster(SPEC, GuideConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), hidden=16),
                            VNetConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), heads=2, ds=4, dv=2, hidden=16,
                                       prev_state=prev_state, nudge=nudge, nudge_carry=carry))
    for p in m.vnet.head_out.parameters():
        torch.nn.init.normal_(p, std=0.1)
    return m


def test_increment_and_nudged_step(graphs):
    model = _model(True)
    gv = graphs.view()
    G = graphs.grid.n
    x, xp = _rand(2, G, SPEC.n_s, SPEC.n_v), _rand(2, G, SPEC.n_s, SPEC.n_v)
    c = _rand(1, G, SPEC.n_cond_s, SPEC.n_cond_v)
    with torch.no_grad():
        inc = model.increment(x, xp, c, gv)
        fg, _ = model(xp, c, gv)  # the un-nudged pre-forecast of the initial time
        ref = Field((x.s - fg.s) / model.delta_scale_s, (x.v - fg.v) / model.delta_scale_v)
        assert relative_error(inc, ref) < 1e-6
        f_none, d_none = model(x, c, gv)
        f_zero, _ = model(x, c, gv, nudge=Field(torch.zeros_like(inc.s), torch.zeros_like(inc.v)))
        assert relative_error(f_none, f_zero) < 1e-7
        f_n, d_n = model(x, c, gv, nudge=inc)
        assert relative_error(f_n, f_none) > 1e-4
        # the relaxation term is alpha * increment with alpha in (0, 1) (initially ~0.12 everywhere)
        a = model.vnet.nudge_block.last_alpha
        assert a.shape == (SPEC.n_s + SPEC.n_v,) and bool(((a > 0) & (a < 1)).all())
        assert torch.allclose(a, torch.full_like(a, 0.12), atol=1e-3)
        diff = Field(d_n.s - d_none.s, d_n.v - d_none.v)
        assert relative_error(diff, Field(0.12 * inc.s, 0.12 * inc.v)) < 2e-2
        # rollout: the term added at step 1 (alpha * e) is carried into step 2 as the red-noise recursion
        r = model.rollout(x, [c, c], gv, nudge=inc)
        f1, _ = model(x, c, gv, nudge=inc)
        carried = model.carried_nudge()
        assert relative_error(carried, Field(0.12 * inc.s, 0.12 * inc.v)) < 2e-2
        s2, _ = model(f1, c, gv, nudge=carried)
        assert relative_error(r[0], f1) < 1e-7 and relative_error(r[1], s2) < 1e-7
        model(x, c, gv)  # a call without an increment leaves nothing to carry
        assert model.carried_nudge() is None
        # first-step-only variant (nudge_carry=False): step 2 is un-nudged
        m0 = _model(True, carry=False)
        m0.load_state_dict(model.state_dict())
        r0 = m0.rollout(x, [c, c], gv, nudge=inc)
        s2b, _ = m0(f1, c, gv)
        assert relative_error(r0[0], f1) < 1e-7 and relative_error(r0[1], s2b) < 1e-7
        assert relative_error(r0[1], r[1]) > 1e-5
    # without the flag the model ignores the argument
    base = _model(False)
    with torch.no_grad():
        a0, _ = base(x, c, gv)
        b0, _ = base(x, c, gv, nudge=inc)
    assert base.vnet.nudge_block is None and relative_error(a0, b0) < 1e-7


def test_nudging_equivariant(graphs):
    model = _model(True, prev_state=True)
    gv = graphs.view()
    G = graphs.grid.n
    x, xp, xpp = (_rand(2, G, SPEC.n_s, SPEC.n_v) for _ in range(3))
    c = _rand(1, G, SPEC.n_cond_s, SPEC.n_cond_v)
    gen = torch.Generator().manual_seed(1)
    tg, tm = random_gauges(graphs, gen)
    gvt = graphs.gauge_transformed(tm, tg)
    with torch.no_grad():
        inc = model.increment(x, xp, c, gv, prev_prev=xpp)
        inc_t = model.increment(x.gauge(tg), xp.gauge(tg), c.gauge(tg), gvt, prev_prev=xpp.gauge(tg))
        assert relative_error(inc_t, inc.gauge(tg)) < 2e-5  # the increment itself is covariant
        f0, _ = model(x, c, gv, prev=xp, nudge=inc)
        f1, _ = model(x.gauge(tg), c.gauge(tg), gvt, prev=xp.gauge(tg), nudge=inc.gauge(tg))
        assert relative_error(f1, f0.gauge(tg)) < 2e-5


def test_stage_b_and_eval_with_nudge(tmp_path):
    from test_train_eval_smoke import _run, _write_data
    data = tmp_path / "synthetic.npz"
    _write_data(data)
    run_dir = tmp_path / "run_nudge"
    _run(run_dir, data, "a", "--steps", "3", "--val-every", "2", "--prev-state", "1", "--nudge", "1")
    _run(run_dir, data, "b", "--steps", "4", "--val-every", "2", "--rollout-steps", "2", "--rollout-start", "2")
    _run(run_dir, data, "b_noguide", "--steps", "3", "--val-every", "2")
    cfg = json.load(open(run_dir / "config.json"))
    assert cfg["nudge"] == 1 and cfg["prev_state"] == 1
    rows = [json.loads(l) for l in open(run_dir / "log_vnet.jsonl") if "val_mse" in l]
    assert np.isfinite(rows[-1]["val_mse"]) and np.isfinite(rows[-1]["val_mse2"])
    alpha = rows[-1]["nudge_alpha"]
    assert len(alpha) > 0 and all(0 < a < 1 for a in alpha)
    sd = torch.load(run_dir / "vnet.pt")
    assert any(k.startswith("nudge_block.") for k in sd)
    _run(run_dir, data, "eval", "--eval-stride", "4", "--max-lead-steps", "2", "--n-ens-cases", "0")
    ev = json.load(open(run_dir / "eval.json"))
    assert {"lie_guided", "no_guide"} <= set(ev["rmse"]["1"]) and all(ev["finite"]["lie_guided"].values())
