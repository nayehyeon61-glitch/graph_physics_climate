"""stage21a physics-first core (beta starts at 1, source budget) and stage21b guide dynamical-core objective."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from gpclimate.data.era5_wb2 import DEFAULT_SPEC  # noqa: E402
from gpclimate.models.dynamics_vnet import LieGuidedForecaster, VNetConfig  # noqa: E402
from test_column_inputs import _write_columns  # noqa: E402
from test_graph_core import _core_forecaster  # noqa: E402
from test_train_eval_smoke import _run, _write_data  # noqa: E402
from test_transport_guide import GUIDE, _graphs, _state  # noqa: E402


def _physics_first(beta_init=1.0, **kw):
    plain = _core_forecaster(core=True)
    cfg = VNetConfig(**{**plain.vnet.cfg.__dict__, "core_beta_init": beta_init, **kw})
    torch.manual_seed(0)
    m = LieGuidedForecaster(DEFAULT_SPEC, GUIDE, cfg)
    sd = {k: v for k, v in plain.vnet.state_dict().items() if "beta_head" not in k}
    m.vnet.load_state_dict(sd, strict=False)
    m.guide.load_state_dict(plain.guide.state_dict())
    m.guide.flow_v_scale.fill_(10.0)
    return m.eval()


def test_beta_starts_at_the_configured_share_and_the_physics_share_is_reported():
    g = _graphs().view()
    x, c = _state(2, wind=0.5)
    zero, one = _core_forecaster(core=True), _physics_first(1.0)
    with torch.no_grad():
        f0, _ = zero(x, c, g)
        assert zero.last_core["beta"] == [0.0, 0.0, 0.0] and zero.last_core["transport"] < 1e-4  # interpolation noise
        assert zero.last_core["phys_share"] < 1e-2 and abs(zero.last_core["source_share"] - 1.0) < 1e-3
        f1, _ = one(x, c, g)
        assert all(abs(b - 1.0) < 1e-6 for b in one.last_core["beta"])  # the whole guide flow carries the state
        assert one.last_core["transport"] > 1e-3 and not torch.allclose(f0.s, f1.s)
        assert 0.0 < one.last_core["phys_share"] and 0.0 < one.last_core["source_share"]
        assert 0.0 < float(one.source_cost) < 1.0  # mean over the sub-steps (last_core reports the last one)
    assert one.source_cost.ndim == 0


def test_source_cost_is_the_same_with_and_without_checkpointing_and_differentiable():
    g = _graphs().view()
    x, c = _state(2, wind=0.5)
    a, b = _physics_first(1.0, ode_checkpoint=True).train(), _physics_first(1.0, ode_checkpoint=False).train()
    b.vnet.load_state_dict(a.vnet.state_dict())
    fa, _ = a(x, c, g)
    fb, _ = b(x, c, g)
    assert torch.allclose(fa.s, fb.s, atol=1e-6) and float(a.source_cost) == pytest.approx(float(b.source_cost), rel=1e-5)
    assert 0.0 < float(a.source_cost) < 1.0
    a.source_cost.backward()
    grads = {n: p.grad for n, p in a.vnet.named_parameters() if p.grad is not None and float(p.grad.abs().max()) > 0}
    assert any(k.startswith("head_out") for k in grads) and any(k.startswith("beta_head") for k in grads), list(grads)


def test_minimising_the_source_budget_moves_the_change_into_the_structured_terms():
    g = _graphs().view()
    x, c = _state(2, wind=0.5)
    m = _physics_first(1.0).train()
    opt = torch.optim.Adam(m.vnet.parameters(), lr=1e-2)
    m(x, c, g)
    c0 = float(m.source_cost)
    for _ in range(6):
        opt.zero_grad()
        m(x, c, g)
        m.source_cost.backward()
        opt.step()
    with torch.no_grad():
        m.eval()
        m(x, c, g)
    assert float(m.source_cost) < 0.8 * c0 and m.last_core["source_share"] < c0 ** 0.5
    assert m.last_core["phys_share"] > 0.0


def test_non_core_models_report_a_zero_source_cost():
    g = _graphs().view()
    x, c = _state(2, wind=0.5)
    m = _core_forecaster(core=False)
    with torch.no_grad():
        m(x, c, g)
    assert float(m.source_cost) == 0.0


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("physfirst")
    base, col = d / "synthetic.npz", d / "columns.npz"
    _write_data(base)
    _write_columns(col, base)
    return base, col


def test_stage21_pipeline(files, tmp_path):
    base, col = files
    common = ["--fusion", "pigt", "--pigt-layers", "1", "--pigt-ops", "1", "--curvature-ops", "1", "--prev-state", "1",
              "--column-inputs", "fcn10", "--column-data", str(col)]
    roll = ("--rollout-steps", "2", "--rollout-start", "0")
    src = tmp_path / "src"
    _run(src, base, "b_noguide", *common, "--steps", "2", "--val-every", "2")
    _run(src, base, "a", "--steps", "2", "--val-every", "2")
    _run(src, base, "b", "--steps", "2", "--val-every", "2")
    run_dir = tmp_path / "run_s21"
    run_dir.mkdir()
    for f in ("normalizer.pt", "column_norm.json"):
        (run_dir / f).write_bytes((src / f).read_bytes())
    flags = common + ["--flow-head", "1", "--integrator", "ode", "--ode-steps", "2", "--ode-drift", "transport",
                      "--ode-path-forcing", "1", "--inter-times", "1", "--tendency-core", "1", "--core-laplace", "1",
                      "--core-beta-init", "1"]
    # stage21b: Stage A with the guide's own dynamics rolled 2 x 6 h (heads only, from the plain guide)
    out_a = _run(run_dir, base, "a", *flags, "--steps", "3", "--val-every", "2", "--init-guide", str(src / "guide.pt"),
                 "--guide-heads-only", "1", "--guide-path-steps", "2", "--guide-path-weight", "0.5")
    assert "guide dynamical-core objective" in out_a and " L_path " in out_a and " path " in out_a
    rows_a = [json.loads(line) for line in open(run_dir / "log_a.jsonl")]
    assert any("l_path" in r for r in rows_a) and any(np.isfinite(r.get("val_path", np.nan)) for r in rows_a)
    cfg = json.load(open(run_dir / "config.json"))
    assert cfg["core_beta_init"] == 1.0
    with pytest.raises(AssertionError):  # the path objective needs the flow head
        _run(tmp_path / "noflow", base, "a", *common, "--steps", "2", "--val-every", "2", "--guide-path-steps", "2")
    # stage21a: Stage B with the source budget on the physics-first core
    _run(run_dir, base, "b_noguide", "--steps", "2", "--val-every", "2", *roll, "--init-vnet", str(src / "vnet_noguide.pt"))
    out = _run(run_dir, base, "b", "--steps", "2", "--val-every", "2", *roll, "--init-vnet", str(src / "vnet.pt"),
               "--source-penalty", "0.1")
    assert "physics-first: beta starts at 1" in out and "source budget penalty 0.1" in out
    assert " L_src " in out and " share P " in out
    rows = [json.loads(line) for line in open(run_dir / "log_vnet.jsonl")]
    assert any("l_src" in r for r in rows)
    assert any({"phys_share", "source_share", "beta"} <= set(r.get("core", {})) for r in rows)
    with pytest.raises(AssertionError):  # the budget needs the core
        _run(src, base, "b", "--steps", "1", "--val-every", "1", "--source-penalty", "0.1")
    _run(run_dir, base, "eval", "--eval-stride", "4", "--max-lead-steps", "2", "--n-ens-cases", "0")
    ev = json.load(open(run_dir / "eval.json"))
    assert set(ev["physics_share"]) == {"1", "2"} and all(np.isfinite(v) for v in ev["physics_share"].values())
    for name in ("lie_guided", "no_guide", "physics_only"):
        assert all(np.isfinite(v) for v in ev["rmse"]["2"][name].values()), name
    assert "물리 항 비중" in (run_dir / "eval.md").read_text()
