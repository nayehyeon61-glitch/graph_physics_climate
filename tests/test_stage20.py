"""stage20 integrated version: composition rule and the full pipeline with every stage17-19 component + Stage C."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from stage20_compose import compose  # noqa: E402
from test_column_inputs import _write_columns  # noqa: E402
from test_train_eval_smoke import _run, _write_data  # noqa: E402

VARS = ("z500", "t850")


def _eval(path: Path, scale: float):
    path.mkdir(parents=True, exist_ok=True)
    rmse = {str(k): {"lie_guided": {v: scale * k for v in VARS}} for k in range(1, 6)}
    json.dump({"rmse": rmse}, open(path / "eval.json", "w"))


def test_compose_picks_the_best_ode_run_against_stage17a(tmp_path):
    """Rule of the 2026-11-08 decision card: the best run against stage17a (adoption rule), else stage17a itself;
    the gains against the 1000-step delta control are only recorded."""
    runs = {"stage17a": "r17a", "stage18": "r18", "stage18c": "r18c", "stage19c": "r19c"}
    _eval(tmp_path / "ctl", 1.0)
    _eval(tmp_path / "r17a", 1.03)  # -3 % vs the control, like the real stage17a
    _eval(tmp_path / "r18", 1.04)  # worse than 17a
    _eval(tmp_path / "r18c", 1.00)  # +2.9 % vs 17a, best
    _eval(tmp_path / "r19c", 1.02)  # +1 % vs 17a
    for d in ("r17a", "r18", "r18c", "r19c"):
        json.dump({"integrator": "ode", "tendency_core": int(d != "r17a")}, open(tmp_path / d / "config.json", "w"))
    (tmp_path / "r18c" / "guide_finetuned.pt").write_bytes(b"x")
    (tmp_path / "r18c" / "guide.pt").write_bytes(b"x")
    (tmp_path / "r17a" / "guide.pt").write_bytes(b"x")
    res = compose(tmp_path, ctl="ctl", runs=runs)
    assert res["chosen"] == "stage18c" and res["src"] == "r18c" and res["base"] == "stage17a"
    assert res["bflags"] == "--guide-lr 3e-5 --guide-heads-only 1" and res["files"] == ["guide.pt"]
    assert not res["vs_base"]["stage18"]["adopt"] and res["vs_base"]["stage18c"]["mean_gain_pct"] == 2.91
    assert not res["vs_control"]["stage17a"]["adopt"] and "stage17a" not in res["vs_base"]
    _eval(tmp_path / "r18c", 1.04)
    _eval(tmp_path / "r19c", 1.028)  # +0.2 %: below the 0.5 % mean-gain threshold
    res = compose(tmp_path, ctl="ctl", runs=runs)
    assert res["chosen"] == "stage17a" and res["bflags"] == "" and res["files"] == ["guide.pt"]  # nothing beats 17a
    _eval(tmp_path / "r19c", 0.98)
    assert compose(tmp_path, ctl="ctl", runs=runs)["bflags"] == "--pinn-weight-b 0.02"
    import shutil
    shutil.rmtree(tmp_path / "r17a")
    assert compose(tmp_path, ctl="ctl", runs=runs)["chosen"] is None  # no stage17a result: stage20 is not trained


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("s20")
    base, col = d / "synthetic.npz", d / "columns.npz"
    _write_data(base)
    _write_columns(col, base)
    return base, col


def test_stage20_full_pipeline_with_every_component(files, tmp_path):
    """The richest configuration stage20 can pick (ODE + flow head + tendency core + path memory + Laplace block +
    PINN + graph embedding, guide heads and path PINN in Stage B) through Stage C, ensemble eval and diagnose."""
    base, col = files
    common = ["--fusion", "pigt", "--pigt-layers", "1", "--pigt-ops", "1", "--curvature-ops", "1", "--prev-state", "1",
              "--column-inputs", "fcn10", "--column-data", str(col)]
    roll = ("--rollout-steps", "2", "--rollout-start", "0")
    src = tmp_path / "src"
    _run(src, base, "b_noguide", *common, "--steps", "2", "--val-every", "2")
    _run(src, base, "a", "--steps", "2", "--val-every", "2")
    _run(src, base, "b", "--steps", "2", "--val-every", "2")
    run_dir = tmp_path / "run_s20"
    run_dir.mkdir()
    for f in ("normalizer.pt", "column_norm.json"):
        (run_dir / f).write_bytes((src / f).read_bytes())
    flags = common + ["--flow-head", "1", "--integrator", "ode", "--ode-steps", "2", "--ode-drift", "transport",
                      "--ode-path-forcing", "1", "--inter-times", "1", "--tendency-core", "1", "--prev-path", "1",
                      "--core-laplace", "1", "--graph-embed", "4"]
    _run(run_dir, base, "a", *flags, "--steps", "2", "--val-every", "2", "--init-guide", str(src / "guide.pt"),
         "--guide-heads-only", "1", "--pinn-weight", "0.05")
    _run(run_dir, base, "b_noguide", "--steps", "2", "--val-every", "2", *roll, "--init-vnet", str(src / "vnet_noguide.pt"))
    out = _run(run_dir, base, "b", "--steps", "2", "--val-every", "2", *roll, "--init-vnet", str(src / "vnet.pt"),
               "--guide-lr", "1e-3", "--guide-heads-only", "1", "--pinn-weight-b", "0.02")
    assert "Laplace block" in out and " L_pinn " in out and (run_dir / "guide_finetuned.pt").exists()
    _run(run_dir, base, "c", "--steps", "2", "--val-every", "2", "--noise-constraints", "1")
    _run(run_dir, base, "eval", "--eval-stride", "4", "--max-lead-steps", "2", "--n-ens-cases", "2", "--members", "2",
         "--sample-steps", "2")
    ev = json.load(open(run_dir / "eval.json"))
    for name in ("lie_guided", "no_guide", "physics_only"):
        assert all(np.isfinite(v) for v in ev["rmse"]["2"][name].values()), name
    assert any(k.startswith("ensemble") for k in ev), sorted(ev)
    d = subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "scripts/diagnose_guide.py"), "--run", str(run_dir),
                        "--threads", "1"], capture_output=True, text=True, cwd=ROOT, timeout=900)
    assert d.returncode == 0, d.stdout[-2000:] + d.stderr[-3000:]
