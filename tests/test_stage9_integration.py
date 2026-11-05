"""stage9 integrated structure on a synthetic 12x24 data set: every adopted component switched on together
(PIGT fusion, curvature operators in the guide, previous-change input, Stage C noise constraints) runs through the
stage9 order guide-free V-Net -> Stage A with forecast-state perturbation -> Stage B -> Stage C -> eval -> diagnose."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from test_train_eval_smoke import _run, _write_data  # noqa: E402

STAGE9 = ["--fusion", "pigt", "--pigt-layers", "1", "--pigt-ops", "1", "--curvature-ops", "1", "--prev-state", "1"]


def test_stage9_pipeline(tmp_path):
    data = tmp_path / "synthetic.npz"
    _write_data(data)
    run_dir = tmp_path / "run_s9"
    roll = ("--rollout-steps", "2", "--rollout-start", "2")
    _run(run_dir, data, "b_noguide", *STAGE9, "--steps", "3", "--val-every", "2", *roll)
    cfg = json.load(open(run_dir / "config.json"))
    assert cfg["fusion"] == "pigt" and cfg["curvature_ops"] == 1 and cfg["prev_state"] == 1
    _run(run_dir, data, "a", "--steps", "3", "--val-every", "2", "--perturb-model", str(run_dir / "vnet_noguide.pt"),
         "--perturb-prob", "0.5")
    _run(run_dir, data, "b", "--steps", "3", "--val-every", "2", *roll)
    _run(run_dir, data, "c", "--steps", "2", "--val-every", "2", "--noise-constraints", "1")
    _run(run_dir, data, "eval", "--eval-stride", "4", "--max-lead-steps", "2", "--n-ens-cases", "2", "--members", "2",
         "--sample-steps", "2")
    # external benchmark year: --n-ens-cases 0 must skip the Stage C ensemble instead of scoring an empty one
    _run(run_dir, data, "eval", "--eval-stride", "4", "--max-lead-steps", "2", "--n-ens-cases", "0",
         "--eval-data", str(data), "--eval-tag", "bench", "--max-cases", "2")
    assert "ensemble_lead1" not in json.load(open(run_dir / "eval_bench.json"))
    ev = json.load(open(run_dir / "eval.json"))
    assert {"lie_guided", "no_guide"} <= set(ev["rmse"]["1"])
    assert all(v == v for v in ev["rmse"]["2"]["lie_guided"].values())  # finite
    out = subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "scripts/diagnose_guide.py"), "--run", str(run_dir),
                          "--threads", "1"], capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-4000:]
    assert (run_dir / "diagnose_guide.json").exists()
