"""stage12 Stage B options (EMA weights, rollout curriculum, guide fine-tuning) on a synthetic 12x24 data set:
the new flags run end to end, write the expected checkpoints, and are off by default (old recipe unchanged)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_eval import parse_rollout_schedule  # noqa: E402

CHANNELS = {"z500": (5.5e4, 300.0), "t850": (280.0, 8.0), "q700": (3e-3, 1e-3), "t2m": (285.0, 10.0),
            "msl": (1.01e5, 800.0), "u850": (0.0, 8.0), "v850": (0.0, 6.0), "u500": (5.0, 12.0), "v500": (0.0, 8.0),
            "u10": (0.0, 5.0), "v10": (0.0, 4.0)}
SMALL = ["--mesh", "2", "1", "0", "--lead", "2", "--cs", "8", "--cv", "4", "--guide-cs", "8", "--guide-cv", "4",
         "--hidden", "16", "--n-basis", "4", "--heads", "2", "--ds", "4", "--dv", "2", "--batch", "2", "--threads", "1"]


def _smooth(rng, T, H, W):
    """Random walk in time of a few large-scale harmonics (so lead-time changes are non-zero and spatially smooth)."""
    lat = np.linspace(-1, 1, H)[None, :, None]
    lon = np.linspace(0, 2 * np.pi, W, endpoint=False)[None, None, :]
    coef = np.cumsum(rng.normal(size=(T, 4, 1, 1)) * 0.3, axis=0) + rng.normal(size=(1, 4, 1, 1))
    f = (coef[:, 0] * np.cos(lon) * np.cos(lat * np.pi / 2) + coef[:, 1] * np.sin(2 * lon) * lat
         + coef[:, 2] * lat ** 2 + coef[:, 3] * np.sin(lon) * np.sin(lat * np.pi))
    return f / (f.std() + 1e-6)


def _write_data(path: Path, T=(48, 24, 24), H=12, W=24, seed=0):
    rng = np.random.default_rng(seed)
    lat = np.linspace(-87.1875, 87.1875, H)
    lon = np.arange(W) * (360.0 / W)
    arrays = {"lat": lat, "lon": lon,
              "static/orog": (rng.random((H, W)) * 2000 * 9.80665).astype(np.float32),
              "static/lsm": (rng.random((H, W)) > 0.6).astype(np.float32)}
    t0 = np.datetime64("2016-01-01T00:00").astype("datetime64[ns]").astype(np.int64)
    for split, n in zip(("train", "calib", "test"), T):
        arrays[f"{split}/time"] = t0 + np.arange(n, dtype=np.int64) * 6 * 3600 * 10 ** 9
        for name, (mean, std) in CHANNELS.items():
            arrays[f"{split}/{name}"] = (mean + std * _smooth(rng, n, H, W)).astype(np.float32)
        arrays[f"{split}/q700"] = np.abs(arrays[f"{split}/q700"])
    np.savez(path, **arrays)


def _run(run_dir: Path, data: Path, stage: str, *extra: str) -> str:
    cmd = [sys.executable, "-W", "ignore", str(ROOT / "scripts/train_eval.py"), "--run", str(run_dir), "--data", str(data),
           "--stage", stage, *SMALL, *extra]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-4000:]
    return out.stdout


def test_parse_rollout_schedule():
    assert parse_rollout_schedule("") == [] and parse_rollout_schedule(None) == []
    assert parse_rollout_schedule("1000:3, 500:2") == [(500, 2), (1000, 3)]
    with pytest.raises(ValueError):
        parse_rollout_schedule("10:0")


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    p = tmp_path_factory.mktemp("era5") / "synthetic.npz"
    _write_data(p)
    return p


def test_stage_b_default_recipe_unchanged(data, tmp_path):
    """Without the stage12 flags Stage B is the old rollout rule (K from --rollout-start) and saves the trained weights."""
    run_dir = tmp_path / "run_default"
    out = _run(run_dir, data, "b_noguide", "--steps", "4", "--val-every", "2", "--rollout-steps", "2", "--rollout-start", "2")
    assert "rollout steps 2 from step 2" in out and "EMA" not in out and "rollout schedule" not in out
    rows = [json.loads(l) for l in open(run_dir / "log_vnet_noguide.jsonl")]
    assert rows[0]["k"] == 1 and (run_dir / "vnet_noguide.pt").exists()
    assert not (run_dir / "guide_finetuned.pt").exists()


def test_stage_b_ema_and_rollout_schedule(data, tmp_path):
    run_dir = tmp_path / "run_ema"
    out = _run(run_dir, data, "b_noguide", "--steps", "6", "--val-every", "3", "--ema", "0.9",
               "--rollout-schedule", "2:2,4:3")
    assert "rollout schedule [(2, 2), (4, 3)]" in out and "EMA decay 0.9" in out
    rows = [json.loads(l) for l in open(run_dir / "log_vnet_noguide.jsonl")]
    val = [r for r in rows if "val_mse" in r]
    assert len(val) == 3 and all(np.isfinite(r["val_mse"]) and np.isfinite(r["val_mse2"]) for r in val)
    sd = torch.load(run_dir / "vnet_noguide.pt")
    assert all(torch.isfinite(v).all() for v in sd.values())


def test_stage_b_guide_finetuning(data, tmp_path):
    run_dir = tmp_path / "run_ft"
    _run(run_dir, data, "a", "--steps", "3", "--val-every", "2")
    assert (run_dir / "guide.pt").exists()
    out = _run(run_dir, data, "b", "--steps", "4", "--val-every", "2", "--guide-lr", "1e-3", "--ema", "0.5")
    assert "guide fine-tuning lr 0.001" in out
    g0, g1 = torch.load(run_dir / "guide.pt"), torch.load(run_dir / "guide_finetuned.pt")
    assert g0.keys() == g1.keys()
    assert any(not torch.equal(g0[k], g1[k]) for k in g0 if g0[k].dtype.is_floating_point and g0[k].numel() > 0)
    # inference prefers the fine-tuned guide, Stage B's own start does not
    from train_eval import Run  # noqa: E402
    import argparse
    ns = argparse.Namespace(run=str(run_dir), data=str(data), **{k: None for k in (
        "mesh", "lead", "cs", "cv", "depth", "heads", "ds", "dv", "hidden", "n_basis", "guide_cs", "guide_cv",
        "diff_cs", "diff_cv", "diff_depth", "physics_ops", "tend_head", "fusion", "pigt_layers", "pigt_ops",
        "closure_tend", "resolved_terms", "resolved_coef")})
    run = Run(ns)
    m = run.forecaster(True)
    run.load_guide(m)
    k = next(k for k in g0 if g0[k].dtype.is_floating_point and not torch.equal(g0[k], g1[k]))
    assert torch.equal(m.guide.state_dict()[k], g1[k])
    run.load_guide(m, finetuned=False)
    assert torch.equal(m.guide.state_dict()[k], g0[k])


def test_stage_b_resume_and_skip(data, tmp_path):
    """Container-restart safety: Stage B resumes from its last validation checkpoint and a finished Stage B is skipped."""
    run_dir = tmp_path / "run_resume"
    args = ("--steps", "5", "--val-every", "2", "--rollout-steps", "2", "--rollout-start", "3")
    out = _run(run_dir, data, "b_noguide", *args, "--stop-after", "3")
    assert "stopping after step 2" in out
    assert (run_dir / "vnet_noguide_train_state.pt").exists() and not (run_dir / "vnet_noguide_done.json").exists()
    out = _run(run_dir, data, "b_noguide", *args)
    assert "resumed from step 3" in out and "done in" in out
    rows = [json.loads(l) for l in open(run_dir / "log_vnet_noguide.jsonl")]
    assert [r["step"] for r in rows if "val_mse" in r] == [0, 2, 4]
    assert (run_dir / "vnet_noguide_done.json").exists() and not (run_dir / "vnet_noguide_train_state.pt").exists()
    mtime = (run_dir / "vnet_noguide.pt").stat().st_mtime
    out = _run(run_dir, data, "b_noguide", *args)
    assert "already finished" in out and (run_dir / "vnet_noguide.pt").stat().st_mtime == mtime
    out = _run(run_dir, data, "b_noguide", *args, "--resume", "0")
    assert "already finished" not in out and "resumed" not in out


def test_eval_on_external_year(tmp_path):
    """--eval-data / --eval-tag / --max-cases: score a trained run on another file's test split (benchmark year)
    without touching eval.json."""
    data, other = tmp_path / "synthetic.npz", tmp_path / "other_year.npz"
    _write_data(data)
    _write_data(other, seed=1)
    run_dir = tmp_path / "run_ext"
    _run(run_dir, data, "a", "--steps", "3", "--val-every", "2")
    _run(run_dir, data, "b", "--steps", "3", "--val-every", "2")
    _run(run_dir, data, "eval", "--eval-stride", "4", "--max-lead-steps", "2", "--n-ens-cases", "0")
    base = json.load(open(run_dir / "eval.json"))
    _run(run_dir, data, "eval", "--eval-stride", "4", "--max-lead-steps", "2", "--n-ens-cases", "0",
         "--eval-data", str(other), "--eval-tag", "other", "--max-cases", "2")
    ext = json.load(open(run_dir / "eval_other.json"))
    assert ext["n_cases"] == 2 and ext["eval_data"] == str(other) and len(ext["init_times"]) == 2
    assert (run_dir / "eval_other.md").exists()
    assert json.load(open(run_dir / "eval.json")) == base  # the run's own evaluation is untouched
    assert ext["rmse"]["1"]["lie_guided"]["z500"] != base["rmse"]["1"]["lie_guided"]["z500"]
