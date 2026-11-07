"""stage15a column inputs: FourCastNet column channels (sp, z1000, z850, t500, z50, r500, r850, tcwv, u/v1000) as
V-Net inputs at the initial time only.  Checks the loader (time alignment, normalisation), that a rollout never
feeds column values after the first step (no leakage of future columns), and the stage9 + columns pipeline end to
end on a synthetic 12x24 data set."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from gpclimate.data.era5_wb2 import COLUMN_SETS, DEFAULT_SPEC, ColumnInputs  # noqa: E402
from gpclimate.geometry.graphs import SphereGraphs  # noqa: E402
from gpclimate.models.dynamics_vnet import LieGuidedForecaster, VNetConfig  # noqa: E402
from gpclimate.models.lie_physical_gnn import GuideConfig  # noqa: E402
from gpclimate.nn.so2 import Field  # noqa: E402
from test_train_eval_smoke import _run, _write_data  # noqa: E402

COL = {"sp": (9.8e4, 900.0), "z1000": (900.0, 500.0), "z850": (1.4e4, 400.0), "t500": (255.0, 6.0),
       "z50": (2.0e5, 1500.0), "r500": (0.5, 0.2), "r850": (0.7, 0.15), "tcwv": (20.0, 12.0),
       "u1000": (0.0, 5.0), "v1000": (0.0, 4.0)}


def _write_columns(path: Path, base: Path, seed=3):
    """Column file with the same splits and times as ``base``."""
    rng = np.random.default_rng(seed)
    z = np.load(base)
    arrays = {"lat": z["lat"], "lon": z["lon"]}
    for split in ("train", "calib", "test"):
        t = z[f"{split}/time"]
        arrays[f"{split}/time"] = t
        H, W = z[f"{split}/z500"].shape[1:]
        for name, (mean, std) in COL.items():
            arrays[f"{split}/{name}"] = (mean + std * rng.normal(size=(len(t), H, W))).astype(np.float32)
    np.savez_compressed(path, **arrays)


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("col")
    base, col = d / "synthetic.npz", d / "columns.npz"
    _write_data(base)
    _write_columns(col, base)
    return base, col


def test_column_loader(files):
    base, col = files
    t = np.load(base)["train/time"]
    ci = ColumnInputs(str(col), "train", t)
    scal, vec = COLUMN_SETS["fcn10"]
    assert ci.n_s == len(scal) == 8 and ci.n_v == len(vec) == 1
    f = ci.field([0, 3])
    assert f.s.shape == (2, 12 * 24, 8) and f.v.shape == (2, 12 * 24, 1) and f.v.is_complex()
    assert abs(float(ci.s.mean())) < 0.05 and abs(float(ci.s.std()) - 1) < 0.05  # train-split z-score
    # the same statistics are reused for another split (no refit)
    tc = np.load(base)["calib/time"]
    assert ColumnInputs(str(col), "calib", tc, stats=ci.stats).stats == ci.stats
    with pytest.raises(ValueError):  # misaligned time axis
        ColumnInputs(str(col), "train", t[1:])


def _model(col: bool):
    torch.manual_seed(0)
    cfg = VNetConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), heads=2, ds=4, dv=2, hidden=16, n_basis=4,
                     use_guide=False, prev_state=True, col_s=8 if col else 0, col_v=1 if col else 0)
    m = LieGuidedForecaster(DEFAULT_SPEC, GuideConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), hidden=16, n_basis=4), cfg)
    with torch.no_grad():  # leave persistence so the output depends on the inputs
        m.vnet.head_out.lin_s.weight.normal_(0, 0.1)
        if m.vnet.head_out.w_re is not None:
            m.vnet.head_out.w_re.normal_(0, 0.1)
    return m.eval()


def _inputs(B=2, G=12 * 24):
    torch.manual_seed(1)
    rf = lambda n_s, n_v: Field(torch.randn(B, G, n_s), torch.randn(B, G, n_v, dtype=torch.complex64))  # noqa: E731
    return rf(5, 3), [rf(5, 1), rf(5, 1)], rf(8, 1)


def test_rollout_uses_columns_only_at_the_first_step():
    g = SphereGraphs(np.linspace(-87.1875, 87.1875, 12), np.arange(24) * 15.0, mesh_splits=(2, 1, 0)).view()
    m = _model(True)
    x0, conds, col = _inputs()
    with torch.no_grad():
        out = m.rollout(x0, conds, g, col=col)
        f1, _ = m(x0, conds[0], g, col=col)
        f2, _ = m(f1, conds[1], g, prev=x0)  # no columns at step 2
        f1_none, _ = m(x0, conds[0], g)
    assert torch.allclose(out[0].s, f1.s) and torch.allclose(out[1].s, f2.s, atol=1e-6)
    assert not torch.allclose(f1.s, f1_none.s)  # the columns do reach the forecast
    # zeros + flag 0 is what "not available" means
    z = m.vnet.column_input(None, 2, 12 * 24, x0)
    assert z.s.shape[-1] == 9 and float(z.s.abs().sum()) == 0
    one = m.vnet.column_input(col, 2, 12 * 24, x0)
    assert torch.all(one.s[..., -1] == 1)
    # a model without columns has the old input width
    assert _model(False).vnet.lift is not None and not _model(False).vnet.use_col


def test_stage15a_pipeline(files, tmp_path):
    base, col = files
    run_dir = tmp_path / "run_s15a"
    flags = ["--fusion", "pigt", "--pigt-layers", "1", "--pigt-ops", "1", "--curvature-ops", "1", "--prev-state", "1",
             "--column-inputs", "fcn10", "--column-data", str(col)]
    roll = ("--rollout-steps", "2", "--rollout-start", "2")
    out = _run(run_dir, base, "b_noguide", *flags, "--steps", "3", "--val-every", "2", *roll)
    assert "column inputs fcn10: 8 scalar + 1 vector channels" in out
    cfg = json.load(open(run_dir / "config.json"))
    assert cfg["column_inputs"] == "fcn10" and cfg["column_data"] == str(col)
    stats = json.load(open(run_dir / "column_norm.json"))
    assert len(stats["s_mean"]) == 8 and len(stats["v_scale"]) == 1
    _run(run_dir, base, "a", "--steps", "3", "--val-every", "2")
    _run(run_dir, base, "b", "--steps", "3", "--val-every", "2", *roll)
    _run(run_dir, base, "c", "--steps", "2", "--val-every", "2", "--noise-constraints", "1")
    _run(run_dir, base, "eval", "--eval-stride", "4", "--max-lead-steps", "2", "--n-ens-cases", "2", "--members", "2",
         "--sample-steps", "2")
    ev = json.load(open(run_dir / "eval.json"))
    assert {"lie_guided", "no_guide"} <= set(ev["rmse"]["1"])
    assert all(np.isfinite(v) for v in ev["rmse"]["2"]["lie_guided"].values())
    # external year: the column file of that year is required
    cmd = [sys.executable, "-W", "ignore", str(ROOT / "scripts/train_eval.py"), "--run", str(run_dir), "--data",
           str(base), "--stage", "eval", "--threads", "1", "--eval-data", str(base), "--eval-tag", "x",
           "--n-ens-cases", "0", "--max-lead-steps", "1"]
    bad = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert bad.returncode != 0 and "--eval-column-data" in bad.stderr
    _run(run_dir, base, "eval", "--eval-stride", "4", "--max-lead-steps", "2", "--n-ens-cases", "0",
         "--eval-data", str(base), "--eval-column-data", str(col), "--eval-tag", "bench", "--max-cases", "2")
    assert (run_dir / "eval_bench.json").exists()
    d = subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "scripts/diagnose_guide.py"), "--run", str(run_dir),
                        "--threads", "1"], capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert d.returncode == 0, d.stdout[-2000:] + d.stderr[-4000:]
