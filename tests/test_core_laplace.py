"""stage18d Laplace block (PhyMPGN-style) in the graph tendency core: operator, capped diffusion, warm start, pipeline."""

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
from gpclimate.nn.graph_core import GridFaceGraph  # noqa: E402
from gpclimate.nn.physics_ops import EARTH_RADIUS  # noqa: E402
from test_graph_core import _core_forecaster  # noqa: E402
from test_column_inputs import _write_columns  # noqa: E402
from test_train_eval_smoke import _run, _write_data  # noqa: E402
from test_transport_guide import GUIDE, H, LAT, LON, W, _graphs, _state  # noqa: E402

LA, LO = np.deg2rad(LAT)[:, None], np.deg2rad(LON)[None, :]


def _faces():
    return GridFaceGraph(torch.as_tensor(LAT), torch.as_tensor(LON))


def test_covering_laplacian_reproduces_spherical_harmonics():
    """(1/A_i) sum_f (l/d)(q_j - q_i) is the Laplace-Beltrami operator: Y_lm -> -l(l+1)/R^2 Y_lm."""
    fg = _faces()
    mid = (np.abs(LAT) < 60)[:, None].repeat(W, 1).reshape(-1)
    for f, ell in ((np.sin(LA) * np.ones_like(LO), 1), (np.cos(LA) * np.cos(LO), 1),
                   (np.cos(LA) ** 2 * np.cos(2 * LO), 2), (np.cos(LA) * (5 * np.sin(LA) ** 2 - 1) * np.sin(LO), 3)):
        q = torch.as_tensor(f.reshape(1, -1, 1), dtype=torch.float32)
        lap = fg.laplacian(q)[0, :, 0].numpy() * EARTH_RADIUS ** 2
        ref = -ell * (ell + 1) * f.reshape(-1)
        assert np.abs(lap - ref)[mid].max() < 0.05 * np.abs(ref).max(), ell
        assert np.abs(lap - ref).max() < 0.15 * np.abs(ref).max(), ell


def test_capped_diffusion_is_conservative_monotone_and_linear_for_small_k():
    fg = _faces()
    torch.manual_seed(0)
    q = torch.randn(2, H * W, 3)
    A = fg.area[None, :, None]
    d = fg.diffusion(q, torch.full((2, len(fg.src), 3), 1e9), 6 * 3600.0)  # far beyond the explicit limit
    assert float((d * A).sum(1).abs().max()) < 1e-6 * float((q.abs() * A).sum(1).max())
    q1 = q + d
    assert float(q1.max()) <= float(q.max()) + 1e-6 and float(q1.min()) >= float(q.min()) - 1e-6  # no new extrema
    assert float(q1.std()) < float(q.std())
    f = torch.as_tensor((np.cos(LA) ** 2 * np.cos(2 * LO)).reshape(1, -1, 1), dtype=torch.float32)
    K = torch.full((1, len(fg.src), 1), 1e3)
    lin = 6 * 3600.0 * 1e3 * fg.laplacian(f)
    assert float((fg.diffusion(f, K, 6 * 3600.0) - lin).abs().max()) < 1e-3 * float(lin.abs().max())


def test_laplace_block_starts_near_off_and_acts():
    g = _graphs().view()
    x, c = _state(2, wind=0.5)
    plain = _core_forecaster(core=True)
    torch.manual_seed(0)
    cfg = VNetConfig(**{**plain.vnet.cfg.__dict__, "core_laplace": True})
    m = LieGuidedForecaster(DEFAULT_SPEC, GUIDE, cfg).eval()
    missing, unexpected = m.vnet.load_state_dict(plain.vnet.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("laplace_head") for k in missing), missing
    m.guide.load_state_dict(plain.guide.state_dict())
    with torch.no_grad():
        f_plain, _ = plain(x, c, g)
        f_lap, _ = m(x, c, g)
        rel = float((f_lap.s - f_plain.s).abs().mean() / (f_plain.s - x.s).abs().mean())
        assert rel < 0.01, rel  # K ~ 700 m^2/s at the start: the stage18 core is kept
        d0 = m.last_core["diffusion"]
        assert d0 > 0 and all(300 < k < 1500 for k in m.last_core["K"])
        m.vnet.laplace_head.lin_s.bias.fill_(0.5)  # K ~ 4e5 m^2/s: smooths, conserves the area integrals
        f_on, _ = m(x, c, g)
        assert m.last_core["diffusion"] > 10 * d0 and not torch.allclose(f_on.s, f_lap.s)
        emb = m.vnet.last_emb
        fg = m.faces(g)
        D = fg.diffusion(x.s, fg.face_mean(m.vnet.core_diffusivity(emb)), 6 * 3600.0)
        AD = D * fg.area[None, :, None]
        assert float(AD.sum(1).abs().max()) < 1e-5 * float(AD.abs().sum(1).max())
    with pytest.raises(ValueError):  # the Laplace block is part of the tendency core
        LieGuidedForecaster(DEFAULT_SPEC, GUIDE, VNetConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), heads=2, ds=4, dv=2,
                                                            hidden=16, n_basis=4, fusion="pigt", pigt_layers=1,
                                                            integrator="ode", core_laplace=True))


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("lap")
    base, col = d / "synthetic.npz", d / "columns.npz"
    _write_data(base)
    _write_columns(col, base)
    return base, col


def test_stage18d_pipeline(files, tmp_path):
    base, col = files
    common = ["--fusion", "pigt", "--pigt-layers", "1", "--pigt-ops", "1", "--curvature-ops", "1", "--prev-state", "1",
              "--column-inputs", "fcn10", "--column-data", str(col)]
    roll = ("--rollout-steps", "2", "--rollout-start", "0")
    src = tmp_path / "src"
    _run(src, base, "b_noguide", *common, "--steps", "2", "--val-every", "2")
    _run(src, base, "a", "--steps", "2", "--val-every", "2")
    _run(src, base, "b", "--steps", "2", "--val-every", "2")
    run_dir = tmp_path / "run_s18d"
    run_dir.mkdir()
    for f in ("normalizer.pt", "column_norm.json"):
        (run_dir / f).write_bytes((src / f).read_bytes())
    flags = common + ["--flow-head", "1", "--integrator", "ode", "--ode-steps", "2", "--ode-drift", "transport",
                      "--ode-path-forcing", "1", "--inter-times", "1", "--tendency-core", "1", "--core-laplace", "1"]
    _run(run_dir, base, "a", *flags, "--steps", "2", "--val-every", "2", "--init-guide", str(src / "guide.pt"),
         "--guide-heads-only", "1")
    assert json.load(open(run_dir / "config.json"))["core_laplace"] == 1
    _run(run_dir, base, "b_noguide", "--steps", "2", "--val-every", "2", *roll, "--init-vnet", str(src / "vnet_noguide.pt"))
    out = _run(run_dir, base, "b", "--steps", "2", "--val-every", "2", *roll, "--init-vnet", str(src / "vnet.pt"))
    assert "Laplace block" in out and " D " in out
    rows = [json.loads(line) for line in open(run_dir / "log_vnet.jsonl")]
    assert any("diffusion" in r.get("core", {}) for r in rows)
    _run(run_dir, base, "eval", "--eval-stride", "4", "--max-lead-steps", "2", "--n-ens-cases", "0")
    ev = json.load(open(run_dir / "eval.json"))
    for name in ("lie_guided", "no_guide", "physics_only"):
        assert all(np.isfinite(v) for v in ev["rmse"]["2"][name].values()), name
