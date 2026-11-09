"""stage18e edge-preserving (total-variation flow) diffusion in the Laplace block: factor, operator, warm start,
pipeline."""

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
from test_column_inputs import _write_columns  # noqa: E402
from test_core_laplace import _faces  # noqa: E402
from test_graph_core import _core_forecaster  # noqa: E402
from test_train_eval_smoke import _run, _write_data  # noqa: E402
from test_transport_guide import GUIDE, H, W, _graphs, _state  # noqa: E402


def test_edge_factor_is_linear_for_small_and_bounded_for_large_differences():
    eps = torch.tensor([0.5, 2.0])
    dq = torch.linspace(-50, 50, 1001)[None, :, None].expand(1, -1, 2)
    g = GridFaceGraph.edge_factor(dq, eps)
    assert float(g.min()) > 0 and float(g.max()) <= 1.0 + 1e-6
    small = dq.abs() < 0.05 * eps
    assert float((1 - g[small]).max()) < 2e-3  # linear diffusion where the difference is small
    flux = (g * dq).abs()
    assert float((flux - eps).max()) < 1e-6  # the flux never exceeds eps: eps sign(dq) across a jump
    assert float(flux[0, -1, 0]) > 0.999 * 0.5


def test_tv_diffusion_keeps_a_front_and_stays_conservative_and_monotone():
    fg = _faces()
    torch.manual_seed(0)
    dt, K = 6 * 3600.0, torch.full((1, len(fg.src), 1), 1e3)  # below the explicit limit everywhere
    eps = torch.tensor([0.1])
    # a zonal front (two plateaus 20 eps apart, jumps at W/2 and at the periodic seam) plus noise of ~ eps / 10
    q = (2.0 * (torch.arange(W) >= W // 2).float())[None, :].expand(H, W).reshape(1, -1, 1) \
        + 0.01 * torch.randn(1, H * W, 1)
    lin, tv = fg.diffusion(q, K, dt), fg.diffusion(q, K, dt, edge_eps=eps)
    A = fg.area[None, :, None]
    assert float((tv * A).sum(1).abs().max()) < 1e-6 * float((q.abs() * A).sum(1).max())  # area integral conserved
    q1 = q + tv
    assert float(q1.max()) <= float(q.max()) + 1e-6 and float(q1.min()) >= float(q.min()) - 1e-6  # no new extrema
    front = torch.zeros(H, W, dtype=torch.bool)
    front[:, [W // 2 - 1, W // 2, 0, W - 1]] = True
    front = front.reshape(-1)
    lin_f, tv_f = lin[0, front, 0], tv[0, front, 0]
    assert float(tv_f.abs().max()) < 0.1 * float(lin_f.abs().max())  # the jump is kept (factor ~ eps / 2 = 0.05)
    lin_i, tv_i = lin[0, ~front, 0], tv[0, ~front, 0]
    assert float((tv_i - lin_i).abs().max()) < 0.1 * float(lin_i.abs().max())  # the noise is diffused as before
    assert float(tv_i.abs().max()) > 0


def test_tv_block_starts_as_the_laplace_block_and_learns_its_scale():
    g = _graphs().view()
    x, c = _state(2, wind=0.5)
    plain = _core_forecaster(core=True)
    lap_cfg = VNetConfig(**{**plain.vnet.cfg.__dict__, "core_laplace": True})
    torch.manual_seed(0)
    lap = LieGuidedForecaster(DEFAULT_SPEC, GUIDE, lap_cfg).eval()
    lap.vnet.load_state_dict(plain.vnet.state_dict(), strict=False)
    lap.guide.load_state_dict(plain.guide.state_dict())
    with torch.no_grad():
        lap.vnet.laplace_head.lin_s.bias.fill_(0.5)  # K ~ 4e5 m^2/s: the diffusion is visible
    tv_cfg = VNetConfig(**{**lap_cfg.__dict__, "core_tv": True})
    m = LieGuidedForecaster(DEFAULT_SPEC, GUIDE, tv_cfg).eval()
    missing, unexpected = m.vnet.load_state_dict(lap.vnet.state_dict(), strict=False)
    assert not unexpected and sorted(missing) == ["tv_eps0", "tv_log_gain"], (missing, unexpected)
    m.guide.load_state_dict(lap.guide.state_dict())
    with torch.no_grad():
        f_lap, _ = lap(x, c, g)
        d_lap = lap.last_core["diffusion"]
        m.vnet.tv_eps0.fill_(1e3)  # eps >> every difference: factor 1, the Laplace block is recovered
        f_same, _ = m(x, c, g)
        assert torch.allclose(f_same.s, f_lap.s, atol=1e-5), float((f_same.s - f_lap.s).abs().max())
        assert all(k > 0.999 for k in m.last_core["tv_keep"]) and len(m.last_core["tv_eps"]) == DEFAULT_SPEC.n_s
        m.vnet.tv_eps0.fill_(0.05)  # fronts in the state: less diffusion than the linear block, same conservation
        f_tv, _ = m(x, c, g)
        assert 0 < m.last_core["diffusion"] < d_lap
        assert all(0 < k < 0.999 for k in m.last_core["tv_keep"]) and all(v == 1.0 for v in m.last_core["tv_gain"])
        assert not torch.allclose(f_tv.s, f_lap.s)
        fg = m.faces(g)
        D = fg.diffusion(x.s, fg.face_mean(m.vnet.core_diffusivity(m.vnet.last_emb)), 6 * 3600.0,
                         edge_eps=m.vnet.core_tv_eps())
        AD = D * fg.area[None, :, None]
        assert float(AD.sum(1).abs().max()) < 1e-5 * float(AD.abs().sum(1).max())
    f, _ = m(x, c, g)  # the edge scale is learned: the loss reaches its factor
    f.s.pow(2).mean().backward()
    assert m.vnet.tv_log_gain.grad is not None and float(m.vnet.tv_log_gain.grad.abs().max()) > 0
    with pytest.raises(ValueError):  # the edge-preserving form needs the Laplace block
        LieGuidedForecaster(DEFAULT_SPEC, GUIDE, VNetConfig(**{**plain.vnet.cfg.__dict__, "core_tv": True}))
    pair = LieGuidedForecaster(DEFAULT_SPEC, GUIDE, VNetConfig(**{**tv_cfg.__dict__, "use_guide": False}))
    assert pair.vnet.tv_log_gain is None  # the guide-free pair has no core and hence no edge scale


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("tv")
    base, col = d / "synthetic.npz", d / "columns.npz"
    _write_data(base)
    _write_columns(col, base)
    return base, col


def test_stage18e_pipeline(files, tmp_path):
    base, col = files
    common = ["--fusion", "pigt", "--pigt-layers", "1", "--pigt-ops", "1", "--curvature-ops", "1", "--prev-state", "1",
              "--column-inputs", "fcn10", "--column-data", str(col)]
    roll = ("--rollout-steps", "2", "--rollout-start", "0")
    src = tmp_path / "src"
    _run(src, base, "b_noguide", *common, "--steps", "2", "--val-every", "2")
    _run(src, base, "a", "--steps", "2", "--val-every", "2")
    _run(src, base, "b", "--steps", "2", "--val-every", "2")
    run_dir = tmp_path / "run_s18e"
    run_dir.mkdir()
    for f in ("normalizer.pt", "column_norm.json"):
        (run_dir / f).write_bytes((src / f).read_bytes())
    flags = common + ["--flow-head", "1", "--integrator", "ode", "--ode-steps", "2", "--ode-drift", "transport",
                      "--ode-path-forcing", "1", "--inter-times", "1", "--tendency-core", "1", "--core-laplace", "1",
                      "--core-tv", "1"]
    out_a = _run(run_dir, base, "a", *flags, "--steps", "2", "--val-every", "2", "--init-guide", str(src / "guide.pt"),
                 "--guide-heads-only", "1")
    cfg = json.load(open(run_dir / "config.json"))
    assert cfg["core_laplace"] == 1 and cfg["core_tv"] == 1
    _run(run_dir, base, "b_noguide", "--steps", "2", "--val-every", "2", *roll, "--init-vnet", str(src / "vnet_noguide.pt"))
    out = _run(run_dir, base, "b", "--steps", "2", "--val-every", "2", *roll, "--init-vnet", str(src / "vnet.pt"))
    assert "edge-preserving" in out and " tv gain " in out
    assert "median neighbour difference" in out_a + out  # the preprocessing runs once per run directory (cached)
    eps = json.load(open(run_dir / "tv_eps.json"))
    assert len(eps["eps"]) == DEFAULT_SPEC.n_s and all(e > 0 for e in eps["eps"])
    rows = [json.loads(line) for line in open(run_dir / "log_vnet.jsonl")]
    assert any("tv_gain" in r.get("core", {}) for r in rows)
    saved = torch.load(run_dir / "vnet.pt")
    assert torch.allclose(saved["tv_eps0"], torch.tensor(eps["eps"]))  # the preprocessing scale travels with the weights
    _run(run_dir, base, "eval", "--eval-stride", "4", "--max-lead-steps", "2", "--n-ens-cases", "0")
    ev = json.load(open(run_dir / "eval.json"))
    for name in ("lie_guided", "no_guide", "physics_only"):
        assert all(np.isfinite(v) for v in ev["rmse"]["2"][name].values()), name
        assert all(np.isfinite(v) and v > 0 for v in ev["sharpness"]["2"][name].values()), name
    assert "경사 선명도" in (run_dir / "eval.md").read_text()
