"""stage17 dynamic guide: the ODE integrator re-evaluates the guide and the fixed operators on every 6 h path state,
with the solar forcing of the sub-step time, the stage14/15a inputs (previous change, column inputs at t0), path
supervision of X_{t + k 6 h}, a zero-initialised learnable gain on the guide's tendency drift (stage17b), and paired
fine-tuning from a trained delta V-Net (--init-vnet)."""

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

from gpclimate.data.era5_wb2 import DEFAULT_SPEC  # noqa: E402
from gpclimate.data.variables import insolation_proxy  # noqa: E402
from gpclimate.geometry.graphs import SphereGraphs  # noqa: E402
from gpclimate.models.dynamics_vnet import LieGuidedForecaster, VNetConfig  # noqa: E402
from gpclimate.models.lie_physical_gnn import GuideConfig  # noqa: E402
from gpclimate.nn.so2 import Field  # noqa: E402
from test_column_inputs import _write_columns  # noqa: E402
from test_train_eval_smoke import _run, _write_data  # noqa: E402

H, W = 12, 24
LAT, LON = np.linspace(-87.1875, 87.1875, H), np.arange(W) * (360.0 / W)
GUIDE = GuideConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), hidden=16, n_basis=4, physics_ops=True, tend_head=True)


def _graph():
    return SphereGraphs(LAT, LON, mesh_splits=(2, 1, 0)).view()


def _model(integrator="ode", **kw):
    torch.manual_seed(0)
    cfg = VNetConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), heads=2, ds=4, dv=2, hidden=16, n_basis=4, fusion="pigt",
                     pigt_layers=1, prev_state=True, col_s=2, col_v=1, integrator=integrator, ode_steps=4, **kw)
    m = LieGuidedForecaster(DEFAULT_SPEC, GUIDE, cfg)
    with torch.no_grad():
        m.vnet.head_out.lin_s.weight.normal_(0, 0.1)
        m.vnet.head_out.w_re.normal_(0, 0.1)
    return m.eval()


def _cond(times):
    """Normalised-like condition Field [B, H*W, 5 scalars + 1 vector] with the real solar channels."""
    B = len(times)
    toa = torch.as_tensor(insolation_proxy(np.asarray(times, "datetime64[ns]"), LAT, LON)).flatten(1)
    toa1 = torch.as_tensor(insolation_proxy(np.asarray(times, "datetime64[ns]") + np.timedelta64(24, "h"), LAT, LON)).flatten(1)
    torch.manual_seed(2)
    s = torch.stack([torch.randn(B, H * W), torch.rand(B, H * W), toa, toa1, torch.randn(B, H * W)], -1)
    return Field(s, torch.randn(B, H * W, 1, dtype=torch.complex64))


def _inputs(B=2):
    torch.manual_seed(1)
    rf = lambda n_s, n_v: Field(torch.randn(B, H * W, n_s), torch.randn(B, H * W, n_v, dtype=torch.complex64))  # noqa: E731
    times = [np.datetime64("2019-03-01T00"), np.datetime64("2019-07-15T18")][:B]
    return rf(5, 3), _cond(times), rf(5, 3), rf(2, 1), times  # col: 2 scalars + 1 vector (flag added inside)


def test_sub_step_forcing_is_the_insolation_at_the_sub_step_time():
    g = _graph()
    m = _model(ode_path_forcing=True)
    _, c, _, _, times = _inputs()
    assert m.sub_cond(c, g, 0) is c
    for k in (1, 2, 3):
        ck = m.sub_cond(c, g, k)
        exact = _cond([t + np.timedelta64(6 * k, "h") for t in times])
        # solar channels move with the sub-step time (only the declination drift, < 0.4 deg per day, is ignored);
        # rotating the wrong way would be off by O(1)
        assert torch.allclose(ck.s[..., 2:4], exact.s[..., 2:4], atol=1e-2)
        assert torch.equal(ck.s[..., [0, 1, 4]], c.s[..., [0, 1, 4]]) and torch.equal(ck.v, c.v)
        assert not torch.allclose(ck.s[..., 2], c.s[..., 2], atol=0.1)
    assert _model().sub_cond(c, g, 2) is c  # flag off: conditions held fixed (stage8 behaviour)


def test_ode_path_uses_prev_and_initial_columns_and_is_supervisable():
    g = _graph()
    m = _model(ode_path_forcing=True)
    x0, c, prev, col, _ = _inputs()
    with torch.no_grad():
        f, d = m(x0, c, g, prev=prev, col=col)
        path = m.inter_forecasts(x0)
        f_noprev, _ = m(x0, c, g, col=col)
        f_nocol, _ = m(x0, c, g, prev=prev)
    assert len(path) == 3 and all(p.s.shape == x0.s.shape for p in path)
    assert not torch.allclose(path[0].s, path[2].s) and not torch.allclose(path[2].s, f.s)
    assert not torch.allclose(f.s, f_noprev.s) and not torch.allclose(f.s, f_nocol.s)  # both inputs reach the forecast
    assert torch.allclose(f.s, x0.s + d.s * m.delta_scale_s, atol=1e-5)
    with pytest.raises(ValueError):
        m(x0, c, g, nudge=prev)


def test_columns_reach_only_the_first_sub_step():
    g = _graph()
    m = _model()
    x0, c, prev, col, _ = _inputs()
    seen = []
    orig = m.vnet.forward

    def spy(*args, **kw):
        seen.append(kw.get("col") is not None)
        return orig(*args, **kw)

    m.vnet.forward = spy
    with torch.no_grad():
        m(x0, c, g, prev=prev, col=col)
    assert seen == [True, False, False, False]


def test_drift_gain_starts_at_zero_and_warm_start_from_delta():
    g = _graph()
    x0, c, prev, col, _ = _inputs()
    plain = _model(ode_path_forcing=True)
    gain = _model(ode_path_forcing=True, ode_physics=True, ode_drift_gain=True)
    missing, unexpected = gain.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and sorted(missing) == ["vnet.drift_gain_s", "vnet.drift_gain_v"]
    with torch.no_grad():
        f0, _ = plain(x0, c, g, prev=prev, col=col)
        f1, _ = gain(x0, c, g, prev=prev, col=col)
        assert torch.allclose(f0.s, f1.s, atol=1e-6) and torch.allclose(f0.v, f1.v, atol=1e-6)  # gain 0 = no drift
        phys = gain.physics_only(x0, c, g)  # diagnostic ignores the gain: the guide's own tendency integrated
        assert not torch.allclose(phys.s, x0.s)
        gain.vnet.drift_gain_s.fill_(0.5)
        gain.vnet.drift_gain_v.fill_(0.5)
        f2, _ = gain(x0, c, g, prev=prev, col=col)
    assert not torch.allclose(f2.s, f0.s)
    # the guide-free pair has no guide, hence no drift and no gain: it loads stage17a's pair weights unchanged
    torch.manual_seed(0)
    pair_cfg = VNetConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), heads=2, ds=4, dv=2, hidden=16, n_basis=4, use_guide=False,
                          prev_state=True, col_s=2, col_v=1, integrator="ode", ode_steps=4, ode_path_forcing=True)
    pair_a = LieGuidedForecaster(DEFAULT_SPEC, GUIDE, pair_cfg)
    pair_b = LieGuidedForecaster(DEFAULT_SPEC, GUIDE, VNetConfig(**{**pair_cfg.__dict__, "ode_physics": True,
                                                                     "ode_drift_gain": True}))
    assert pair_b.vnet.drift_gain_s is None
    pair_b.vnet.load_state_dict(pair_a.vnet.state_dict())  # strict
    with torch.no_grad():
        fa, _ = pair_a.eval()(x0, c, g, prev=prev, col=col)
        fb, _ = pair_b.eval()(x0, c, g, prev=prev, col=col)
    assert torch.allclose(fa.s, fb.s)
    delta = _model("delta")  # a delta-trained V-Net has exactly the ODE V-Net's weights (stage17 fine-tunes from it)
    missing, unexpected = gain.vnet.load_state_dict(delta.vnet.state_dict(), strict=False)
    assert not unexpected and sorted(missing) == ["drift_gain_s", "drift_gain_v"]


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("dyn")
    base, col = d / "synthetic.npz", d / "columns.npz"
    _write_data(base)
    _write_columns(col, base)
    return base, col


def test_stage17_pipeline_fine_tunes_from_a_delta_run(files, tmp_path):
    base, col = files
    common = ["--fusion", "pigt", "--pigt-layers", "1", "--pigt-ops", "1", "--curvature-ops", "1", "--prev-state", "1",
              "--column-inputs", "fcn10", "--column-data", str(col)]
    roll = ("--rollout-steps", "2", "--rollout-start", "0")
    src = tmp_path / "src"  # the delta run being fine-tuned (stands in for stage9b)
    _run(src, base, "b_noguide", *common, "--steps", "2", "--val-every", "2")
    _run(src, base, "a", "--steps", "2", "--val-every", "2")
    _run(src, base, "b", "--steps", "2", "--val-every", "2")
    run_dir = tmp_path / "run_s17"  # SMALL uses lead 2 (12 h) -> 2 sub-steps of 6 h, 1 supervised path state
    run_dir.mkdir()
    for f in ("normalizer.pt", "guide.pt", "a_done.json", "column_norm.json"):
        (run_dir / f).write_bytes((src / f).read_bytes())
    flags = common + ["--integrator", "ode", "--ode-steps", "2", "--ode-physics", "1", "--ode-drift-gain", "1",
                      "--ode-path-forcing", "1", "--inter-times", "1", "--inter-weight", "1.0"]
    out = _run(run_dir, base, "b", *flags, "--steps", "3", "--val-every", "2", *roll, "--init-vnet", str(src / "vnet.pt"))
    assert "ODE path supervision: 1 path states (6 h), weight 1.0" in out
    assert "V-Net initialised from" in out and "drift_gain_s" in out
    assert "sub-step solar forcing" in out and " gain " in out
    cfg = json.load(open(run_dir / "config.json"))
    assert cfg["integrator"] == "ode" and cfg["ode_path_forcing"] == 1 and cfg["ode_drift_gain"] == 1
    rows = [json.loads(line) for line in open(run_dir / "log_vnet.jsonl")]
    assert any("l_int" in r for r in rows) and any(len(r.get("drift_gain", [])) == 8 for r in rows)
    # guide-free pair of the same ODE (no guide -> no drift, no gain), fine-tuned from the delta pair
    out = _run(run_dir, base, "b_noguide", "--steps", "2", "--val-every", "2", *roll,
               "--init-vnet", str(src / "vnet_noguide.pt"))
    assert "V-Net initialised from" in out and "(new:" not in out
    _run(run_dir, base, "eval", "--eval-stride", "4", "--max-lead-steps", "2", "--n-ens-cases", "0")
    ev = json.load(open(run_dir / "eval.json"))
    for name in ("lie_guided", "no_guide", "physics_only"):
        assert all(np.isfinite(v) for v in ev["rmse"]["2"][name].values()), name
    out = subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "scripts/diagnose_guide.py"), "--run", str(run_dir),
                          "--threads", "1"], capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]


def test_ode_path_supervision_needs_6h_sub_steps(files, tmp_path):
    base, _ = files
    cmd = [sys.executable, "-W", "ignore", str(ROOT / "scripts/train_eval.py"), "--run", str(tmp_path / "bad"),
           "--data", str(base), "--stage", "b_noguide", "--threads", "1", "--steps", "1", "--lead", "4",
           "--integrator", "ode", "--ode-steps", "2", "--inter-times", "3"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert r.returncode != 0 and "ODE path supervision needs ode_steps == lead" in r.stderr
