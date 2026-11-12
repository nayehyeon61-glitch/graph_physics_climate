"""stage17c Lie transport guide: semi-Lagrangian transport on the sphere with parallel-transported vectors, the guide's
flow head (level wind + learned correction), gauge equivariance, the transport drift of the dynamic-guide ODE, and the
Stage A -> Stage B pipeline (guide fine-tuned with the 6 h transport + source loss, ODE with transport drift)."""

import json
import math
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
from gpclimate.geometry.graphs import SphereGraphs  # noqa: E402
from gpclimate.models.dynamics_vnet import LieGuidedForecaster, VNetConfig  # noqa: E402
from gpclimate.models.lie_physical_gnn import GuideConfig, LiePhysicalGNN  # noqa: E402
from gpclimate.nn.semi_lagrangian import SphereSemiLagrangian  # noqa: E402
from gpclimate.nn.so2 import Field  # noqa: E402
from test_column_inputs import _write_columns  # noqa: E402
from test_train_eval_smoke import _run, _write_data  # noqa: E402

H, W = 16, 32
LAT, LON = np.linspace(-84.375, 84.375, H), np.arange(W) * (360.0 / W)
GUIDE = GuideConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), hidden=16, n_basis=4, physics_ops=True, tend_head=True,
                    flow_head=True)


def _sphere_fields():
    sl = SphereSemiLagrangian(LAT, LON)
    return sl, sl.p


def test_semi_lagrangian_identity_rotation_and_parallel_transport():
    sl, p = _sphere_fields()
    f = torch.randn(2, 3, H, W)
    v = torch.randn(2, 2, H, W, dtype=torch.complex64)
    s0, v0 = sl(f, v, torch.zeros(2, H, W, dtype=torch.complex64))
    assert torch.allclose(s0, f, atol=1e-4) and torch.allclose(v0, v, atol=1e-4)  # no flow, no change
    # rigid rotation about an arbitrary axis: the transported field is the rotated field (up to O(alpha^2))
    k = torch.tensor([0.3, -0.5, 0.81])
    k = k / k.norm()
    alpha = 0.08
    D = alpha * torch.cross(k.expand_as(p), p, dim=-1)
    disp = torch.complex((D * sl.east).sum(-1), (D * sl.north).sum(-1))[None]

    def rot(x, a):
        return x * math.cos(a) + torch.cross(k.expand_as(x), x, dim=-1) * math.sin(a) + k * (x @ k)[..., None] * (1 - math.cos(a))

    fun = lambda x: x[..., 0] + 0.5 * x[..., 1] * x[..., 2] - x[..., 2] ** 2  # noqa: E731
    out, _ = sl(fun(p)[None, None], torch.zeros(1, 0, H, W, dtype=torch.complex64), disp)
    err = (out[0, 0] - fun(rot(p, -alpha)))[3:-3]
    assert float(err.abs().max()) < 0.02 * float(fun(p).max() - fun(p).min())
    # vectors: a zonal solid-body flow turns an eastward field by -dlambda sin(lat) (parallel transport = the metric
    # terms of the momentum equation); magnitudes are preserved exactly
    dlam = 0.1
    lat = torch.deg2rad(torch.tensor(LAT, dtype=torch.float32))
    dz = torch.complex(dlam * lat.cos()[:, None].expand(H, W), torch.zeros(H, W))[None]
    _, vo = sl(torch.zeros(1, 0, H, W), torch.ones(1, 1, H, W, dtype=torch.complex64), dz)
    assert torch.allclose(torch.angle(vo[0, 0, :, 0]), -dlam * lat.sin(), atol=2e-3)
    assert torch.allclose(vo.abs(), torch.ones_like(vo.abs()), atol=1e-5)


def _graphs():
    return SphereGraphs(LAT, LON, mesh_splits=(2, 1, 0))


def _guide():
    torch.manual_seed(0)
    gd = LiePhysicalGNN(DEFAULT_SPEC, GUIDE)
    gd.flow_v_scale.fill_(10.0)
    return gd.eval()


def _state(B=2, wind=0.0):
    torch.manual_seed(1)
    s = torch.randn(B, H * W, 5)
    v = torch.randn(B, H * W, 3, dtype=torch.complex64) * 0.3 + wind
    return Field(s, v), Field(torch.randn(B, H * W, 5), torch.randn(B, H * W, 1, dtype=torch.complex64))


def test_flow_head_starts_at_the_level_winds_and_moves_fields_downstream():
    gd, gr = _guide(), _graphs()
    g = gr.view()
    x, c = _state(1)
    with torch.no_grad():
        feats = gd(x, c, torch.zeros(1), g)
        assert torch.allclose(gd.flow(feats, x, g), x.v * 10.0)  # zero-initialised correction
        still = Field(x.s, torch.zeros_like(x.v))
        xt = gd.transport(feats, still, g)
    assert torch.allclose(xt.s, still.s, atol=1e-4) and torch.allclose(xt.v, still.v, atol=1e-4)  # calm: no motion
    # uniform 20 m/s westerly at 850 hPa carries a t850 wave east by 20 * 6 h / (a cos lat)
    lon = torch.deg2rad(torch.tensor(LON, dtype=torch.float32))
    s = torch.zeros(1, H, W, 5)
    s[..., 1] = torch.cos(lon)  # t850 (group 850 hPa)
    v = torch.zeros(1, H * W, 3, dtype=torch.complex64)
    v[..., 0] = 2.0  # u850 = 2 normalised units = 20 m/s
    xw = Field(s.reshape(1, H * W, 5), v)
    with torch.no_grad():
        out = gd.transport(gd(xw, c, torch.zeros(1), g), xw, g).s.reshape(1, H, W, 5)
    i = H // 2  # latitude row closest to the equator
    shift = 20.0 * 6 * 3600 / (6.371e6 * math.cos(math.radians(LAT[i])))
    expect = torch.cos(lon - shift)
    assert torch.allclose(out[0, i, :, 1], expect, atol=0.02)
    assert torch.allclose(out[0, i, :, 0], s[0, i, :, 0])  # z500 (500 hPa group, calm) unchanged


def test_transport_is_gauge_equivariant():
    gd, gr = _guide(), _graphs()
    with torch.no_grad():
        gd.flow_out.w_re.normal_(0, 0.3)
        gd.flow_out.w_im.normal_(0, 0.3)
    x, c = _state(1, wind=0.5)
    gen = torch.Generator().manual_seed(3)
    tg = torch.rand(gr.grid.n, generator=gen) * 2 * math.pi
    tm = [torch.rand(m.n, generator=gen) * 2 * math.pi for m in gr.mesh]
    g0, gt = gr.view(), gr.gauge_transformed(tm, tg)
    with torch.no_grad():
        y0 = gd.transport(gd(x, c, torch.zeros(1), g0), x, g0)
        xg, cg = x.gauge(tg), c.gauge(tg)
        yt = gd.transport(gd(xg, cg, torch.zeros(1), gt), xg, gt)
    ref = y0.gauge(tg)
    assert torch.allclose(yt.s, ref.s, atol=2e-4) and torch.allclose(yt.v, ref.v, atol=2e-4)


def _forecaster(drift="transport", gain=True):
    torch.manual_seed(0)
    cfg = VNetConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), heads=2, ds=4, dv=2, hidden=16, n_basis=4, fusion="pigt",
                     pigt_layers=1, integrator="ode", ode_steps=4, ode_physics=True, ode_drift_gain=gain,
                     ode_drift=drift, ode_path_forcing=True)
    m = LieGuidedForecaster(DEFAULT_SPEC, GUIDE, cfg)
    m.guide.flow_v_scale.fill_(10.0)
    with torch.no_grad():
        m.vnet.head_out.lin_s.weight.normal_(0, 0.1)
    return m.eval()


def test_transport_drift_in_the_ode():
    g = _graphs().view()
    x, c = _state(2, wind=0.5)
    m = _forecaster()
    with torch.no_grad():
        f0, _ = m(x, c, g)
        P = m.physics(x, c, g)
        tr, src = m.transport_drift(x, g, P, parts=True)
        assert float(tr.s.abs().mean()) > 1e-3  # the winds move the fields
        phys = m.physics_only(x, c, g)  # the guide alone, gain ignored: a forecast of its own
        assert not torch.allclose(phys.s, x.s)
        m.vnet.drift_gain_s.fill_(1.0)
        m.vnet.drift_gain_v.fill_(1.0)
        f1, _ = m(x, c, g)
    assert not torch.allclose(f0.s, f1.s)  # gain 0 = stage17a dynamics, gain 1 = transport guide in the dynamics
    with pytest.raises(ValueError):  # transport drift needs the flow head
        cfg = VNetConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), heads=2, ds=4, dv=2, hidden=16, n_basis=4,
                         integrator="ode", ode_steps=4, ode_physics=True, ode_drift="transport")
        plain = LieGuidedForecaster(DEFAULT_SPEC, GuideConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), hidden=16, n_basis=4,
                                                              tend_head=True), cfg)
        plain(x, c, g)


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("tr")
    base, col = d / "synthetic.npz", d / "columns.npz"
    _write_data(base)
    _write_columns(col, base)
    return base, col


def test_stage17c_pipeline(files, tmp_path):
    base, col = files
    common = ["--fusion", "pigt", "--pigt-layers", "1", "--pigt-ops", "1", "--curvature-ops", "1", "--prev-state", "1",
              "--column-inputs", "fcn10", "--column-data", str(col)]
    roll = ("--rollout-steps", "2", "--rollout-start", "0")
    src = tmp_path / "src"  # stands in for stage9b
    _run(src, base, "b_noguide", *common, "--steps", "2", "--val-every", "2")
    _run(src, base, "a", "--steps", "2", "--val-every", "2")
    _run(src, base, "b", "--steps", "2", "--val-every", "2")
    run_dir = tmp_path / "run_s17c"
    run_dir.mkdir()
    for f in ("normalizer.pt", "column_norm.json"):
        (run_dir / f).write_bytes((src / f).read_bytes())
    flags = common + ["--flow-head", "1", "--integrator", "ode", "--ode-steps", "2", "--ode-physics", "1",
                      "--ode-drift-gain", "1", "--ode-drift", "transport", "--ode-path-forcing", "1", "--inter-times", "1"]
    out = _run(run_dir, base, "a", *flags, "--steps", "3", "--val-every", "2", "--init-guide", str(src / "guide.pt"),
               "--perturb-model", str(src / "vnet_noguide.pt"), "--guide-heads-only", "1")
    assert "guide initialised from" in out and "flow_out" in out and "Lie transport guide" in out
    assert "trunk frozen, training the heads only" in out
    import torch as _t
    old_g, new_g = _t.load(src / "guide.pt"), _t.load(run_dir / "guide.pt")
    heads = ("head_dec.", "rec_out.", "aux_out.", "tend_out.", "flow_out.")
    assert all(_t.equal(old_g[k], new_g[k]) for k in old_g if not k.startswith(heads))  # features P unchanged
    assert any(not _t.equal(old_g[k], new_g[k]) for k in old_g if k.startswith("tend_out."))  # source head retrained
    assert " transport only " in out
    rows = [json.loads(line) for line in open(run_dir / "log_a.jsonl")]
    assert any(r.get("val_transport_only") is not None for r in rows)
    cfg = json.load(open(run_dir / "config.json"))
    assert cfg["flow_head"] == 1 and cfg["ode_drift"] == "transport"
    out = _run(run_dir, base, "b", "--steps", "3", "--val-every", "2", *roll, "--init-vnet", str(src / "vnet.pt"))
    assert " gain " in out
    _run(run_dir, base, "b_noguide", "--steps", "2", "--val-every", "2", *roll, "--init-vnet", str(src / "vnet_noguide.pt"))
    _run(run_dir, base, "eval", "--eval-stride", "4", "--max-lead-steps", "2", "--n-ens-cases", "0")
    ev = json.load(open(run_dir / "eval.json"))
    for name in ("lie_guided", "no_guide", "physics_only"):
        assert all(np.isfinite(v) for v in ev["rmse"]["2"][name].values()), name
    r = subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "scripts/diagnose_guide.py"), "--run", str(run_dir),
                        "--threads", "1"], capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
