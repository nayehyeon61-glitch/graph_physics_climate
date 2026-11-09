"""stage18f edge conductance (theta_p) and stage18g learned dual geometry (theta_g) of the graph tendency core."""

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
from test_core_laplace import _faces  # noqa: E402
from test_graph_core import _core_forecaster  # noqa: E402
from test_train_eval_smoke import _run, _write_data  # noqa: E402
from test_transport_guide import GUIDE, H, W, _graphs, _state  # noqa: E402


def test_deformed_geometry_is_the_grid_at_zero_and_a_consistent_power_diagram_otherwise():
    fg = _faces()
    N, F = fg.n_cells, len(fg.src)
    g0 = fg.deformed(torch.zeros(N))
    for name in ("area", "length", "weight", "face_mass"):
        assert torch.allclose(getattr(g0, name), getattr(fg, name), rtol=1e-6, atol=0), name
    torch.manual_seed(0)
    a = torch.randn(N)
    g1 = fg.deformed(a, 0.25)
    assert abs(float(g1.area.sum() / fg.area.sum()) - 1) < 1e-5  # total area kept
    assert float(g1.area.min()) > 0 and float(g1.length.min()) > 0 and float(g1.face_mass.min()) > 0
    assert float(g1.deformation["shift"].abs().max()) <= 0.25 + 1e-6
    assert not torch.allclose(g1.area, fg.area) and not torch.allclose(g1.weight, fg.weight)
    b = torch.zeros(N)  # a heavier cell grows, its east neighbour shrinks
    i = (H // 2) * W + W // 2
    b[i] = 3.0
    g2 = fg.deformed(b, 0.25)
    assert float(g2.area[i]) > 1.5 * float(fg.area[i]) and float(g2.area[i + 1]) < float(fg.area[i + 1])
    q = torch.randn(2, N, 3)
    K, dt = torch.full((2, F, 3), 1e9), 6 * 3600.0
    A1 = g1.area[None, :, None]
    for d in (g1.diffusion(q, K, dt), g1.divergence(torch.randn(2, F, 3))):  # conserved w.r.t. the new areas
        assert float((d * A1).sum(1).abs().max()) < 1e-5 * float((d.abs() * A1).sum(1).max())
    q1 = q + g1.diffusion(q, K, dt)
    assert float(q1.max()) <= float(q.max()) + 1e-6 and float(q1.min()) >= float(q.min()) - 1e-6  # monotone
    a2 = a.clone().requires_grad_(True)  # the geometry is learnable
    fg.deformed(a2).diffusion(q, torch.full((2, F, 3), 1e3), dt).pow(2).sum().backward()
    assert a2.grad is not None and float(a2.grad.abs().max()) > 0


def test_edge_conductance_is_one_at_init_symmetric_and_learnable():
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
    m = LieGuidedForecaster(DEFAULT_SPEC, GUIDE, VNetConfig(**{**lap_cfg.__dict__, "core_edge": True})).eval()
    missing, unexpected = m.vnet.load_state_dict(lap.vnet.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith("edge_head") for k in missing), (missing, unexpected)
    m.guide.load_state_dict(lap.guide.state_dict())
    n_st = len(DEFAULT_SPEC.static_scalars)
    with torch.no_grad():
        f_lap, _ = lap(x, c, g)
        f0, _ = m(x, c, g)
        assert torch.allclose(f0.s, f_lap.s, atol=1e-6)  # multiplier 1 at the start
        assert all(abs(v - 1) < 1e-6 for v in m.last_core["edge_mult"]) and m.last_core["edge_sd"] == 0
        for p in m.vnet.edge_head.psi[-1].parameters():
            p.normal_(0, 0.3)
        f1, _ = m(x, c, g)
        assert not torch.allclose(f1.s, f_lap.s) and m.last_core["edge_sd"] > 0.01
        emb, faces, gauge = m.vnet.last_emb, m.faces(g), g.extras["grid_gauge"]
        flow, _, _ = m.core_terms(x, m.physics(x, c, g), g)
        phys = m.vnet.edge_head.phys_features(faces, x.s, flow, c.s[..., :n_st], gauge)
        assert phys.shape == (2, len(faces.src), m.vnet.edge_head.n_phys) and float(phys.min()) >= 0
        phi = m.vnet.edge_head(emb, faces, gauge, phys)
        assert float(phi.abs().max()) <= 4.0
        rev = faces.deformed(torch.zeros(faces.n_cells))  # the same graph with every face reversed (i <-> j, n -> -n)
        rev.src, rev.dst, rev.normal = faces.dst, faces.src, -faces.normal
        phys_r = m.vnet.edge_head.phys_features(rev, x.s, flow, c.s[..., :n_st], gauge)
        assert torch.allclose(phys_r, phys, atol=1e-6)
        assert torch.allclose(m.vnet.edge_head(emb, rev, gauge, phys_r), phi, atol=1e-5)  # kappa_ij = kappa_ji
    f, _ = m(x, c, g)
    f.s.pow(2).mean().backward()
    assert any(p.grad is not None and float(p.grad.abs().max()) > 0 for p in m.vnet.edge_head.parameters())
    with pytest.raises(ValueError):  # the conductance multiplies the Laplace block
        LieGuidedForecaster(DEFAULT_SPEC, GUIDE, VNetConfig(**{**plain.vnet.cfg.__dict__, "core_edge": True}))


def test_geometry_head_starts_at_the_grid_and_moves_the_faces():
    g = _graphs().view()
    x, c = _state(2, wind=0.5)
    plain = _core_forecaster(core=True)
    torch.manual_seed(0)
    with torch.no_grad():  # a non-zero face flux and a visible Laplace block: the geometry has operators to move
        for p in plain.vnet.flux_head.psi[-1].parameters():
            p.normal_(0, 0.3)
    lap_cfg = VNetConfig(**{**plain.vnet.cfg.__dict__, "core_laplace": True})
    lap = LieGuidedForecaster(DEFAULT_SPEC, GUIDE, lap_cfg).eval()
    lap.vnet.load_state_dict(plain.vnet.state_dict(), strict=False)
    lap.guide.load_state_dict(plain.guide.state_dict())
    with torch.no_grad():
        lap.vnet.laplace_head.lin_s.bias.fill_(0.5)
    m = LieGuidedForecaster(DEFAULT_SPEC, GUIDE, VNetConfig(**{**lap_cfg.__dict__, "core_geometry": True})).eval()
    missing, unexpected = m.vnet.load_state_dict(lap.vnet.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith("geom_head") for k in missing), (missing, unexpected)
    m.guide.load_state_dict(lap.guide.state_dict())
    with torch.no_grad():
        f_lap, _ = lap(x, c, g)
        f0, _ = m(x, c, g)
        assert torch.allclose(f0.s, f_lap.s, atol=1e-6) and torch.allclose(f0.v, f_lap.v, atol=1e-6)  # the grid
        assert m.last_core["geom_shift"] == 0 and m.last_core["geom_area"] == [1.0, 1.0]
        assert m.last_core["flux"] > 0 and m.last_core["diffusion"] > 0
        for p in m.vnet.geom_head[-1].parameters():
            p.normal_(0, 1.0)
        f1, _ = m(x, c, g)
        assert 0 < m.last_core["geom_shift"] <= 0.25 and m.last_core["geom_area"][0] < 1 < m.last_core["geom_area"][1]
        assert not torch.allclose(f1.s, f_lap.s, atol=1e-6)
    f, _ = m(x, c, g)
    f.s.pow(2).mean().backward()
    assert any(p.grad is not None and float(p.grad.abs().max()) > 0 for p in m.vnet.geom_head.parameters())
    with pytest.raises(ValueError):  # the geometry belongs to the graph tendency core
        LieGuidedForecaster(DEFAULT_SPEC, GUIDE, VNetConfig(**{**plain.vnet.cfg.__dict__, "tendency_core": False,
                                                            "core_geometry": True}))


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("dual")
    base, col = d / "synthetic.npz", d / "columns.npz"
    _write_data(base)
    _write_columns(col, base)
    return base, col


def test_stage18f_18g_pipeline(files, tmp_path):
    base, col = files
    common = ["--fusion", "pigt", "--pigt-layers", "1", "--pigt-ops", "1", "--curvature-ops", "1", "--prev-state", "1",
              "--column-inputs", "fcn10", "--column-data", str(col)]
    roll = ("--rollout-steps", "2", "--rollout-start", "0")
    src = tmp_path / "src"
    _run(src, base, "b_noguide", *common, "--steps", "2", "--val-every", "2")
    _run(src, base, "a", "--steps", "2", "--val-every", "2")
    _run(src, base, "b", "--steps", "2", "--val-every", "2")
    run_dir = tmp_path / "run_s18fg"
    run_dir.mkdir()
    for f in ("normalizer.pt", "column_norm.json"):
        (run_dir / f).write_bytes((src / f).read_bytes())
    flags = common + ["--flow-head", "1", "--integrator", "ode", "--ode-steps", "2", "--ode-drift", "transport",
                      "--ode-path-forcing", "1", "--inter-times", "1", "--tendency-core", "1", "--core-laplace", "1",
                      "--core-tv", "1", "--core-edge", "1", "--core-geometry", "1"]
    _run(run_dir, base, "a", *flags, "--steps", "2", "--val-every", "2", "--init-guide", str(src / "guide.pt"),
         "--guide-heads-only", "1")
    cfg = json.load(open(run_dir / "config.json"))
    assert cfg["core_edge"] == 1 and cfg["core_geometry"] == 1
    _run(run_dir, base, "b_noguide", "--steps", "2", "--val-every", "2", *roll, "--init-vnet", str(src / "vnet_noguide.pt"))
    out = _run(run_dir, base, "b", "--steps", "2", "--val-every", "2", *roll, "--init-vnet", str(src / "vnet.pt"))
    assert "edge conductance" in out and "learned dual geometry" in out and " edge x " in out and " geom |d| " in out
    rows = [json.loads(line) for line in open(run_dir / "log_vnet.jsonl")]
    assert any({"edge_mult", "geom_shift", "tv_gain", "diffusion"} <= set(r.get("core", {})) for r in rows)
    _run(run_dir, base, "eval", "--eval-stride", "4", "--max-lead-steps", "2", "--n-ens-cases", "0")
    ev = json.load(open(run_dir / "eval.json"))
    for name in ("lie_guided", "no_guide", "physics_only"):
        assert all(np.isfinite(v) for v in ev["rmse"]["2"][name].values()), name
