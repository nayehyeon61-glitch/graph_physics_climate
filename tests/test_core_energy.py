"""stage18h energy-based partition of the graph tendency core (power diagram re-drawn from the state's energy)."""

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
from gpclimate.models.dynamics_vnet import ENERGY_COMPONENTS, LieGuidedForecaster, VNetConfig  # noqa: E402
from gpclimate.nn.so2 import Field  # noqa: E402
from test_column_inputs import _write_columns  # noqa: E402
from test_core_laplace import _faces  # noqa: E402
from test_graph_core import _core_forecaster  # noqa: E402
from test_train_eval_smoke import _run, _write_data  # noqa: E402
from test_transport_guide import GUIDE, H, W, _graphs, _state  # noqa: E402


def test_batched_partition_matches_each_sample_and_fixed_areas_conserve_the_grid_integral():
    fg = _faces()
    N, F = fg.n_cells, len(fg.src)
    torch.manual_seed(0)
    a = torch.randn(2, N)
    gb = fg.deformed(a, 0.25)
    for k in range(2):  # a per-sample partition is the same power diagram as the shared one of that sample
        gk = fg.deformed(a[k], 0.25)
        for name in ("area", "length", "weight", "face_mass"):
            assert torch.allclose(getattr(gb, name)[k], getattr(gk, name), rtol=1e-5, atol=0), name
    gf = fg.deformed(a, 0.25, fixed_areas=True)
    assert gf.area is fg.area and gf.face_mass is fg.face_mass  # the grid control volumes stay
    assert torch.allclose(gf.length, gb.length) and torch.allclose(gf.weight, gb.weight)  # the faces move
    assert torch.allclose(gf.deformation["area_ratio"], gb.deformation["area_ratio"])  # diagnostic: new areas
    q = torch.randn(2, N, 3)
    K, dt = torch.full((2, F, 3), 1e9), 6 * 3600.0
    A = fg.area[None, :, None]
    for d in (gf.diffusion(q, K, dt), gf.divergence(torch.randn(2, F, 3))):  # conserved for the GRID areas
        assert float((d * A).sum(1).abs().max()) < 1e-5 * float((d.abs() * A).sum(1).max())
    q1 = q + gf.diffusion(q, K, dt)
    assert float(q1.max()) <= float(q.max()) + 1e-6 and float(q1.min()) >= float(q.min()) - 1e-6  # monotone
    assert float(gf.diffusion(torch.ones(2, N, 3), K, dt).abs().max()) < 1e-6  # a uniform field stays uniform
    d0 = fg.diffusion(q, torch.full((2, F, 3), 1e3), dt)
    d1 = gf.diffusion(q, torch.full((2, F, 3), 1e3), dt)
    assert not torch.allclose(d0, d1)  # the partition changes the exchange
    a2 = a.clone().requires_grad_(True)
    fg.deformed(a2, 0.25, fixed_areas=True).diffusion(q, torch.full((2, F, 3), 1e3), dt).pow(2).sum().backward()
    assert a2.grad is not None and float(a2.grad.abs().max()) > 0


def _lap_pair(extra: dict):
    """(Laplace-block core forecaster, the same + ``extra`` config keys with the shared weights loaded)."""
    plain = _core_forecaster(core=True)
    torch.manual_seed(0)
    with torch.no_grad():  # a non-zero face flux and a visible Laplace block: the partition has operators to move
        for p in plain.vnet.flux_head.psi[-1].parameters():
            p.normal_(0, 0.3)
    lap_cfg = VNetConfig(**{**plain.vnet.cfg.__dict__, "core_laplace": True})
    lap = LieGuidedForecaster(DEFAULT_SPEC, GUIDE, lap_cfg).eval()
    lap.vnet.load_state_dict(plain.vnet.state_dict(), strict=False)
    lap.guide.load_state_dict(plain.guide.state_dict())
    with torch.no_grad():
        lap.vnet.laplace_head.lin_s.bias.fill_(0.5)
    m = LieGuidedForecaster(DEFAULT_SPEC, GUIDE, VNetConfig(**{**lap_cfg.__dict__, **extra})).eval()
    missing, unexpected = m.vnet.load_state_dict(lap.vnet.state_dict(), strict=False)
    m.guide.load_state_dict(lap.guide.state_dict())
    return lap, m, missing, unexpected


def test_energy_partition_starts_at_the_parent_and_shrinks_cells_where_the_energy_is_high():
    g = _graphs().view()
    x, c = _state(2, wind=0.5)
    lap, m, missing, unexpected = _lap_pair({"core_energy_geom": True})
    assert not unexpected and missing and all(k.startswith("energy_") for k in missing), (missing, unexpected)
    m.vnet.set_energy_scales(torch.zeros(5), torch.ones(5), torch.full((3,), 10.0))
    with torch.no_grad():
        f_lap, _ = lap(x, c, g)
        f0, _ = m(x, c, g)
        assert torch.allclose(f0.s, f_lap.s, atol=1e-6) and torch.allclose(f0.v, f_lap.v, atol=1e-6)  # the parent
        assert m.last_core["energy_alpha"] == [0.0] * len(ENERGY_COMPONENTS) and m.last_core["geom_shift"] == 0
        m.vnet.energy_raw[ENERGY_COMPONENTS.index("ke500")] = 0.25  # alpha = 1: small cells where |u500| is large
        f1, _ = m(x, c, g)
        assert 0 < m.last_core["geom_shift"] <= 0.25 and m.last_core["geom_area"][0] < 1 < m.last_core["geom_area"][1]
        assert not torch.allclose(f1.s, f_lap.s, atol=1e-6)
        corr = dict(zip(ENERGY_COMPONENTS, m.last_core["energy_area_corr"]))
        assert corr["ke500"] < -0.3, corr
        # the energy components are gauge invariant (|u|^2, u . du) and non-negative
        e = m.vnet.core_energy(x, x.v * 0.1, H, W)
        rot = torch.exp(1j * torch.rand(x.v.shape[:2]))[..., None]
        e_rot = m.vnet.core_energy(Field(x.s, x.v * rot), x.v * 0.1 * rot, H, W)
        assert e.shape[-1] == len(ENERGY_COMPONENTS) and float(e.min()) >= 0 and torch.allclose(e, e_rot, atol=1e-5)
    f, _ = m(x, c, g)
    f.s.pow(2).mean().backward()
    assert m.vnet.energy_raw.grad is not None and float(m.vnet.energy_raw.grad.abs().max()) > 0
    with pytest.raises(ValueError):  # the partition belongs to the graph tendency core
        LieGuidedForecaster(DEFAULT_SPEC, GUIDE, VNetConfig(**{**m.vnet.cfg.__dict__, "tendency_core": False}))


def test_energy_partition_adds_to_the_static_geometry():
    g = _graphs().view()
    x, c = _state(2, wind=0.5)
    _, m, missing, _ = _lap_pair({"core_geometry": True, "core_energy_geom": True})
    assert all(k.startswith(("energy_", "geom_head")) for k in missing), missing
    m.vnet.set_energy_scales(torch.zeros(5), torch.ones(5), torch.full((3,), 10.0))
    with torch.no_grad():
        for p in m.vnet.geom_head[-1].parameters():
            p.normal_(0, 1.0)
        m(x, c, g)
        static = m.last_core["geom_shift"]
        m.vnet.energy_raw.fill_(0.1)
        m(x, c, g)
        assert static > 0 and m.last_core["geom_shift"] != static


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("energy")
    base, col = d / "synthetic.npz", d / "columns.npz"
    _write_data(base)
    _write_columns(col, base)
    return base, col


def test_stage18h_pipeline(files, tmp_path):
    base, col = files
    common = ["--fusion", "pigt", "--pigt-layers", "1", "--pigt-ops", "1", "--curvature-ops", "1", "--prev-state", "1",
              "--column-inputs", "fcn10", "--column-data", str(col)]
    roll = ("--rollout-steps", "2", "--rollout-start", "0")
    src = tmp_path / "src"
    _run(src, base, "b_noguide", *common, "--steps", "2", "--val-every", "2")
    _run(src, base, "a", "--steps", "2", "--val-every", "2")
    _run(src, base, "b", "--steps", "2", "--val-every", "2")
    run_dir = tmp_path / "run_s18h"
    run_dir.mkdir()
    for f in ("normalizer.pt", "column_norm.json"):
        (run_dir / f).write_bytes((src / f).read_bytes())
    flags = common + ["--flow-head", "1", "--integrator", "ode", "--ode-steps", "2", "--ode-drift", "transport",
                      "--ode-path-forcing", "1", "--inter-times", "1", "--tendency-core", "1", "--core-laplace", "1",
                      "--core-energy-geom", "1"]
    _run(run_dir, base, "a", *flags, "--steps", "2", "--val-every", "2", "--init-guide", str(src / "guide.pt"),
         "--guide-heads-only", "1")
    cfg = json.load(open(run_dir / "config.json"))
    assert cfg["core_energy_geom"] == 1
    _run(run_dir, base, "b_noguide", "--steps", "2", "--val-every", "2", *roll, "--init-vnet", str(src / "vnet_noguide.pt"))
    out = _run(run_dir, base, "b", "--steps", "2", "--val-every", "2", *roll, "--init-vnet", str(src / "vnet.pt"))
    assert "energy-based partition" in out and " energy a " in out and " geom |d| " in out
    sd = torch.load(run_dir / "vnet.pt")
    assert float(sd["energy_v_scale"].min()) > 0 and not torch.equal(sd["energy_s_std"], torch.ones(5))  # scales saved
    rows = [json.loads(line) for line in open(run_dir / "log_vnet.jsonl")]
    assert any({"energy_alpha", "energy_area_corr", "geom_shift"} <= set(r.get("core", {})) for r in rows)
    _run(run_dir, base, "eval", "--eval-stride", "4", "--max-lead-steps", "2", "--n-ens-cases", "0")
    ev = json.load(open(run_dir / "eval.json"))
    for name in ("lie_guided", "no_guide", "physics_only"):
        assert all(np.isfinite(v) for v in ev["rmse"]["2"][name].values()), name
