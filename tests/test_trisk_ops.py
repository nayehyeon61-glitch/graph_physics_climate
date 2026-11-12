"""stage22: TRiSK / MPAS C-grid quantities of the flow as guide inputs -- steady geostrophic balance (Williamson test
case 2: the vector-invariant tendency vanishes and converges with the mesh), kinetic energy, PV sign, mass-flux
divergence and edge-flux advection of analytic fields, gauge equivariance, the zero-initialised guide adapter and the
training pipeline."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from gpclimate.data.era5_wb2 import DEFAULT_SPEC  # noqa: E402
from gpclimate.geometry.graphs import SphereGraphs  # noqa: E402
from gpclimate.models.lie_physical_gnn import GuideConfig, LiePhysicalGNN  # noqa: E402
from gpclimate.nn.physics_ops import EARTH_RADIUS, OMEGA  # noqa: E402
from gpclimate.nn.so2 import Field  # noqa: E402
from gpclimate.nn.trisk_ops import G0, H_REF, U_REF, TriskOperators, edge_faces  # noqa: E402
from gpclimate.training.losses import random_gauges, relative_error  # noqa: E402
from test_column_inputs import _write_columns  # noqa: E402
from test_train_eval_smoke import _run, _write_data  # noqa: E402

LAT64, LON64 = np.linspace(-87.1875, 87.1875, 32), np.arange(64) * 5.625
U0, GH0 = 2 * np.pi * EARTH_RADIUS / (12 * 86400), 2.94e4  # Williamson et al. (1992) test case 2


@pytest.fixture(scope="module")
def graphs():
    return SphereGraphs(LAT64, LON64, mesh_splits=(4, 3, 2))


def _rel_rms(a, b):
    return float(torch.sqrt(torch.mean((a - b) ** 2) / torch.mean(b ** 2)))


def _tc2(nodes):
    """u = u0 cos(lat) eastward, Phi = g h0 - (a Omega u0 + u0^2 / 2) sin^2(lat): a steady state of the shallow-water
    equations; in the (east, north) node gauge the eastward wind is the real axis."""
    lat = np.deg2rad(nodes.lat)
    u = torch.as_tensor(U0 * np.cos(lat)).to(torch.complex64)
    phi = torch.as_tensor(GH0 - (EARTH_RADIUS * OMEGA * U0 + U0 ** 2 / 2) * np.sin(lat) ** 2, dtype=torch.float32)
    return u, phi


def _balance(gr):
    nodes, cx, ops = gr.mesh[0], gr.complexes[0], TriskOperators(DEFAULT_SPEC)
    u, phi = _tc2(nodes)
    s = torch.zeros(1, nodes.n, DEFAULT_SPEC.n_s)
    s[..., 0] = phi  # z500 = geopotential, unit scales
    v = torch.zeros(1, nodes.n, DEFAULT_SPEC.n_v, dtype=torch.complex64)
    v[..., 1] = u  # the 500 hPa pair
    v[..., 0] = u
    out = ops(s, v, gr.mesh_sinlat[0], cx)
    vn = ops.vector_names
    pvf, gb, vi = (out.v[..., vn.index(n)] for n in ("pvflux500", "gradB500", "vi_tend500"))
    rms = lambda z: float(z.abs().pow(2).mean().sqrt())  # noqa: E731
    return rms(vi) / rms(gb), rms(pvf) / rms(gb), out, ops


def test_tc2_steady_state_balance_converges_with_the_mesh(graphs):
    r3, _, _, _ = _balance(SphereGraphs(LAT64, LON64, mesh_splits=(3, 2, 1)))
    r4, pv_over_gb, out, ops = _balance(graphs)
    assert 0.8 < pv_over_gb < 1.2  # the two terms of the vector-invariant equation balance ...
    assert r4 < 0.1 and r4 < 0.7 * r3, (r3, r4)  # ... and their residual vanishes with resolution
    nodes = graphs.mesh[0]
    lat = torch.as_tensor(np.deg2rad(nodes.lat), dtype=torch.float32)
    ke = out.s[..., ops.scalar_names.index("ke1")][0] * U_REF ** 2
    assert _rel_rms(ke, 0.5 * U0 ** 2 * torch.cos(lat) ** 2) < 0.05  # TRiSK kinetic energy = u^2 / 2
    pv = out.s[..., ops.scalar_names.index("pv500")][0] * (2 * OMEGA / H_REF)
    _, phi = _tc2(nodes)
    eta = 2 * OMEGA * torch.sin(lat) + 2 * U0 * torch.sin(lat) / EARTH_RADIUS  # absolute vorticity of u0 cos(lat)
    assert _rel_rms(pv, eta / (phi / G0)) < 0.08  # q = eta / h
    assert ops.out_dims == (out.s.shape[-1], out.v.shape[-1]) == (7, 5)
    assert torch.isfinite(out.s).all() and torch.isfinite(out.v).all()
    ef = edge_faces(graphs.complexes[0])
    assert ef.shape == (graphs.complexes[0].n_e, 2) and bool((ef[:, 0] != ef[:, 1]).all())


def test_mass_flux_divergence_and_edge_flux_advection_of_analytic_fields(graphs):
    """u = cos(lat) north (= grad sin(lat), 1 m/s): div(h u) = -2 h sin(lat) / a for uniform h; the tracer sin(lat)
    advected by it has -div(u T) = (2 sin^2 - cos^2)(lat) / a."""
    nodes, cx, ops = graphs.mesh[0], graphs.complexes[0], TriskOperators(DEFAULT_SPEC)
    lat = torch.as_tensor(np.deg2rad(nodes.lat), dtype=torch.float32)
    u = (1j * torch.cos(lat)).to(torch.complex64)
    s = torch.zeros(1, nodes.n, DEFAULT_SPEC.n_s)
    s[..., 0] = H_REF * G0  # uniform depth H_REF
    s[..., 1] = torch.sin(lat)  # t850 (normalised units) = sin(lat)
    v = torch.zeros(1, nodes.n, DEFAULT_SPEC.n_v, dtype=torch.complex64)
    v[..., 0] = u
    v[..., 1] = u
    out = ops(s, v, graphs.mesh_sinlat[0], cx)
    div = out.s[..., ops.scalar_names.index("div_hu500")][0] * (H_REF * 1e-6)
    assert _rel_rms(div, -2 * H_REF * torch.sin(lat) / EARTH_RADIUS) < 0.08
    adv = out.s[..., ops.scalar_names.index("adv_t850")][0] * 1e-5
    assert _rel_rms(adv, (2 * torch.sin(lat) ** 2 - torch.cos(lat) ** 2) / EARTH_RADIUS) < 0.15
    assert float(out.s[..., ops.scalar_names.index("adv_q700")].abs().max()) < 1e-6  # q700 = 0 everywhere


def test_trisk_operators_are_gauge_equivariant(graphs):
    nodes, ops = graphs.mesh[0], TriskOperators(DEFAULT_SPEC)
    ops.set_scales(torch.tensor([5.4e4, 280.0, 3e-3, 285.0, 1e5]), torch.tensor([3e3, 15.0, 2e-3, 20.0, 1e3]),
                   torch.tensor([10.0, 15.0, 5.0]))
    s = torch.randn(2, nodes.n, DEFAULT_SPEC.n_s)
    v = torch.complex(torch.randn(2, nodes.n, 3), torch.randn(2, nodes.n, 3))
    gen = torch.Generator().manual_seed(1)
    _, tm = random_gauges(graphs, gen)
    g0, g1 = graphs.view(), graphs.gauge_transformed(tm)
    out0 = ops(s, v, g0.mesh_sinlat[0], g0.extras["complexes"][0])
    out1 = ops(s, Field(s[..., :0], v).gauge(tm[0]).v, g1.mesh_sinlat[0], g1.extras["complexes"][0])
    assert relative_error(Field(out1.s, out1.v[..., :0]), Field(out0.s, out0.v[..., :0])) < 1e-4
    assert relative_error(out1, Field(out0.s, out0.v).gauge(tm[0])) < 1e-4


def test_guide_with_trisk_ops_starts_unchanged_and_is_equivariant():
    gr = SphereGraphs(np.linspace(-87.1875, 87.1875, 16), np.arange(0, 360, 11.25), mesh_splits=(2, 1, 0))
    cfg = GuideConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), hidden=16, n_basis=4, physics_ops=True, tend_head=True,
                      curvature_ops=True)
    torch.manual_seed(0)
    m0 = LiePhysicalGNN(DEFAULT_SPEC, cfg).eval()
    m1 = LiePhysicalGNN(DEFAULT_SPEC, replace(cfg, trisk_ops=True)).eval()
    missing, unexpected = m1.load_state_dict(m0.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith(("trisk_ops.", "trisk_norm.", "trisk_embed.")) for k in missing), missing
    G = gr.grid.n
    x = Field(torch.randn(2, G, DEFAULT_SPEC.n_s), torch.complex(torch.randn(2, G, 3), torch.randn(2, G, 3)))
    c = Field(torch.randn(1, G, DEFAULT_SPEC.n_cond_s),
              torch.complex(torch.randn(1, G, DEFAULT_SPEC.n_cond_v), torch.randn(1, G, DEFAULT_SPEC.n_cond_v)))
    with torch.no_grad():
        f0, f1 = m0(x, c, torch.zeros(2), gr.view()), m1(x, c, torch.zeros(2), gr.view())
        assert relative_error(f1["P0u"], f0["P0u"]) < 1e-6  # the zero-initialised adapter leaves the features P
        raw = m1.trisk_terms(x, gr.view())
        assert raw.s.shape == (2, gr.mesh[0].n, 7) and raw.v.shape == (2, gr.mesh[0].n, 5)
        assert torch.isfinite(raw.s).all() and torch.isfinite(raw.v).all()
        for p in m1.trisk_embed.parameters():
            p.normal_(0, 0.3)
        f2 = m1(x, c, torch.zeros(2), gr.view())
        assert relative_error(f2["P0u"], f0["P0u"]) > 1e-3
        gen = torch.Generator().manual_seed(2)
        tg, tm = random_gauges(gr, gen)
        f3 = m1(x.gauge(tg), c.gauge(tg), torch.zeros(2), gr.gauge_transformed(tm, tg))
    for k, lvl in (("P0", 0), ("P1", 1), ("P2", 2), ("P1u", 1), ("P0u", 0)):
        assert relative_error(f3[k], f2[k].gauge(tm[lvl])) < 5e-5, k


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("trisk")
    base, col = d / "synthetic.npz", d / "columns.npz"
    _write_data(base)
    _write_columns(col, base)
    return base, col


def test_stage22_pipeline(files, tmp_path):
    base, col = files
    common = ["--fusion", "pigt", "--pigt-layers", "1", "--pigt-ops", "1", "--curvature-ops", "1", "--prev-state", "1",
              "--column-inputs", "fcn10", "--column-data", str(col)]
    src = tmp_path / "src"
    _run(src, base, "b_noguide", *common, "--steps", "2", "--val-every", "2")
    _run(src, base, "a", "--steps", "2", "--val-every", "2")
    run_dir = tmp_path / "run_s22"
    run_dir.mkdir()
    for f in ("normalizer.pt", "column_norm.json"):
        (run_dir / f).write_bytes((src / f).read_bytes())
    out_a = _run(run_dir, base, "a", *common, "--trisk-ops", "1", "--steps", "3", "--val-every", "2",
                 "--init-guide", str(src / "guide.pt"), "--guide-heads-only", "1")
    assert "TRiSK C-grid inputs (7, 5)" in out_a and "trisk_embed" in out_a and "trunk frozen" in out_a
    assert json.load(open(run_dir / "config.json"))["trisk_ops"] == 1
    _run(run_dir, base, "b_noguide", "--steps", "2", "--val-every", "2", "--init-vnet", str(src / "vnet_noguide.pt"))
    _run(run_dir, base, "b", "--steps", "2", "--val-every", "2")
    _run(run_dir, base, "eval", "--eval-stride", "4", "--max-lead-steps", "2", "--n-ens-cases", "0")
    ev = json.load(open(run_dir / "eval.json"))
    for name in ("lie_guided", "no_guide"):
        assert all(np.isfinite(v) for v in ev["rmse"]["2"][name].values()), name
