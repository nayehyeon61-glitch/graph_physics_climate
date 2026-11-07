"""stage19 GPINN: physics residuals, terrain-weighted graph embedding, guide input, pipeline."""

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
from gpclimate.geometry.graph_embedding import terrain_laplacian_eigvecs  # noqa: E402
from gpclimate.models.lie_physical_gnn import GuideConfig, LiePhysicalGNN  # noqa: E402
from gpclimate.nn.physics_ops import EARTH_RADIUS, OMEGA  # noqa: E402
from gpclimate.nn.so2 import Field  # noqa: E402
from gpclimate.training.pinn import ALL_TERMS, PINNResiduals  # noqa: E402
from test_column_inputs import _write_columns  # noqa: E402
from test_train_eval_smoke import _run, _write_data  # noqa: E402
from test_transport_guide import GUIDE, H, LAT, LON, W, _graphs  # noqa: E402

LATR, LONR = np.deg2rad(LAT)[:, None], np.deg2rad(LON)[None, :]


def _balanced_state(B=2, noise=0.0, seed=0):
    """z500 = 5.5e4 + A cos^2(lat) sin(2 lon) (m^2/s^2) with the exactly geostrophic 500 hPa wind; physical units."""
    A = 4000.0
    phi = 5.5e4 + A * np.cos(LATR) ** 2 * np.sin(2 * LONR)
    dphi_dx = A * np.cos(LATR) ** 2 * 2 * np.cos(2 * LONR) / (EARTH_RADIUS * np.cos(LATR))
    dphi_dy = A * 2 * np.cos(LATR) * (-np.sin(LATR)) * np.sin(2 * LONR) / EARTH_RADIUS
    f = 2 * OMEGA * np.sin(LATR)
    f = np.where(np.abs(f) < 1e-5, 1e-5, f)
    u, v = -dphi_dy / f, dphi_dx / f  # V_g = (1/f) k x grad(phi)
    rng = np.random.default_rng(seed)
    s = np.stack([phi, np.full_like(phi, 270.0), np.full_like(phi, 3e-3), np.full_like(phi, 285.0),
                  np.full_like(phi, 1.01e5)], -1).reshape(-1, 5)
    w500 = (u + 1j * v).reshape(-1) + noise * (rng.standard_normal(H * W) + 1j * rng.standard_normal(H * W))
    vec = np.stack([w500 * 0.5, w500, w500 * 0.3], -1)
    S = torch.as_tensor(s, dtype=torch.float32)[None].expand(B, -1, -1).clone()
    V = torch.as_tensor(vec, dtype=torch.complex64)[None].expand(B, -1, -1).clone()
    return Field(S, V)


def _pinn(g, terms=ALL_TERMS):
    gr = _graphs()
    return PINNResiduals(torch.zeros(5), torch.ones(5), torch.ones(3), gr.mesh[0].lat, terms), g


def test_geostrophic_state_has_small_residual_and_losses_are_differentiable():
    g = _graphs().view()
    pinn, _ = _pinn(g, ("geo500",))
    good, bad = _balanced_state(), _balanced_state(noise=15.0)
    r_good, r_bad = pinn.ratios(good, None, g)["geo500"], pinn.ratios(bad, None, g)["geo500"]
    assert r_good < 0.15 and r_bad > 3 * r_good, (r_good, r_bad)
    pinn.fit_scales([(good, good)], g)
    x = Field(bad.s.clone().requires_grad_(True), bad.v.clone().requires_grad_(True))
    loss = sum(pinn.losses(x, None, g).values())
    loss.backward()
    assert torch.isfinite(loss) and float(x.v.grad.abs().sum()) > 0


def test_change_relations_and_scales():
    g = _graphs().view()
    pinn, _ = _pinn(g)
    x0 = _balanced_state()
    x1 = Field(x0.s * 1.001, x0.v * 1.01)
    rel = pinn.relations(x0, x1, g)
    assert set(rel) == set(ALL_TERMS)
    sc = pinn.fit_scales([(x0, x1)], g)
    assert all(v > 0 for v in sc.values())
    losses = pinn.losses(x0, x1, g)
    assert all(torch.isfinite(v) for v in losses.values())
    with pytest.raises(ValueError):
        PINNResiduals(torch.zeros(5), torch.ones(5), torch.ones(3), _graphs().mesh[0].lat, ("momentum",))


def test_terrain_embedding_reduces_to_l1_harmonics_on_a_flat_ocean_planet():
    flat = terrain_laplacian_eigvecs(LAT, LON, np.zeros((H, W)), np.zeros((H, W)), k=3)
    harm = np.stack([(np.cos(LATR) * np.cos(LONR)).ravel(), (np.cos(LATR) * np.sin(LONR)).ravel(),
                     (np.sin(LATR) * np.ones_like(LONR)).ravel()], 1)
    q, _ = np.linalg.qr(harm)
    v = flat.numpy() / np.linalg.norm(flat.numpy(), axis=0)
    assert np.linalg.norm(q.T @ v, axis=0).min() > 0.95  # the first three modes span the l = 1 harmonics
    rng = np.random.default_rng(0)
    orog = np.zeros((H, W))
    orog[:, W // 4:W // 4 + 2] = 5000.0  # a meridional mountain wall
    lsm = (rng.random((H, W)) > 0.7).astype(float)
    emb = terrain_laplacian_eigvecs(LAT, LON, orog, lsm, k=4).numpy()
    assert emb.shape == (H * W, 4) and np.isfinite(emb).all()
    assert np.linalg.norm(q.T @ (emb / np.linalg.norm(emb, axis=0)), axis=0).min() < 0.9  # terrain changes the modes


def test_graph_embedding_input_starts_at_zero():
    g = _graphs().view()
    torch.manual_seed(0)
    plain = LiePhysicalGNN(DEFAULT_SPEC, GUIDE).eval()
    cfg = GuideConfig(**{**GUIDE.__dict__, "graph_embed": 4})
    emb_g = LiePhysicalGNN(DEFAULT_SPEC, cfg).eval()
    missing, unexpected = emb_g.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and all(k.startswith(("gemb", "gemb_in")) for k in missing)
    x = Field(torch.randn(2, H * W, 5), torch.randn(2, H * W, 3, dtype=torch.complex64))
    c = Field(torch.randn(2, H * W, 5), torch.randn(2, H * W, 1, dtype=torch.complex64))
    with pytest.raises(ValueError):
        emb_g(x, c, torch.zeros(2), g)  # embedding not set
    emb_g.set_graph_embedding(terrain_laplacian_eigvecs(LAT, LON, np.zeros((H, W)), np.zeros((H, W)), k=4))
    with torch.no_grad():
        a, b = plain(x, c, torch.zeros(2), g), emb_g(x, c, torch.zeros(2), g)
    assert torch.allclose(a["P0u"].s, b["P0u"].s, atol=1e-6)
    assert "gemb" in emb_g.state_dict()  # saved with the guide


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("gpinn")
    base, col = d / "synthetic.npz", d / "columns.npz"
    _write_data(base)
    _write_columns(col, base)
    return base, col


def test_stage19_pipeline(files, tmp_path):
    base, col = files
    common = ["--fusion", "pigt", "--pigt-layers", "1", "--pigt-ops", "1", "--curvature-ops", "1", "--prev-state", "1",
              "--column-inputs", "fcn10", "--column-data", str(col)]
    roll = ("--rollout-steps", "2", "--rollout-start", "0")
    src = tmp_path / "src"
    _run(src, base, "b_noguide", *common, "--steps", "2", "--val-every", "2")
    _run(src, base, "a", "--steps", "2", "--val-every", "2")
    _run(src, base, "b", "--steps", "2", "--val-every", "2")
    run_dir = tmp_path / "run_s19"
    run_dir.mkdir()
    for f in ("normalizer.pt", "column_norm.json"):
        (run_dir / f).write_bytes((src / f).read_bytes())
    flags = common + ["--flow-head", "1", "--integrator", "ode", "--ode-steps", "2", "--ode-drift", "transport",
                      "--ode-path-forcing", "1", "--inter-times", "1", "--tendency-core", "1", "--graph-embed", "4"]
    out = _run(run_dir, base, "a", *flags, "--steps", "3", "--val-every", "2", "--init-guide", str(src / "guide.pt"),
               "--pinn-weight", "0.05")
    assert "GPINN residuals" in out and "PINN guide/ERA5" in out and "graph embedding" in out
    assert (run_dir / "pinn_scales.json").exists() and np.load(run_dir / "graph_embed.npy").shape[1] == 4
    rows = [json.loads(line) for line in open(run_dir / "log_a.jsonl")]
    assert any("val_pinn" in r for r in rows) and any(r.get("l_pinn", 0) > 0 for r in rows)
    out = _run(run_dir, base, "b", "--steps", "3", "--val-every", "2", *roll, "--init-vnet", str(src / "vnet.pt"),
               "--pinn-weight-b", "0.05")
    assert " L_pinn " in out
    _run(run_dir, base, "b_noguide", "--steps", "2", "--val-every", "2", *roll, "--init-vnet", str(src / "vnet_noguide.pt"))
    _run(run_dir, base, "eval", "--eval-stride", "4", "--max-lead-steps", "2", "--n-ens-cases", "0")
    ev = json.load(open(run_dir / "eval.json"))
    for name in ("lie_guided", "no_guide", "physics_only"):
        assert all(np.isfinite(v) for v in ev["rmse"]["2"][name].values()), name
