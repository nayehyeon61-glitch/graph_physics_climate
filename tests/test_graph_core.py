"""stage18 graph tendency core: face graph, conservative antisymmetric flux, core sub-step, pipeline."""

from __future__ import annotations

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
from gpclimate.models.dynamics_vnet import LieGuidedForecaster, VNetConfig  # noqa: E402
from gpclimate.nn.graph_core import FaceFluxHead, GridFaceGraph  # noqa: E402
from gpclimate.nn.physics_ops import EARTH_RADIUS  # noqa: E402
from gpclimate.nn.so2 import Field  # noqa: E402
from test_column_inputs import _write_columns  # noqa: E402
from test_train_eval_smoke import _run, _write_data  # noqa: E402
from test_transport_guide import GUIDE, H, LAT, LON, W, _graphs, _state  # noqa: E402


def _faces():
    return GridFaceGraph(torch.as_tensor(LAT), torch.as_tensor(LON))


def test_face_graph_geometry_and_exact_conservation():
    fg = _faces()
    assert fg.n_cells == H * W
    assert abs(float(fg.area.double().sum()) / (4 * math.pi * EARTH_RADIUS ** 2) - 1) < 1e-5  # the whole sphere
    torch.manual_seed(0)
    J = torch.randn(2, len(fg.src), 3)
    div = fg.divergence(J)
    total = (div * fg.area[None, :, None]).sum(1)
    assert float(total.abs().max()) < 1e-6 * float((div.abs() * fg.area[None, :, None]).sum(1).max())
    # a zonally uniform eastward flux has no divergence; a uniform northward one moves mass between rows only
    east = torch.zeros(1, len(fg.src), 1)
    east[:, :H * W] = 1.0
    assert float(fg.divergence(east).abs().max()) < 1e-9
    north = torch.zeros(1, len(fg.src), 1)
    north[:, H * W:] = 1.0
    d = fg.divergence(north).view(H, W)
    assert float(d[0].mean()) > 0 and float(d[-1].mean()) < 0  # out of the southern cap, into the northern one


def test_face_flux_is_antisymmetric_and_gauge_invariant():
    fg = _faces()
    head = FaceFluxHead(6, 3, 4, hidden=16)
    torch.manual_seed(1)
    with torch.no_grad():
        head.psi[-1].weight.normal_(0, 0.5)
    emb = Field(torch.randn(2, H * W, 6), torch.randn(2, H * W, 3, dtype=torch.complex64))
    gauge = torch.zeros(H * W)
    J = head(emb, fg, gauge)
    # the same faces listed the other way round (dst -> src, normal -n) give -J
    rev = _faces()
    rev.src, rev.dst, rev.normal = fg.dst.clone(), fg.src.clone(), -fg.normal
    assert torch.allclose(head(emb, rev, gauge), -J, atol=1e-5)
    # rotating every cell's frame (vectors and gauge together) leaves the flux unchanged
    a = torch.rand(H * W) * 2 * math.pi
    emb_r = Field(emb.s, emb.v * torch.exp(-1j * a)[None, :, None])
    assert torch.allclose(head(emb_r, fg, a), J, atol=1e-4)


def _core_forecaster(core=True, tokens=True):
    torch.manual_seed(0)
    cfg = VNetConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), heads=2, ds=4, dv=2, hidden=16, n_basis=4, fusion="pigt",
                     pigt_layers=1, integrator="ode", ode_steps=4, ode_physics=False, ode_drift="transport",
                     ode_path_forcing=True, tendency_core=core, core_tokens=tokens)
    m = LieGuidedForecaster(DEFAULT_SPEC, GUIDE, cfg)
    m.guide.flow_v_scale.fill_(10.0)
    with torch.no_grad():
        m.vnet.head_out.lin_s.weight.normal_(0, 0.1)
    return m.eval()


def test_core_starts_at_the_plain_ode_dynamics_and_its_terms_act():
    g = _graphs().view()
    x, c = _state(2, wind=0.5)
    plain = _core_forecaster(core=False)
    m = _core_forecaster(core=True)
    missing, unexpected = m.vnet.load_state_dict(plain.vnet.state_dict(), strict=False)
    assert not unexpected and missing and all(("_x" in k) or k.startswith(("core_embed", "beta_head", "flux_head"))
                                              or ".norm_c." in k for k in missing), missing
    m.guide.load_state_dict(plain.guide.state_dict())
    with torch.no_grad():
        f_plain, _ = plain(x, c, g)
        f_core, _ = m(x, c, g)
        assert m.last_core is not None and m.last_core["beta"] == [0.0, 0.0, 0.0] and m.last_core["flux"] == 0.0
        assert m.last_core["guide_transport"] > 1e-3  # the guide's flow moves the fields (V-Net input)
        # zero-initialised core parts: only the low-logit C tokens perturb the trained V-Net, and only slightly
        rel = float((f_core.s - f_plain.s).abs().mean() / (f_plain.s - x.s).abs().mean())
        assert rel < 0.05, rel
        # beta = 1 everywhere: the guide's flow carries the state
        m.vnet.beta_head.lin_s.bias.fill_(1.0)
        f_beta, _ = m(x, c, g)
        assert m.last_core["transport"] > 1e-3 and not torch.allclose(f_beta.s, f_core.s)
        # a non-zero face flux changes cells but conserves the area integral of every scalar channel
        m.vnet.beta_head.lin_s.bias.fill_(0.0)
        m.vnet.flux_head.psi[-1].weight.normal_(0, 0.5)
        emb = m.vnet.last_emb
        C = m.vnet.core_flux(emb, m.faces(g), g.extras["grid_gauge"])
        A = m.faces(g).area
        assert float(C.abs().mean()) > 0
        assert float((C * A[None, :, None]).sum(1).abs().max()) < 1e-5 * float((C.abs() * A[None, :, None]).sum(1).max())
        phys = m.physics_only(x, c, g)  # the guide's own core (transport + source), V-Net off
        assert not torch.allclose(phys.s, x.s)


def test_core_needs_flow_head_and_excludes_drift():
    from gpclimate.models.lie_physical_gnn import GuideConfig
    cfg = VNetConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), heads=2, ds=4, dv=2, hidden=16, n_basis=4, fusion="pigt",
                     pigt_layers=1, integrator="ode", tendency_core=True)
    with pytest.raises(ValueError):
        LieGuidedForecaster(DEFAULT_SPEC, GuideConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), hidden=16, n_basis=4,
                                                      tend_head=True), cfg)
    with pytest.raises(ValueError):
        LieGuidedForecaster(DEFAULT_SPEC, GUIDE, VNetConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), heads=2, ds=4, dv=2,
                                                            hidden=16, n_basis=4, fusion="pigt", pigt_layers=1,
                                                            integrator="ode", tendency_core=True, ode_physics=True))
    # the guide-free pair of a core run is the plain ODE V-Net
    pair = LieGuidedForecaster(DEFAULT_SPEC, GUIDE, VNetConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), heads=2, ds=4, dv=2,
                                                               hidden=16, n_basis=4, fusion="pigt", pigt_layers=1,
                                                               integrator="ode", tendency_core=True, use_guide=False))
    assert pair.guide is None and not pair.vnet.core_on and pair.vnet.flux_head is None


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("core")
    base, col = d / "synthetic.npz", d / "columns.npz"
    _write_data(base)
    _write_columns(col, base)
    return base, col


def test_stage18_pipeline(files, tmp_path):
    base, col = files
    common = ["--fusion", "pigt", "--pigt-layers", "1", "--pigt-ops", "1", "--curvature-ops", "1", "--prev-state", "1",
              "--column-inputs", "fcn10", "--column-data", str(col)]
    roll = ("--rollout-steps", "2", "--rollout-start", "0")
    src = tmp_path / "src"  # stands in for stage9b
    _run(src, base, "b_noguide", *common, "--steps", "2", "--val-every", "2")
    _run(src, base, "a", "--steps", "2", "--val-every", "2")
    _run(src, base, "b", "--steps", "2", "--val-every", "2")
    run_dir = tmp_path / "run_s18"
    run_dir.mkdir()
    for f in ("normalizer.pt", "column_norm.json"):
        (run_dir / f).write_bytes((src / f).read_bytes())
    flags = common + ["--flow-head", "1", "--integrator", "ode", "--ode-steps", "2", "--ode-drift", "transport",
                      "--ode-path-forcing", "1", "--inter-times", "1", "--tendency-core", "1"]
    _run(run_dir, base, "a", *flags, "--steps", "3", "--val-every", "2", "--init-guide", str(src / "guide.pt"),
         "--perturb-model", str(src / "vnet_noguide.pt"), "--guide-heads-only", "1")
    cfg = json.load(open(run_dir / "config.json"))
    assert cfg["tendency_core"] == 1 and cfg["core_tokens"] == 1 and cfg["ode_physics"] == 0
    out = _run(run_dir, base, "b", "--steps", "3", "--val-every", "2", *roll, "--init-vnet", str(src / "vnet.pt"))
    assert "graph tendency core" in out and " core beta " in out
    rows = [json.loads(line) for line in open(run_dir / "log_vnet.jsonl") if '"val_mse"' in line]
    assert rows and "core" in rows[-1] and len(rows[-1]["core"]["beta"]) == 3
    assert any(len(m[0]) == 4 for m in rows[-1]["attention_mass"].values() if m and m[0])  # H, P, O, C
    _run(run_dir, base, "b_noguide", "--steps", "2", "--val-every", "2", *roll, "--init-vnet", str(src / "vnet_noguide.pt"))
    _run(run_dir, base, "eval", "--eval-stride", "4", "--max-lead-steps", "2", "--n-ens-cases", "0")
    ev = json.load(open(run_dir / "eval.json"))
    for name in ("lie_guided", "no_guide", "physics_only"):
        assert all(np.isfinite(v) for v in ev["rmse"]["2"][name].values()), name
    r = subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "scripts/diagnose_guide.py"), "--run", str(run_dir),
                        "--threads", "1"], capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]


def test_prev_path_follows_the_ode_path():
    g = _graphs().view()
    x, c = _state(2, wind=0.5)
    torch.manual_seed(3)
    prevs = [Field(x.s + 0.3 * torch.randn_like(x.s), x.v) for _ in range(4)]
    outs = {}
    for flag in (False, True):
        torch.manual_seed(0)
        cfg = VNetConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), heads=2, ds=4, dv=2, hidden=16, n_basis=4, fusion="pigt",
                         pigt_layers=1, integrator="ode", ode_steps=4, ode_drift="transport", prev_state=True,
                         tendency_core=True, prev_path=flag)
        m = LieGuidedForecaster(DEFAULT_SPEC, GUIDE, cfg).eval()
        with torch.no_grad():
            m.vnet.head_out.lin_s.weight.normal_(0, 0.1)
            outs[flag] = (m(x, c, g, prev=prevs)[0], m(x, c, g, prev=prevs[0])[0])
            nxt = m.next_prev(x)
        if flag:
            assert isinstance(nxt, list) and len(nxt) == 4 and nxt[0] is x  # [start, X+6h, X+12h, X+18h]
            assert all(torch.equal(a.s, b.s) for a, b in zip(nxt[1:], m.last_path))
        else:
            assert nxt is x
    assert torch.allclose(outs[False][0].s, outs[False][1].s)  # off: a list means its first entry for every sub-step
    assert not torch.allclose(outs[True][0].s, outs[True][1].s)  # on: sub-step k reads its own previous state


def test_dataset_prev_path(files):
    from gpclimate.data.era5_wb2 import ERA5Pairs
    base, _ = files
    ds = ERA5Pairs(str(base), "train", DEFAULT_SPEC, lead_steps=4)
    idx = [6, 9]
    path = ds.prev_path(idx)
    assert len(path) == 4 and torch.equal(path[0].s, ds.prev_state(idx).s)
    assert torch.equal(path[3].s, ds.prev_path([i + 1 for i in idx])[2].s)  # one 6 h step later = next entry


def test_stage18b_and_18c_pipeline(files, tmp_path):
    base, col = files
    common = ["--fusion", "pigt", "--pigt-layers", "1", "--pigt-ops", "1", "--curvature-ops", "1", "--prev-state", "1",
              "--column-inputs", "fcn10", "--column-data", str(col)]
    roll = ("--rollout-steps", "2", "--rollout-start", "0")
    src = tmp_path / "src"
    _run(src, base, "b_noguide", *common, "--steps", "2", "--val-every", "2")
    _run(src, base, "a", "--steps", "2", "--val-every", "2")
    _run(src, base, "b", "--steps", "2", "--val-every", "2")
    run_dir = tmp_path / "run_s18b"
    run_dir.mkdir()
    for f in ("normalizer.pt", "column_norm.json"):
        (run_dir / f).write_bytes((src / f).read_bytes())
    flags = common + ["--flow-head", "1", "--integrator", "ode", "--ode-steps", "2", "--ode-drift", "transport",
                      "--ode-path-forcing", "1", "--inter-times", "1", "--tendency-core", "1", "--prev-path", "1"]
    _run(run_dir, base, "a", *flags, "--steps", "2", "--val-every", "2", "--init-guide", str(src / "guide.pt"),
         "--guide-heads-only", "1")
    assert json.load(open(run_dir / "config.json"))["prev_path"] == 1
    _run(run_dir, base, "b_noguide", "--steps", "2", "--val-every", "2", *roll, "--init-vnet", str(src / "vnet_noguide.pt"))
    out = _run(run_dir, base, "b", "--steps", "2", "--val-every", "2", *roll, "--init-vnet", str(src / "vnet.pt"),
               "--guide-lr", "1e-3", "--guide-heads-only", "1")
    assert "guide fine-tuning restricted to the heads" in out
    import torch as _t
    old_g, new_g = _t.load(run_dir / "guide.pt"), _t.load(run_dir / "guide_finetuned.pt")
    heads = ("head_dec.", "rec_out.", "aux_out.", "tend_out.", "flow_out.")
    assert all(_t.equal(old_g[k], new_g[k]) for k in old_g if not k.startswith(heads))  # trunk / P unchanged
    assert any(not _t.equal(old_g[k], new_g[k]) for k in old_g if k.startswith(("flow_out.", "tend_out.")))
    _run(run_dir, base, "eval", "--eval-stride", "4", "--max-lead-steps", "2", "--n-ens-cases", "0")
    ev = json.load(open(run_dir / "eval.json"))
    for name in ("lie_guided", "no_guide", "physics_only"):
        assert all(np.isfinite(v) for v in ev["rmse"]["2"][name].values()), name
