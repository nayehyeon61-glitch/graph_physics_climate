"""ODE sub-step gradient checkpointing (stage17a OOM fix, 2026-11-08): same forecast, same path states and the same
gradients as without checkpointing, for the plain ODE V-Net and the stage18 graph tendency core with the stage18d
Laplace block; evaluation without gradients never checkpoints."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from dataclasses import replace  # noqa: E402

from gpclimate.data.era5_wb2 import DEFAULT_SPEC  # noqa: E402
from gpclimate.models.dynamics_vnet import LieGuidedForecaster, VNetConfig  # noqa: E402
from test_dynamic_guide import GUIDE, _graph, _inputs  # noqa: E402


def _model(**kw):
    torch.manual_seed(0)
    core = kw.get("tendency_core", False)
    cfg = VNetConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), heads=2, ds=4, dv=2, hidden=16, n_basis=4, fusion="pigt",
                     pigt_layers=1, prev_state=True, col_s=2, col_v=1, integrator="ode", ode_steps=4,
                     ode_drift="transport" if core else "tend", **kw)
    m = LieGuidedForecaster(DEFAULT_SPEC, replace(GUIDE, flow_head=core), cfg)
    with torch.no_grad():
        m.vnet.head_out.lin_s.weight.normal_(0, 0.1)
        m.vnet.head_out.w_re.normal_(0, 0.1)
        if core:
            m.guide.flow_v_scale.fill_(10.0)
    return m.eval()


def _grads(m, g, x0, c, prev, col):
    m.zero_grad()
    f, _ = m(x0, c, g, prev=prev, col=col)
    loss = f.s.pow(2).mean() + f.v.abs().pow(2).mean() + sum(p.s.pow(2).mean() for p in m.last_path)
    loss.backward()
    return f, [p.s.detach().clone() for p in m.last_path], {n: p.grad.clone() for n, p in m.named_parameters()
                                                           if p.grad is not None}


@pytest.mark.parametrize("kw", [dict(ode_path_forcing=True, inter_times=3),
                                dict(ode_path_forcing=True, tendency_core=True, core_laplace=True)])
def test_checkpointed_sub_steps_give_the_same_gradients(kw):
    g = _graph()
    x0, c, prev, col, _ = _inputs()
    out = {}
    for flag in (False, True):
        m = _model(ode_checkpoint=flag, **kw).train()
        for p in m.guide.parameters():
            p.requires_grad_(True)  # guide heads fine-tuned in Stage B (stage18c) must get the same gradients too
        out[flag] = _grads(m, g, x0, c, prev, col)
    (f0, p0, g0), (f1, p1, g1) = out[False], out[True]
    assert torch.allclose(f0.s, f1.s, atol=1e-6) and torch.allclose(f0.v, f1.v, atol=1e-6)
    assert len(p0) == 3 and all(torch.allclose(a, b, atol=1e-6) for a, b in zip(p0, p1))
    assert g0.keys() == g1.keys() and len(g0) > 10
    for n in g0:
        assert torch.allclose(g0[n], g1[n], rtol=1e-4, atol=1e-7), n


def test_no_checkpoint_without_gradients(monkeypatch):
    import torch.utils.checkpoint as cp
    called = []
    monkeypatch.setattr(cp, "checkpoint", lambda *a, **k: called.append(1))
    g = _graph()
    x0, c, prev, col, _ = _inputs()
    m = _model(ode_checkpoint=True)
    with torch.no_grad():
        m(x0, c, g, prev=prev, col=col)
    assert not called
