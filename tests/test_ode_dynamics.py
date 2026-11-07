"""Stage S8: graph neural ODE dynamics on top of the learned physics (fixed-step integration of the frozen guide's
tendency + the V-Net correction): reduction to the delta model, pure-physics limit, gauge equivariance."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpclimate.data.era5_wb2 import DEFAULT_SPEC  # noqa: E402
from gpclimate.data.variables import VariableSpec  # noqa: E402
from gpclimate.geometry.graphs import SphereGraphs  # noqa: E402
from gpclimate.models.dynamics_vnet import LieGuidedForecaster, VNetConfig  # noqa: E402
from gpclimate.models.lie_physical_gnn import GuideConfig  # noqa: E402
from gpclimate.nn.so2 import Field  # noqa: E402
from gpclimate.training.losses import gauge_equivariance_report, relative_error  # noqa: E402

torch.manual_seed(0)
TOL = 2e-5
SPEC = VariableSpec(scalars=("z500", "t850"), vectors=(("u850", "v850"), ("u10", "v10")),
                    static_scalars=("orog",), static_vectors=(("dox", "doy"),), forcing_scalars=("toa",))


def _graphs():
    return SphereGraphs(np.linspace(-90, 90, 13), np.arange(24) * 15.0, mesh_splits=(2, 1, 0))


def _rand(B, N, cs, cv):
    return Field(torch.randn(B, N, cs), torch.complex(torch.randn(B, N, cv), torch.randn(B, N, cv)))


def _model(spec, steps, solver, physics, **guide_kw):
    gcfg = GuideConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), hidden=16, physics_ops=True, tend_head=True, **guide_kw)
    vcfg = VNetConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), heads=2, ds=4, dv=2, hidden=16, integrator="ode",
                      ode_steps=steps, ode_solver=solver, ode_physics=physics)
    m = LieGuidedForecaster(spec, gcfg, vcfg)
    m.delta_scale_s.copy_(torch.rand(spec.n_s) + 0.5)
    m.delta_scale_v.copy_(torch.rand(spec.n_v) + 0.5)
    m.tend_scale_s.copy_(torch.rand(spec.n_s) * 0.2 + 0.1)
    m.tend_scale_v.copy_(torch.rand(spec.n_v) * 0.2 + 0.1)
    m.lead_steps = 4
    return m


def test_one_euler_step_without_drift_is_the_delta_model():
    gr = _graphs(); g = gr.view()
    m = _model(SPEC, 1, "euler", False)
    for p in m.vnet.head_out.parameters():
        torch.nn.init.normal_(p, std=0.1)
    x, c = _rand(2, gr.grid.n, SPEC.n_s, SPEC.n_v), _rand(1, gr.grid.n, SPEC.n_cond_s, SPEC.n_cond_v)
    with torch.no_grad():
        f_ode, d_ode = m(x, c, g)
        m.vnet.cfg.integrator = "delta"
        f_delta, d_delta = m(x, c, g)
    assert relative_error(f_ode, f_delta) < 1e-6 and relative_error(d_ode, d_delta) < 1e-5


def test_zero_head_gives_the_pure_physics_integration():
    """With the V-Net head at its zero initialisation, the ODE forecast is the guide's tendency integrated in K
    Euler sub-steps of (lead / K) data steps -- the model starts from physics instead of persistence."""
    gr = _graphs(); g = gr.view()
    m = _model(SPEC, 2, "euler", True)
    x, c = _rand(2, gr.grid.n, SPEC.n_s, SPEC.n_v), _rand(1, gr.grid.n, SPEC.n_cond_s, SPEC.n_cond_v)
    with torch.no_grad():
        f, _ = m(x, c, g)
        xk = x
        for _ in range(2):
            T = m.guide.physics_tendency(m.physics(xk, c, g), xk, g)
            xk = Field(xk.s + T.s * m.tend_scale_s * 2.0, xk.v + T.v * m.tend_scale_v * 2.0)
        assert relative_error(f, xk) < 1e-6
        assert relative_error(f, m.physics_only(x, c, g)) < 1e-6
        assert relative_error(f, x) > 1e-4  # the drift is not trivially zero


def test_ode_forecaster_equivariance_heun_with_drift():
    gr = _graphs(); g = gr.view()
    m = _model(SPEC, 2, "heun", True)
    for p in m.vnet.head_out.parameters():
        torch.nn.init.normal_(p, std=0.1)
    x, c = _rand(2, gr.grid.n, SPEC.n_s, SPEC.n_v), _rand(1, gr.grid.n, SPEC.n_cond_s, SPEC.n_cond_v)

    def fn(xx, cc, gv):
        f, d = m(xx, cc, gv)
        return {"forecast": f, "delta": d}

    rep = gauge_equivariance_report(fn, x, c, gr, {"forecast": "grid", "delta": "grid"})
    assert max(rep.values()) < TOL, rep
    # gradients reach the V-Net through every sub-step and the guide stays frozen
    m.freeze_guide()
    f, _ = m(x, c, g)
    (f.s.square().mean() + f.v.abs().square().mean()).backward()
    assert all(p.grad is not None for p in m.vnet.head_out.parameters())
    assert all(p.grad is None for p in m.guide.parameters())


def test_closure_guide_physics_tendency_on_the_grid():
    """S5a closure guides provide the drift as F(X) + S_theta transported from mesh 0 to the grid."""
    gr = SphereGraphs(np.linspace(-87.1875, 87.1875, 16), np.arange(0, 360, 11.25), mesh_splits=(2, 1, 0)); g = gr.view()
    gcfg = GuideConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), hidden=16, physics_ops=True, tend_head=False,
                       closure_tend=True)
    vcfg = VNetConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), heads=2, ds=4, dv=2, hidden=16, integrator="ode",
                      ode_steps=4, ode_physics=True)
    m = LieGuidedForecaster(DEFAULT_SPEC, gcfg, vcfg)
    m.guide.resolved.set_scales(torch.tensor([54000.0, 280.0, 0.005, 288.0, 101300.0]),
                                torch.tensor([3000.0, 10.0, 0.003, 10.0, 1000.0]), torch.tensor([8.0, 12.0, 5.0]),
                                torch.full((5,), 0.1), torch.full((3,), 0.3))
    x = _rand(2, gr.grid.n, DEFAULT_SPEC.n_s, DEFAULT_SPEC.n_v)
    c = _rand(1, gr.grid.n, DEFAULT_SPEC.n_cond_s, DEFAULT_SPEC.n_cond_v)
    with torch.no_grad():
        T = m.guide.physics_tendency(m.physics(x, c, g), x, g)
        f, d = m(x, c, g)
    assert T.s.shape == (2, gr.grid.n, DEFAULT_SPEC.n_s) and T.v.shape == (2, gr.grid.n, DEFAULT_SPEC.n_v)
    assert f.s.shape == x.s.shape and torch.isfinite(f.s).all() and torch.isfinite(d.v).all()
