"""Geometry and SO(2) gauge-equivariance tests (CPU, tiny graphs, random weights)."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpclimate.data.variables import VariableSpec  # noqa: E402
from gpclimate.geometry import sphere  # noqa: E402
from gpclimate.geometry.graphs import SphereGraphs  # noqa: E402
from gpclimate.models.diffusion import DiffusionConfig, GraphResidualDiffusion, ResidualDenoiser  # noqa: E402
from gpclimate.models.dynamics_vnet import LieGuidedForecaster, VNetConfig  # noqa: E402
from gpclimate.models.lie_physical_gnn import GuideConfig  # noqa: E402
from gpclimate.nn.attention import PhysicsFusion  # noqa: E402
from gpclimate.nn.gauge_conv import GaugeBlock, GaugeTransfer  # noqa: E402
from gpclimate.nn.so2 import Field  # noqa: E402
from gpclimate.training.losses import gauge_equivariance_report, random_gauges, relative_error  # noqa: E402

torch.manual_seed(0)
TOL = 2e-5

SPEC = VariableSpec(scalars=("z500", "t850"), vectors=(("u850", "v850"), ("u10", "v10")),
                    static_scalars=("orog",), static_vectors=(("dox", "doy"),), forcing_scalars=("toa",))


@pytest.fixture(scope="module")
def graphs():
    lat = np.linspace(-90, 90, 13)  # includes both poles (KAI-like)
    lon = np.arange(24) * 15.0
    return SphereGraphs(lat, lon, mesh_splits=(2, 1, 0))


def rand_field(B, N, cs, cv):
    return Field(torch.randn(B, N, cs), torch.complex(torch.randn(B, N, cv), torch.randn(B, N, cv)))


def test_areas_sum_to_sphere(graphs):
    assert np.isclose(graphs.grid.area.sum(), 4 * np.pi)
    for m in graphs.mesh:
        assert np.isclose(m.area.sum(), 4 * np.pi)


def test_transport_angle_matches_3d_rotation(graphs):
    """exp(i alpha) z_j equals the frame-i coordinates of the Levi-Civita transported 3D vector."""
    m = graphs.mesh[0]
    e = graphs.mesh_edges[0]
    src, dst = e.src.numpy(), e.dst.numpy()
    z = np.random.randn(len(src)) + 1j * np.random.randn(len(src))
    vec = z.real[:, None] * m.e1[src] + z.imag[:, None] * m.e2[src]
    R = sphere.transport_rotation(m.pos[src], m.pos[dst])
    tv = np.einsum("eij,ej->ei", R, vec)
    coords = np.sum(tv * m.e1[dst], -1) + 1j * np.sum(tv * m.e2[dst], -1)
    assert np.allclose(coords, np.exp(1j * e.alpha.double().numpy()) * z, atol=1e-5)


def test_geodesic_direction_identity(graphs):
    """On the sphere, phi_ji + alpha_ij = phi_ij + pi (transported back-direction is opposite)."""
    e = graphs.mesh_edges[1]
    key = {(int(s), int(d)): k for k, (s, d) in enumerate(zip(e.src, e.dst))}
    diffs = []
    for k, (s, d) in enumerate(zip(e.src.tolist(), e.dst.tolist())):
        r = key[(d, s)]
        diffs.append(np.angle(np.exp(1j * (float(e.phi[r]) + float(e.alpha[k]) - float(e.phi[k]) - math.pi))))
    assert np.max(np.abs(diffs)) < 1e-5


def test_gauge_transform_closed_form(graphs):
    """Recomputing edge data with rotated frames equals the closed-form update."""
    from gpclimate.geometry.graphs import _make_edges

    m = graphs.mesh[0]
    th = np.random.rand(m.n) * 2 * np.pi
    m2 = m.with_gauge(th)
    e = graphs.mesh_edges[0]
    e2 = _make_edges(m2, m2, e.src.numpy(), e.dst.numpy(), e.cutoff)
    e3 = e.gauge_transformed(torch.as_tensor(th, dtype=torch.float32), torch.as_tensor(th, dtype=torch.float32))
    for a, b in ((e2.alpha, e3.alpha), (e2.phi, e3.phi)):
        assert np.max(np.abs(np.angle(np.exp(1j * (a - b).double().numpy())))) < 1e-4


@pytest.mark.parametrize("layer", ["block", "transfer", "fusion"])
def test_layer_equivariance(graphs, layer):
    tg, tm = random_gauges(graphs, torch.Generator().manual_seed(1))
    gv = graphs.gauge_transformed(tm, tg)
    g0 = graphs.view()
    cs, cv = 6, 3
    with torch.no_grad():
        if layer == "block":
            mod = GaugeBlock(cs, cv, 4, 16)
            x = rand_field(2, graphs.mesh[0].n, cs, cv)
            y0, y1 = mod(x, g0.mesh_edges[0]), mod(x.gauge(tm[0]), gv.mesh_edges[0])
            th = tm[0]
        elif layer == "transfer":
            mod = GaugeTransfer(cs, cv, cs, cv, 4, 16)
            x = rand_field(2, graphs.mesh[0].n, cs, cv)
            y0, y1 = mod(x, g0.down[0]), mod(x.gauge(tm[0]), gv.down[0])
            th = tm[1]
        else:
            mod = PhysicsFusion(cs, cv, 5, 2, heads=2, ds=4, dv=2, n_basis=4)
            h = rand_field(2, graphs.mesh[1].n, cs, cv)
            p = rand_field(2, graphs.mesh[1].n, 5, 2)
            y0 = mod(h, p, g0.attn_edges[1])
            y1 = mod(h.gauge(tm[1]), p.gauge(tm[1]), gv.attn_edges[1])
            th = tm[1]
    assert relative_error(y1, y0.gauge(th)) < TOL


def test_full_forecaster_equivariance(graphs):
    model = LieGuidedForecaster(SPEC, GuideConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), hidden=16),
                                VNetConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), heads=2, ds=4, dv=2, hidden=16))
    for p in model.vnet.head_out.parameters():  # head starts at zero (persistence); randomise for the test
        torch.nn.init.normal_(p, std=0.1)
    G = graphs.grid.n
    x = rand_field(2, G, SPEC.n_s, SPEC.n_v)
    c = rand_field(1, G, SPEC.n_cond_s, SPEC.n_cond_v)

    def fn(xx, cc, gv):
        f, d = model(xx, cc, gv)
        P = model.physics(xx, cc, gv)
        return {"forecast": f, "delta": d, **P}

    lv = {"forecast": "grid", "delta": "grid", "P0": "0", "P1": "1", "P2": "2", "P1u": "1", "P0u": "0"}
    rep = gauge_equivariance_report(fn, x, c, graphs, lv)
    assert max(rep.values()) < TOL, rep
    # the output must actually depend on the input (guards against a trivial pass)
    with torch.no_grad():
        f1, _ = model(x, c, graphs.view())
    assert relative_error(f1, x) > 1e-4


def test_denoiser_equivariance(graphs):
    den = ResidualDenoiser(SPEC, DiffusionConfig(cs=8, cv=4, depth=1, hidden=16), guide_dims=(5, 2))
    G, N0 = graphs.grid.n, graphs.mesh[0].n
    ctx = {"xbar": rand_field(1, G, SPEC.n_s, SPEC.n_v), "x": rand_field(1, G, SPEC.n_s, SPEC.n_v),
           "cond": rand_field(1, G, SPEC.n_cond_s, SPEC.n_cond_v), "P0u": rand_field(1, N0, 5, 2)}
    r = rand_field(1, G, SPEC.n_s, SPEC.n_v)
    tg, tm = random_gauges(graphs, torch.Generator().manual_seed(3))
    gv = graphs.gauge_transformed(tm, tg)
    ctx_g = {"xbar": ctx["xbar"].gauge(tg), "x": ctx["x"].gauge(tg), "cond": ctx["cond"].gauge(tg),
             "P0u": ctx["P0u"].gauge(tm[0])}
    s = torch.tensor([10])
    with torch.no_grad():
        e0 = den(r, s, ctx, graphs.view())
        e1 = den(r.gauge(tg), s, ctx_g, gv)
    assert relative_error(e1, e0.gauge(tg)) < TOL


def test_diffusion_loss_and_sampling_shapes(graphs):
    den = ResidualDenoiser(SPEC, DiffusionConfig(cs=8, cv=4, depth=1, hidden=16), guide_dims=(5, 2))
    diff = GraphResidualDiffusion(den, 100)
    G, N0 = graphs.grid.n, graphs.mesh[0].n
    ctx = {"xbar": rand_field(2, G, SPEC.n_s, SPEC.n_v), "x": rand_field(2, G, SPEC.n_s, SPEC.n_v),
           "cond": rand_field(1, G, SPEC.n_cond_s, SPEC.n_cond_v), "P0u": rand_field(2, N0, 5, 2)}
    loss = diff.loss(rand_field(2, G, SPEC.n_s, SPEC.n_v), ctx, graphs.view(), graphs.grid_area_weight)
    loss.backward()
    assert torch.isfinite(loss)
    mem = diff.sample(ctx, graphs.view(), n_members=2, steps=3, seed=0)
    assert len(mem) == 2 and mem[0].s.shape == (2, G, SPEC.n_s) and mem[0].v.shape == (2, G, SPEC.n_v)
    assert relative_error(mem[0], mem[1]) > 1e-3  # independent seeds -> different members


# ----------------------------------------------------------------------------- fixed physics operators
def test_physics_operators_analytic(graphs):
    """LS stencil reproduces grad / curl / div of analytic fields on the finest mesh."""
    from gpclimate.nn.physics_ops import PhysicsOperators, Stencil

    m = graphs.mesh[0]
    e = graphs.mesh_edges[0]
    lat = np.deg2rad(m.lat)
    st = Stencil(e)
    s = torch.as_tensor(np.sin(lat), dtype=torch.float32)[None, :, None]  # s = z  ->  grad = cos(lat) * north
    g = st.grad(s)[0, :, 0]
    expect = 1j * torch.as_tensor(np.cos(lat), dtype=torch.float32) * st.h  # per mean-edge-length unit
    interior = np.abs(np.cos(lat)) > 0.3
    err = (g - expect).abs()[interior] / expect.abs()[interior]
    assert float(err.median()) < 0.05, float(err.median())
    # solid-body rotation u = cos(lat) east: divergence 0, relative vorticity 2 sin(lat)
    u = torch.complex(torch.as_tensor(np.cos(lat), dtype=torch.float32), torch.zeros(m.n))[None, :, None]
    J = st.jacobian(u)[0, :, 0]
    div = J[0, 0] + J[1, 1] if J.dim() == 2 else J[..., 0, 0] + J[..., 1, 1]
    curl = J[..., 1, 0] - J[..., 0, 1]
    curl_expect = 2 * torch.as_tensor(np.sin(lat), dtype=torch.float32) * st.h
    sel = np.abs(np.sin(lat)) > 0.3
    assert float(((curl - curl_expect).abs()[sel] / curl_expect.abs()[sel]).median()) < 0.1
    assert float(div.abs().median()) < 0.1 * float(curl_expect.abs().median())
    ops = PhysicsOperators(1, 1, 0)
    out = ops(s, u, s.new_zeros(1, m.n, 0), graphs.mesh_sinlat[0], e)
    assert out.s.shape == (1, m.n, ops.out_dims[0]) and out.v.shape == (1, m.n, ops.out_dims[1])
    assert torch.isfinite(out.s).all() and torch.isfinite(out.v).all()


def test_physics_operators_equivariance(graphs):
    from gpclimate.nn.physics_ops import PhysicsOperators

    ops = PhysicsOperators(3, 2, 2, v_scale_mps=torch.tensor([8.0, 5.0]))
    N = graphs.mesh[0].n
    s, v, c = torch.randn(2, N, 3), torch.complex(torch.randn(2, N, 2), torch.randn(2, N, 2)), torch.randn(1, N, 2)
    tg, tm = random_gauges(graphs)
    gv = graphs.gauge_transformed(tm, tg)
    y0 = ops(s, v, c, graphs.mesh_sinlat[0], graphs.mesh_edges[0])
    y1 = ops(s, v * torch.exp(-1j * tm[0])[None, :, None], c, gv.mesh_sinlat[0], gv.mesh_edges[0])
    assert relative_error(y1, y0.gauge(tm[0])) < TOL
    # scalar outputs are exactly invariant, vectors rotate: guard against trivially small outputs
    assert float(y0.s.abs().mean()) > 1e-3 and float(y0.v.abs().mean()) > 1e-3


def test_guide_with_physics_ops_equivariance(graphs):
    cfg = GuideConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), hidden=16, physics_ops=True, tend_head=True)
    model = LieGuidedForecaster(SPEC, cfg, VNetConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), heads=2, ds=4, dv=2,
                                                      hidden=16))
    model.guide.phys_ops.v_scale.copy_(torch.tensor([8.0, 4.0]))
    G = graphs.grid.n
    x = rand_field(2, G, SPEC.n_s, SPEC.n_v)
    c = rand_field(1, G, SPEC.n_cond_s, SPEC.n_cond_v)

    def fn(xx, cc, gv):
        P = model.physics(xx, cc, gv)
        rec, aux, tend = model.guide.heads_all(P, gv)
        return {"rec": rec, "tend": tend, "aux": Field(aux, aux.new_zeros(*aux.shape[:-1], 0, dtype=torch.complex64)),
                **P}

    lv = {"rec": "grid", "tend": "grid", "aux": "grid", "P0": "0", "P1": "1", "P2": "2", "P1u": "1", "P0u": "0"}
    rep = gauge_equivariance_report(fn, x, c, graphs, lv)
    assert max(rep.values()) < TOL, rep
    # the operator branch must change the features (guards against a dead branch)
    with torch.no_grad():
        P_on = model.physics(x, c, graphs.view())["P0"]
        model.guide.phys_ops = None
        P_off = model.physics(x, c, graphs.view())["P0"]
    assert relative_error(P_on, P_off) > 1e-3
