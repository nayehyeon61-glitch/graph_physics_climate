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


def test_pigt_forecaster_equivariance(graphs):
    """Physics-integrated graph-transformer fusion (H/P/O sources, bidirectional) stays gauge equivariant."""
    cfg = GuideConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), hidden=16, physics_ops=True, tend_head=True)
    model = LieGuidedForecaster(SPEC, cfg, VNetConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), heads=2, ds=4, dv=2,
                                                      hidden=16, fusion="pigt", pigt_layers=2, pigt_ops=True))
    for p in model.vnet.head_out.parameters():
        torch.nn.init.normal_(p, std=0.1)
    for f in model.vnet.fusion:  # make the attention branches large enough to matter in the test
        for layer in f.layers:
            for lin in (layer.attn_h.o, layer.attn_p.o) if layer.update_p else (layer.attn_h.o,):
                for p in lin.parameters():
                    torch.nn.init.normal_(p, std=0.3)
    G = graphs.grid.n
    x = rand_field(2, G, SPEC.n_s, SPEC.n_v)
    c = rand_field(1, G, SPEC.n_cond_s, SPEC.n_cond_v)

    def fn(xx, cc, gv):
        f, d = model(xx, cc, gv)
        return {"forecast": f, "delta": d}

    rep = gauge_equivariance_report(fn, x, c, graphs, {"forecast": "grid", "delta": "grid"})
    assert max(rep.values()) < TOL, rep
    mass = model.vnet.attention_mass()
    assert set(mass) == {"P0", "P1", "P2", "P1u", "P0u"}
    for k, layers in mass.items():
        for m in layers:
            assert m is not None and len(m) == 3 and abs(sum(m) - 1.0) < 1e-3, (k, m)
    # the fusion must change the output (guards against a dead branch)
    with torch.no_grad():
        f1, _ = model(x, c, graphs.view())
        P = model.physics(x, c, graphs.view())
        d_on = model.vnet(x, c, P, graphs.view())
        d_off = model.vnet(x, c, None, graphs.view())
    assert relative_error(d_on, d_off) > 1e-3


def test_reduced_primitive_tendencies_units_and_equivariance():
    """Stage S5a: fixed resolved dynamics on the 11-variable layout (bilinear grid->mesh sampling, SI units)."""
    from gpclimate.data.era5_wb2 import DEFAULT_SPEC
    from gpclimate.nn.primitive2d import DT_DATA, EARTH_RADIUS, ReducedPrimitiveTendencies

    lat = np.linspace(-87.1875, 87.1875, 32)
    lon = np.arange(0, 360, 5.625)
    gr = SphereGraphs(lat, lon, mesh_splits=(3, 2, 1))
    g = gr.view()
    G, m0 = gr.grid.n, gr.mesh[0]
    mlat = np.deg2rad(m0.lat)
    mid = np.abs(mlat) < np.deg2rad(60)
    LAT, LON = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    rp = ReducedPrimitiveTendencies(DEFAULT_SPEC)  # identity scales: normalised == physical
    s = torch.zeros(1, G, 5)
    s[..., 1], s[..., 2], s[..., 3], s[..., 4] = 280.0, 0.005, 290.0, 101325.0
    v = torch.zeros(1, G, 3, dtype=torch.complex64)
    # z500 = c * lat  ->  du500/dt = -grad Phi = -i c / R (northward component negative)
    c = 1e4
    s1 = s.clone()
    s1[..., 0] = torch.as_tensor(c * LAT.ravel(), dtype=torch.float32)
    fu = rp.mesh_forward(Field(s1, v), g).v[0, :, 1] / DT_DATA
    assert float((fu[mid] - (-1j * c / EARTH_RADIUS)).abs().mean() / (c / EARTH_RADIUS)) < 0.02
    # t850 = T0 + d sin(lon) cos(lat), u850 = U eastward  ->  dT/dt = -U d cos(lon) / R
    d, U = 5.0, 10.0
    s2 = s.clone()
    s2[..., 1] = torch.as_tensor(280 + d * np.sin(LON) * np.cos(LAT), dtype=torch.float32).ravel()
    v2 = v.clone()
    v2[..., 0] = U
    ft = rp.mesh_forward(Field(s2, v2), g).s[0, :, 1] / DT_DATA
    expt = torch.as_tensor(-U * d * np.cos(np.deg2rad(m0.lon)) / EARTH_RADIUS, dtype=torch.float32)
    assert float((ft[mid] - expt[mid]).abs().mean() / expt[mid].abs().mean()) < 0.03
    # geostrophic balance: Phi = A sin^2(lat), u_g = -(dPhi/dlat) / (f R)  ->  Coriolis + PGF nearly cancel
    A = 2e4
    f = 2 * 7.292e-5 * np.sin(LAT)
    ug = -(2 * A * np.sin(LAT) * np.cos(LAT)) / (f * EARTH_RADIUS + 1e-9)
    s3 = s.clone()
    s3[..., 0] = torch.as_tensor((A * np.sin(LAT) ** 2).ravel(), dtype=torch.float32)
    v3 = v.clone()
    v3[..., 1] = torch.as_tensor(ug.ravel(), dtype=torch.complex64)
    band = (np.abs(mlat) > np.deg2rad(25)) & (np.abs(mlat) < np.deg2rad(65))
    both = ReducedPrimitiveTendencies(DEFAULT_SPEC, terms=("coriolis", "pgf")).mesh_forward(Field(s3, v3), g).v[0, :, 1]
    pgf = ReducedPrimitiveTendencies(DEFAULT_SPEC, terms=("pgf",)).mesh_forward(Field(s3, v3), g).v[0, :, 1]
    assert float(both[band].abs().mean() / pgf[band].abs().mean()) < 0.05
    # exact SO(2) gauge equivariance of the full operator
    torch.manual_seed(0)
    sr = torch.randn(2, G, 5) * torch.tensor([3000.0, 10.0, 0.003, 10.0, 1000.0]) + torch.tensor(
        [54000.0, 280.0, 0.005, 288.0, 101300.0])
    vr = torch.complex(torch.randn(2, G, 3), torch.randn(2, G, 3)) * 8
    x = Field(sr, vr)
    tg, tm = random_gauges(gr)
    gv = gr.gauge_transformed(tm, tg)
    assert relative_error(rp.mesh_forward(x.gauge(tg), gv), rp.mesh_forward(x, g).gauge(tm[0])) < TOL


def test_guide_closure_head_equivariance():
    """Stage S5a guide: closure head on mesh 0 and the resolved dynamics are equivariant together."""
    from gpclimate.data.era5_wb2 import DEFAULT_SPEC

    lat = np.linspace(-87.1875, 87.1875, 16)
    lon = np.arange(0, 360, 11.25)
    gr = SphereGraphs(lat, lon, mesh_splits=(2, 1, 0))
    g = gr.view()
    cfg = GuideConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), hidden=16, physics_ops=True, tend_head=False,
                      closure_tend=True)
    model = LieGuidedForecaster(DEFAULT_SPEC, cfg, VNetConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), heads=2, ds=4, dv=2,
                                                              hidden=16))
    model.guide.resolved.set_scales(torch.tensor([54000.0, 280.0, 0.005, 288.0, 101300.0]),
                                    torch.tensor([3000.0, 10.0, 0.003, 10.0, 1000.0]), torch.tensor([8.0, 12.0, 5.0]),
                                    torch.full((5,), 0.1), torch.full((3,), 0.3))
    G = gr.grid.n
    x = rand_field(2, G, DEFAULT_SPEC.n_s, DEFAULT_SPEC.n_v)
    c = rand_field(1, G, DEFAULT_SPEC.n_cond_s, DEFAULT_SPEC.n_cond_v)
    tg, tm = random_gauges(gr)
    gv = gr.gauge_transformed(tm, tg)
    with torch.no_grad():
        P0 = model.physics(x, c, g)
        P1 = model.physics(x.gauge(tg), c.gauge(tg), gv)
        y0 = model.guide.tendency_mesh(P0, x, g)
        y1 = model.guide.tendency_mesh(P1, x.gauge(tg), gv)
    assert relative_error(y1, y0.gauge(tm[0])) < TOL
    assert float(model.guide.resolved_tendency(x, g).v.abs().mean()) > 1e-6


# ----------------------------------------------------------------------------- stage S5c: Noise-layer constraints
def test_feasibility_projection_constraints_gauge_and_bias_estimator(graphs):
    """P_C enforces q700 >= 0, equal total mass and column water; it is idempotent, leaves the other channels
    untouched, commutes with gauge changes; ``pairwise_bias`` estimates the squared ensemble-mean bias without the
    spread term."""
    from gpclimate.models.diffusion import FeasibilityProjection, pairwise_bias

    spec = VariableSpec(scalars=("z500", "q700", "msl"), vectors=(("u850", "v850"),), static_scalars=("orog",),
                        static_vectors=(("dox", "doy"),), forcing_scalars=("toa",))
    G, w = graphs.grid.n, graphs.grid_area_weight
    s_mean, s_std = torch.tensor([5.4e4, 2.0e-3, 1.0e5]), torch.tensor([3.0e3, 2.0e-3, 1.0e3])
    proj = FeasibilityProjection(spec, s_mean, s_std, water=True)
    gen = torch.Generator().manual_seed(5)
    mu = rand_field(2, G, 3, 1)
    mu.s[..., 1] = 0.3 * torch.rand(2, G, generator=gen) - 0.6  # q700 in a plausible normalised range (0 kg/kg = -1)
    x = Field(mu.s + 0.6 * torch.randn(2, G, 3, generator=gen), mu.v + 0.1 * torch.randn(2, G, 1, generator=gen))
    assert (x.s[..., 1] < proj.q_min).float().mean() > 0.05  # the raw sample violates q >= 0 somewhere
    y = proj(x, mu, w)
    am = lambda t: (t * w).sum(-1) / w.sum()  # noqa: E731
    water = lambda f: am((f.s[..., 2] * s_std[2] + s_mean[2]) * (f.s[..., 1] * s_std[1] + s_mean[1]))  # noqa: E731
    assert (y.s[..., 1] >= proj.q_min - 1e-6).all()
    assert torch.allclose(am(y.s[..., 2]), am(mu.s[..., 2]), atol=1e-5)
    assert torch.allclose(water(y), water(mu), rtol=1e-4)
    assert torch.equal(y.s[..., 0], x.s[..., 0]) and torch.equal(y.v, x.v)
    assert relative_error(proj(y, mu, w), y) < 1e-6  # idempotent
    tg, _ = random_gauges(graphs, torch.Generator().manual_seed(6))
    assert relative_error(proj(x.gauge(tg), mu.gauge(tg), w), y.gauge(tg)) < TOL
    v = proj.violations(y, mu, w)
    assert float(v["neg_q_fraction"].max()) == 0.0 and float(v["mass_drift_pa"].abs().max()) < 1e-2
    assert float(v["water_drift_rel"].abs().max()) < 1e-4
    v0 = proj.violations(x, mu, w)
    assert float(v0["neg_q_fraction"].mean()) > 0.05 and float(v0["mass_drift_pa"].abs().max()) > 1.0
    # unbiased squared-bias estimator: zero-bias members -> ~0, biased members -> |b|^2 (not + variance / K)
    zero = Field(torch.zeros(2, G, 3), torch.zeros(2, G, 1, dtype=torch.complex64))
    scale_s, scale_v = torch.ones(3), torch.ones(1)
    noise = lambda: rand_field(2, G, 3, 1)  # noqa: E731
    members = [noise() for _ in range(6)]
    assert abs(float(pairwise_bias(members, zero, scale_s, scale_v, w))) < 0.05
    b = 0.5
    biased = [Field(m.s + b, m.v) for m in members]
    expect = 3 * b ** 2 / (3 + 2 * 1)  # per real component: scalars biased, vector components not
    assert abs(float(pairwise_bias(biased, zero, scale_s, scale_v, w)) - expect) < 0.05
    # the naive K-member mean error would include the spread term (variance 1 per component) / K
    mean_s = torch.stack([m.s for m in biased]).mean(0)
    naive = float((((mean_s - zero.s) ** 2) * w[None, :, None]).sum() / (w.sum() * 3 * 2))
    assert naive > expect + 0.5 / 6 * 0.5


def test_constrained_diffusion_losses_and_mean_constraint(graphs):
    """Stage S5c losses: the feasibility penalty is finite and non-negative, the mean constraint is differentiable
    through the sampler and the projection, and projected samples satisfy the constraints."""
    from gpclimate.models.diffusion import FeasibilityProjection

    spec = VariableSpec(scalars=("z500", "q700", "msl"), vectors=(("u850", "v850"),), static_scalars=("orog",),
                        static_vectors=(("dox", "doy"),), forcing_scalars=("toa",))
    G, N0, w = graphs.grid.n, graphs.mesh[0].n, graphs.grid_area_weight
    proj = FeasibilityProjection(spec, torch.tensor([5.4e4, 2.0e-3, 1.0e5]), torch.tensor([3.0e3, 2.0e-3, 1.0e3]))
    den = ResidualDenoiser(spec, DiffusionConfig(cs=8, cv=4, depth=1, hidden=16), guide_dims=(5, 2))
    diff = GraphResidualDiffusion(den, 100, proj)
    ctx = {"xbar": rand_field(2, G, 3, 1), "x": rand_field(2, G, 3, 1),
           "cond": rand_field(1, G, spec.n_cond_s, spec.n_cond_v), "P0u": rand_field(2, N0, 5, 2)}
    ls = diff.losses(rand_field(2, G, 3, 1), ctx, graphs.view(), w)
    assert set(ls) == {"dsm", "feas"} and torch.isfinite(ls["dsm"]) and float(ls["feas"].detach()) >= 0
    l_mean = diff.mean_loss(ctx, graphs.view(), w, n_members=2, steps=2, gen=torch.Generator().manual_seed(1))
    (ls["dsm"] + ls["feas"] + l_mean).backward()
    assert torch.isfinite(l_mean) and any(p.grad is not None and torch.isfinite(p.grad).all() for p in den.parameters())
    res = diff.sample(ctx, graphs.view(), n_members=2, steps=3, seed=0)
    members = diff.assemble(ctx["xbar"], res, w)
    for m in members:
        v = proj.violations(m, ctx["xbar"], w)
        assert float(v["neg_q_fraction"].max()) == 0.0 and float(v["mass_drift_pa"].abs().max()) < 1e-2
    # state dict round trip keeps the projection (eval reconstructs it from the keys)
    sd = diff.state_dict()
    assert "project.s_mean" in sd and bool(sd["project.water"])
