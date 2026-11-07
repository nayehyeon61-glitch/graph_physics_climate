"""stage11: connection and curvature on the simplicial complex -- exact cochain identities, gauge invariance /
equivariance of flat / sharp / curvature / divergence / potentials, physical checks on Killing fields (solid-body
rotation: curvature = 2 sin(lat), zero divergence, streamfunction -sin(lat)), gradient fields (zero curvature,
divergence = Laplacian), the Lie bracket of two rotation generators ([xi_z, xi_x] = -xi_y) and the guide with the
connection / curvature operators switched on."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpclimate.data.variables import VariableSpec  # noqa: E402
from gpclimate.geometry.graphs import SphereGraphs  # noqa: E402
from gpclimate.models.lie_physical_gnn import GuideConfig, LiePhysicalGNN  # noqa: E402
from gpclimate.nn.connection_ops import ConnectionCurvatureOperators  # noqa: E402
from gpclimate.nn.so2 import Field  # noqa: E402
from gpclimate.training.losses import random_gauges, relative_error  # noqa: E402

torch.manual_seed(0)


@pytest.fixture(scope="module")
def graphs():
    lat = np.linspace(-87.1875, 87.1875, 32)
    lon = np.arange(64) * 5.625
    return SphereGraphs(lat, lon, mesh_splits=(4, 3, 2))


def _frame_coords(nodes, vec3):
    """3-D tangent vectors at the nodes -> complex gauge coordinates (e1 + i e2)."""
    return torch.as_tensor(np.sum(vec3 * nodes.e1, -1) + 1j * np.sum(vec3 * nodes.e2, -1)).to(torch.complex64)


def _killing(nodes, axis):
    """xi_a(r) = a x r, the velocity of a solid-body rotation about ``axis`` (unit sphere, rate 1)."""
    return np.cross(np.asarray(axis, float)[None, :], nodes.pos)


def _rel_rms(a, b):
    return float(torch.sqrt(torch.mean((a - b) ** 2) / torch.mean(b ** 2)))


def _rel(a, b):
    """Relative error of (possibly complex) tensors."""
    return float(torch.sqrt(((a - b).abs() ** 2).sum() / (b.abs() ** 2).sum()))


def test_cochain_identities_and_geometric_holonomy(graphs):
    for l, cx in enumerate(graphs.complexes):
        s = torch.randn(2, cx.n_v, 3)
        assert torch.abs(cx.d1(cx.d0(s))).max() < 1e-5  # d d = 0 exactly (curl grad = 0)
        w = torch.randn(2, cx.n_e, 3)
        # <d0 s, w>_1 = <s, d0^T w>_0 (adjoint)
        assert torch.allclose((cx.d0(s) * w).sum(), (s * cx.d0T(w)).sum(), rtol=1e-4, atol=1e-3)
        c = torch.randn(2, cx.n_f, 3)
        assert torch.allclose((cx.d1(w) * c).sum(), (w * cx.d1T(c)).sum(), rtol=1e-4, atol=1e-3)
        # geometric connection: holonomy around every face = face area (Gauss-Bonnet), total 4 pi
        hol = cx.holonomy.abs() / cx.h ** 2  # face areas were divided by h^2
        assert torch.allclose(hol, cx.face_area, atol=2e-5 / cx.h ** 2)
        assert abs(float(cx.holonomy.abs().sum()) - 4 * np.pi) < 1e-3
        assert abs(float(cx.face_area.sum()) * cx.h ** 2 - 4 * np.pi) < 1e-4  # float32 sums


def test_solid_body_rotation_curvature_divergence_and_streamfunction(graphs):
    """u = z x r: relative vorticity 2 sin(lat), divergence 0, streamfunction -sin(lat); flat/sharp round trip."""
    nodes, cx = graphs.mesh[0], graphs.complexes[0]
    u = _frame_coords(nodes, _killing(nodes, (0, 0, 1)))[None, :, None]  # [1, V, 1]
    sinlat = torch.as_tensor(np.sin(np.deg2rad(nodes.lat)), dtype=torch.float32)[None, :, None]
    zeta = cx.vorticity(u) / cx.h  # normalised units -> per unit length
    assert _rel_rms(zeta, 2 * sinlat) < 0.05
    div = cx.divergence(u) / cx.h
    assert float(div.abs().mean()) < 0.05 * float(zeta.abs().mean())
    psi, chi = cx.potentials(u)
    psi, chi = psi * cx.h, chi * cx.h
    assert _rel_rms(psi - psi.mean(), -(sinlat - sinlat.mean())) < 0.05
    assert float(chi.std()) < 0.05 * float(psi.std())
    # the vertex field is recovered from its edge 1-form
    assert _rel(cx.sharp(cx.flat(u)), u) < 0.03
    # the rotational wind reconstructed from psi points along u again (u_psi = i grad psi)
    from gpclimate.nn.physics_ops import Stencil
    st = Stencil(graphs.mesh_edges[0])
    u_psi = 1j * st.grad(cx.potentials(u)[0])
    assert _rel(u_psi, u) < 0.1


def test_gradient_field_has_no_curvature(graphs):
    """u = grad sin(lat) = cos(lat) north: circulation ~ 0, divergence = Lap sin(lat) = -2 sin(lat)."""
    nodes, cx = graphs.mesh[0], graphs.complexes[0]
    lat = np.deg2rad(nodes.lat)
    u = torch.as_tensor(1j * np.cos(lat)).to(torch.complex64)[None, :, None]  # north = imaginary axis
    sinlat = torch.as_tensor(np.sin(lat), dtype=torch.float32)[None, :, None]
    div = cx.divergence(u) / cx.h
    assert _rel_rms(div, -2 * sinlat) < 0.06
    zeta = cx.vorticity(u) / cx.h
    assert float(zeta.abs().mean()) < 0.05 * float(div.abs().mean())
    # built from a potential, the 1-form d0 f has exactly zero curvature
    assert torch.abs(cx.d1(cx.d0(sinlat))).max() < 1e-6


def test_lie_bracket_of_rotation_generators(graphs):
    """[xi_z, xi_x] = -(z x x) x r = -xi_y  (commutator of the two flow generators)."""
    nodes = graphs.mesh[0]
    ops = ConnectionCurvatureOperators(2)
    u = torch.stack([_frame_coords(nodes, _killing(nodes, (0, 0, 1))),
                     _frame_coords(nodes, _killing(nodes, (1, 0, 0)))], -1)[None]  # [1, V, 2]
    sinlat = torch.as_tensor(np.sin(np.deg2rad(nodes.lat)), dtype=torch.float32)
    out = ops(u, sinlat, graphs.mesh_edges[0], graphs.complexes[0])
    assert out.s.shape[-1] == 14 and out.v.shape[-1] == 5 and ops.vector_names[-1] == "bracket01"
    cx = graphs.complexes[0]
    bracket = out.v[..., -1] / cx.h  # stencil lengths are in units of h
    expected = _frame_coords(nodes, -_killing(nodes, (0, 1, 0)))[None]
    assert _rel(bracket, expected) < 0.1
    # Killing fields are divergence-free and strain-free: W = -zeta^2 < 0 (rotation dominated) at every vertex
    i = ops.scalar_names.index("okubo_weiss0")
    assert float((out.s[..., i] <= 1e-3).float().mean()) > 0.99
    j = ops.scalar_names.index("strain0")
    assert float(out.s[..., j].mean()) < 0.1 * float(out.s[..., ops.scalar_names.index("zeta_dec0")].abs().mean())


def test_connection_operators_are_gauge_equivariant(graphs):
    """Invariants unchanged, vectors rotated, under random per-node gauge rotations (also of the complex)."""
    nodes = graphs.mesh[0]
    ops = ConnectionCurvatureOperators(3)
    v = torch.complex(torch.randn(2, nodes.n, 3), torch.randn(2, nodes.n, 3))
    sinlat = torch.as_tensor(np.sin(np.deg2rad(nodes.lat)), dtype=torch.float32)
    gen = torch.Generator().manual_seed(1)
    _, tm = random_gauges(graphs, gen)
    g0, g1 = graphs.view(), graphs.gauge_transformed(tm)
    out0 = ops(v, sinlat, g0.mesh_edges[0], g0.extras["complexes"][0])
    out1 = ops(Field(v.real[..., :0], v).gauge(tm[0]).v, sinlat, g1.mesh_edges[0], g1.extras["complexes"][0])
    assert _rel(out1.s, out0.s) < 2e-5
    assert _rel(out1.v, Field(out0.s, out0.v).gauge(tm[0]).v) < 2e-5
    cx0, cx1 = g0.extras["complexes"][0], g1.extras["complexes"][0]
    assert _rel(cx1.flat(Field(v.real[..., :0], v).gauge(tm[0]).v), cx0.flat(v)) < 2e-5


def test_guide_with_curvature_ops_runs_and_is_equivariant():
    spec = VariableSpec(scalars=("z500", "t850"), vectors=(("u850", "v850"), ("u10", "v10")),
                        static_scalars=("orog",), static_vectors=(("dox", "doy"),), forcing_scalars=("toa",))
    gr = SphereGraphs(np.linspace(-87.1875, 87.1875, 16), np.arange(0, 360, 11.25), mesh_splits=(2, 1, 0))
    cfg = GuideConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), hidden=16, physics_ops=True, tend_head=True,
                      curvature_ops=True)
    m = LiePhysicalGNN(spec, cfg).eval()
    assert m.curv_ops is not None and m.curv_ops.out_dims == (14, 5)
    x = Field(torch.randn(2, gr.grid.n, 2), torch.complex(torch.randn(2, gr.grid.n, 2), torch.randn(2, gr.grid.n, 2)))
    c = Field(torch.randn(1, gr.grid.n, 2), torch.complex(torch.randn(1, gr.grid.n, 1), torch.randn(1, gr.grid.n, 1)))
    gen = torch.Generator().manual_seed(2)
    tg, tm = random_gauges(gr, gen)
    with torch.no_grad():
        out0 = m(x, c, torch.zeros(2), gr.view())
        out1 = m(x.gauge(tg), c.gauge(tg), torch.zeros(2), gr.gauge_transformed(tm, tg))
    for k, lvl in (("P0", 0), ("P1", 1), ("P2", 2), ("P1u", 1), ("P0u", 0)):
        assert relative_error(out1[k], out0[k].gauge(tm[lvl])) < 5e-5, k
    assert torch.isfinite(out0["P0"].s).all() and torch.isfinite(out0["P0"].v).all()
