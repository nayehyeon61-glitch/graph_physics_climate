"""stage11b: simplicial vertex-edge-face message passing (gpclimate/nn/simplicial.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpclimate.data.variables import VariableSpec  # noqa: E402
from gpclimate.geometry.graphs import SphereGraphs  # noqa: E402
from gpclimate.models.lie_physical_gnn import GuideConfig, LiePhysicalGNN  # noqa: E402
from gpclimate.nn.simplicial import SimplicialBlock  # noqa: E402
from gpclimate.nn.so2 import Field  # noqa: E402
from gpclimate.training.losses import random_gauges, relative_error  # noqa: E402


@pytest.fixture(scope="module")
def graphs():
    return SphereGraphs(np.linspace(-87.1875, 87.1875, 32), np.arange(0, 360, 5.625), mesh_splits=(4, 3, 2))


def _rel(a, b):
    return float((a - b).abs().pow(2).sum().sqrt() / (b.abs().pow(2).sum().sqrt() + 1e-12))


def _randomise_readout(block: SimplicialBlock, gen: torch.Generator) -> None:
    with torch.no_grad():
        for p in block.v_out.parameters():
            p.copy_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * 0.5)


def _vertex_field(n, cs, cv, gen):
    return Field(torch.randn(2, n, cs, generator=gen),
                 torch.complex(torch.randn(2, n, cv, generator=gen), torch.randn(2, n, cv, generator=gen)))


def test_block_is_identity_at_init(graphs):
    cx = graphs.view().extras["complexes"][0]
    sinlat = torch.as_tensor(np.sin(np.deg2rad(graphs.mesh[0].lat)), dtype=torch.float32)
    gen = torch.Generator().manual_seed(0)
    x = _vertex_field(cx.n_v, 6, 3, gen)
    blk = SimplicialBlock(6, 3, ce=8, cf=8, hidden=16)
    y = blk(x, cx, sinlat)
    assert torch.equal(y.s, x.s) and torch.equal(y.v, x.v)
    assert sum(p.numel() for p in blk.parameters()) > 0


def test_cochain_features_are_gauge_invariant_and_block_equivariant(graphs):
    """Edge and face inputs are gauge invariants; after a random read-out the vertex update is equivariant."""
    gen = torch.Generator().manual_seed(1)
    _, tm = random_gauges(graphs, gen)
    g0, g1 = graphs.view(), graphs.gauge_transformed(tm)
    cx0, cx1 = g0.extras["complexes"][0], g1.extras["complexes"][0]
    sinlat = g0.mesh_sinlat[0]
    x = _vertex_field(cx0.n_v, 6, 3, gen)
    xg = x.gauge(tm[0])
    blk = SimplicialBlock(6, 3, ce=8, cf=8, hidden=16)
    _randomise_readout(blk, gen)
    e0, e1 = blk.edge_features(x, cx0), blk.edge_features(xg, cx1)
    assert _rel(e1, e0) < 2e-5
    h0 = blk.edge_in(e0)
    assert _rel(blk.face_features(blk.edge_in(e1), cx1, sinlat), blk.face_features(h0, cx0, sinlat)) < 2e-5
    with torch.no_grad():
        y0, y1 = blk(x, cx0, sinlat), blk(xg, cx1, sinlat)
    assert _rel(y1.s, y0.s) < 2e-5
    assert _rel(y1.v, y0.gauge(tm[0]).v) < 2e-5
    assert _rel(y0.s, x.s) > 1e-3 and _rel(y0.v, x.v) > 1e-3  # the block actually changed the field


def test_rotational_content_reaches_faces_gradient_does_not(graphs):
    """d1 d0 = 0: a pure-gradient vertex scalar makes zero circulation; a solid-body rotation makes 2 sin(lat)·A."""
    cx = graphs.view().extras["complexes"][0]
    nodes = graphs.mesh[0]
    sinlat = torch.as_tensor(np.sin(np.deg2rad(nodes.lat)), dtype=torch.float32)
    assert float(cx.d1(cx.d0(sinlat[None, :, None])).abs().max()) < 1e-5
    xi = np.cross(np.array([0.0, 0.0, 1.0])[None, :], nodes.pos)  # Killing field of the rotation about z
    v = torch.as_tensor(np.sum(xi * nodes.e1, -1) + 1j * np.sum(xi * nodes.e2, -1)).to(torch.complex64)
    circ = cx.d1(cx.flat(v[None, :, None]))[0, :, 0] / cx.h  # circulation [rad·(1/h)] -> physical units / h^2
    expected = 2.0 * sinlat[cx.faces].mean(1) * cx.face_area  # vorticity 2 sin(lat) times the face area
    assert _rel(circ, expected) < 0.1


def test_guide_with_simplicial_blocks_runs_and_is_equivariant():
    spec = VariableSpec(scalars=("z500", "t850"), vectors=(("u850", "v850"), ("u10", "v10")),
                        static_scalars=("orog",), static_vectors=(("dox", "doy"),), forcing_scalars=("toa",))
    gr = SphereGraphs(np.linspace(-87.1875, 87.1875, 16), np.arange(0, 360, 11.25), mesh_splits=(2, 1, 0))
    cfg = GuideConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), hidden=16, physics_ops=True, tend_head=True, simplicial=2)
    m = LiePhysicalGNN(spec, cfg).eval()
    assert m.simp_in is not None and m.simp_out is not None
    gen = torch.Generator().manual_seed(3)
    _randomise_readout(m.simp_in, gen)
    _randomise_readout(m.simp_out, gen)
    x = Field(torch.randn(2, gr.grid.n, 2), torch.complex(torch.randn(2, gr.grid.n, 2), torch.randn(2, gr.grid.n, 2)))
    c = Field(torch.randn(1, gr.grid.n, 2), torch.complex(torch.randn(1, gr.grid.n, 1), torch.randn(1, gr.grid.n, 1)))
    tg, tm = random_gauges(gr, gen)
    with torch.no_grad():
        out0 = m(x, c, torch.zeros(2), gr.view())
        out1 = m(x.gauge(tg), c.gauge(tg), torch.zeros(2), gr.gauge_transformed(tm, tg))
    for k, lvl in (("P0", 0), ("P1", 1), ("P2", 2), ("P1u", 1), ("P0u", 0)):
        assert relative_error(out1[k], out0[k].gauge(tm[lvl])) < 5e-5, k
    assert torch.isfinite(out0["P0u"].s).all() and torch.isfinite(out0["P0u"].v).all()
    # the config without blocks has fewer parameters and the blocks are small
    n_with = sum(p.numel() for p in m.parameters())
    n_without = sum(p.numel() for p in LiePhysicalGNN(spec, GuideConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), hidden=16,
                                                                          physics_ops=True, tend_head=True)).parameters())
    assert n_with > n_without
