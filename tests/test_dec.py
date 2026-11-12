"""Discrete exterior calculus on the icosahedral complex: exact identities, Betti numbers of S^2, Gauss-Bonnet
holonomy of the SO(2) gauge connection, Helmholtz-Hodge decomposition (homology augmentation of the S6 plan)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpclimate.geometry import dec  # noqa: E402
from gpclimate.geometry.graphs import SphereGraphs  # noqa: E402


@pytest.fixture(scope="module")
def graphs():
    lat = np.linspace(-87.1875, 87.1875, 32)
    lon = np.arange(64) * 5.625
    return SphereGraphs(lat, lon, mesh_splits=(3, 2, 1))


def test_coboundaries_and_betti_numbers_of_the_sphere(graphs):
    for level in range(3):
        faces, nodes = graphs.mesh_faces[level], graphs.mesh[level]
        d0, d1, edges = dec.coboundaries(nodes.n, faces)
        V, E, F = nodes.n, len(edges), len(faces)
        assert V - E + F == 2  # Euler characteristic of S^2
        assert abs(d1 @ d0).max() == 0.0  # d d = 0 exactly
    # ranks on the coarsest level (42 vertices): b0 = 1, b1 = 0 (no harmonic 1-forms), b2 = 1
    faces, nodes = graphs.mesh_faces[2], graphs.mesh[2]
    d0, d1, _ = dec.coboundaries(nodes.n, faces)
    assert dec.betti_numbers(d0, d1) == (1, 0, 1)


def test_gauge_connection_holonomy_is_gauss_bonnet(graphs):
    """Transport angles of the SO(2) connection around every face add up to the face's spherical area
    (discrete curvature), with one orientation sign, and the total is 4*pi."""
    for level in range(3):
        e, faces, nodes = graphs.mesh_edges[level], graphs.mesh_faces[level], graphs.mesh[level]
        hol, area = dec.holonomy(e.src.numpy(), e.dst.numpy(), e.alpha.double().numpy(), faces, nodes.pos)
        sign = np.sign(hol[0])
        assert np.all(np.sign(hol) == sign)
        assert np.max(np.abs(np.abs(hol) - area)) < 2e-5  # float32 alphas
        assert abs(np.abs(hol).sum() - 4 * np.pi) < 1e-3
        assert abs(area.sum() - 4 * np.pi) < 1e-9


def test_hodge_stars_and_helmholtz_decomposition(graphs):
    level = 1  # 162 vertices, 480 edges, 320 faces
    faces, nodes = graphs.mesh_faces[level], graphs.mesh[level]
    d0, d1, edges = dec.coboundaries(nodes.n, faces)
    _, face_edges, _ = dec.oriented_edges(faces)
    stars = dec.hodge_stars(nodes.pos, faces, edges, face_edges, nodes.area)
    assert np.all(stars["s1"] > 0) and abs(stars["face_area"].sum() - 4 * np.pi) < 1e-9
    L0, L1, L2, delta1, delta2 = dec.hodge_laplacians(d0, d1, stars)
    # exact calculus identities: curl grad = 0 and div curl = 0 for arbitrary potentials
    rng = np.random.default_rng(0)
    chi, psi = rng.standard_normal(nodes.n), rng.standard_normal(len(faces))
    assert np.abs(d1 @ (d0 @ chi)).max() < 1e-12
    assert np.abs(delta1 @ (delta2 @ psi)).max() < 1e-9
    # Helmholtz-Hodge: a random 1-form splits into gradient + curl, orthogonal in the s1 inner product, no harmonic
    u = rng.standard_normal(len(edges))
    parts = dec.helmholtz_hodge(u, d0, d1, stars)
    w = stars["s1"]
    assert np.linalg.norm(parts["harmonic"]) < 1e-6 * np.linalg.norm(u)  # b1(S^2) = 0
    assert abs(np.sum(w * parts["grad"] * parts["curl"])) < 1e-8 * np.sum(w * u * u)
    # a pure gradient field has no curl part
    g = d0 @ chi
    pg = dec.helmholtz_hodge(g, d0, d1, stars)
    assert np.linalg.norm(pg["curl"]) < 1e-6 * np.linalg.norm(g)
    assert np.linalg.norm(pg["grad"] - g) < 1e-6 * np.linalg.norm(g)
