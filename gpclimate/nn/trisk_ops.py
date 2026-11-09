"""stage22: TRiSK / MPAS C-grid quantities of the flow as fixed guide inputs (감자 2026-11-09 16:00 KST, "MPAS처럼
기후에 대한 보르노이와 들로네도 참고").

MPAS (Skamarock et al. 2012) integrates the vector-invariant momentum equation on a Voronoi (cell) / Delaunay (vertex)
dual with the C-grid staggering of Ringler, Thuburn, Klemp and Skamarock (2010): mass and tracers at the Voronoi
cells, the *normal* velocity u_e at the edges (along the Delaunay edge joining two cell centres), vorticity and
potential vorticity at the vertices (the Delaunay triangles),

    du_e/dt = -(q h u_perp)_e - grad_e (K + Phi),     K_i = (1/A_i) sum_e (l_e d_e / 4) u_e^2,     q_v = (zeta_v + f_v) / h_v,

with d_e the cell-centre distance (dcEdge), l_e the Voronoi edge length (dvEdge), A_i the cell area (the sign of the
PV-flux term depends on the tangential convention; with t = k x n, the counter-clockwise tangent, the normal
component of -eta k x u is +eta u_t, which is what ``pvflux`` holds).  Here the
guide's finest mesh (icosahedral nodes = Voronoi cells, triangles = dual vertices, ``MeshComplex``) plays the MPAS
mesh and the same quantities are computed from the state as *features* (no parameters, exactly SO(2) equivariant):

scalars
* ``ke{k}``       TRiSK kinetic energy of wind pair k from the edge normal components (/ U_REF^2)
* ``pv500``       shallow-water potential vorticity of the 500 hPa pair at the cells (vertex PV averaged with the
                  triangle areas), equivalent depth h = Phi_500 / g (x H_REF / 2 Omega)
* ``div_hu500``   divergence of the 500 hPa mass flux  div(h u)  (/ H_REF 1e-6 s^-1)
* ``adv_t850``, ``adv_q700``  edge-flux (Delaunay) advective tendencies -div(u T) of the 850 hPa temperature (850 hPa
                  wind) and 700 hPa humidity (mean of the 850 / 500 hPa winds), normalised tracer units / 1e-5 s^-1

vectors (frequency 1; edge normal values lifted to the nodes with the least-squares ``sharp``)
* ``pvflux500``   the nonlinear Coriolis / PV-flux term, normal component of -eta k x u (= +q h u_t)
* ``gradB500``    grad_e (K + Phi), the Bernoulli-function gradient
* ``vi_tend500``  pvflux - gradB: the vector-invariant momentum tendency du/dt the balance leaves (0 for a steady
                  state; Williamson test case 2 in tests/test_trisk_ops.py)
* ``gradK{k}``    grad_e K for the other wind pairs (their geopotential is not in the state)

MPAS reconstructs u_perp from the normal components with the Thuburn (2009) weights because it stores u_e only; the
state here carries the full cell-centred wind, so the tangential component is read directly (the energy-conserving
property of those weights matters for a prognostic core, not for input features).  All outputs are divided by
reference scales (2 Omega, U_REF = 10 m/s, H_REF = 5500 m) so that they are O(1).
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from gpclimate.data.variables import VariableSpec
from gpclimate.geometry.connection import MeshComplex
from gpclimate.nn.physics_ops import EARTH_RADIUS, OMEGA
from gpclimate.nn.so2 import Field

G0 = 9.80665
U_REF, H_REF, DIV_REF, ADV_REF = 10.0, 5500.0, 1e-6, 1e-5
PV_PAIR, PV_SCALAR = "u500", "z500"
ADVECTED = (("t850", ("u850",)), ("q700", ("u850", "u500")))


def edge_faces(cx: MeshComplex) -> torch.Tensor:
    """[E, 2] the two triangles (dual vertices) sharing every edge of a closed mesh."""
    idx = cx.face_edges.reshape(-1)
    fid = torch.arange(cx.n_f, device=idx.device).repeat_interleave(3)
    order = torch.argsort(idx, stable=True)
    return fid[order].reshape(cx.n_e, 2)


class TriskOperators(nn.Module):
    def __init__(self, spec: VariableSpec):
        super().__init__()
        scalars, winds = list(spec.scalars), [a for a, _ in spec.vectors]
        need_s = [PV_SCALAR] + [t for t, _ in ADVECTED]
        need_v = [PV_PAIR] + [w for _, ws in ADVECTED for w in ws]
        missing = [n for n in need_s if n not in scalars] + [n for n in need_v if n not in winds]
        if missing:
            raise ValueError(f"TRiSK operators need the base11 channels; missing {missing}")
        self.n_s, self.n_v = spec.n_s, spec.n_v
        self.iz, self.kpv = scalars.index(PV_SCALAR), winds.index(PV_PAIR)
        self.adv = [(scalars.index(t), [winds.index(w) for w in ws]) for t, ws in ADVECTED]
        self.others = [k for k in range(self.n_v) if k != self.kpv]
        self.register_buffer("s_mean", torch.zeros(self.n_s))
        self.register_buffer("s_std", torch.ones(self.n_s))
        self.register_buffer("v_scale", torch.ones(self.n_v))

    def set_scales(self, s_mean: torch.Tensor, s_std: torch.Tensor, v_scale: torch.Tensor) -> None:
        """Physical units of the normalised state: scalars = s * s_std + s_mean, winds = v * v_scale [m/s]."""
        self.s_mean.copy_(s_mean)
        self.s_std.copy_(s_std)
        self.v_scale.copy_(v_scale)

    @property
    def out_dims(self) -> Tuple[int, int]:
        return self.n_v + 2 + len(self.adv), 3 + len(self.others)

    @property
    def scalar_names(self) -> Tuple[str, ...]:
        return tuple([f"ke{k}" for k in range(self.n_v)] + ["pv500", "div_hu500"] + [f"adv_{t}" for t, _ in ADVECTED])

    @property
    def vector_names(self) -> Tuple[str, ...]:
        return tuple(["pvflux500", "gradB500", "vi_tend500"] + [f"gradK{k}" for k in self.others])

    def edge_components(self, U: torch.Tensor, cx: MeshComplex) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normal (along the oriented edge i -> j) and tangential (90 deg counter-clockwise) components [B, E, C] of a
        node vector field [B, N, C] (complex, node gauges): the mean of the two end points' components."""
        ci = cx.dir_i.conj()[None, :, None] * U.index_select(1, cx.edges[:, 0])
        cj = cx.dir_j.conj()[None, :, None] * U.index_select(1, cx.edges[:, 1])
        return 0.5 * (ci.real - cj.real), 0.5 * (ci.imag - cj.imag)

    def forward(self, s: torch.Tensor, v: torch.Tensor, sinlat: torch.Tensor, cx: MeshComplex) -> Field:
        """s [B, N, n_s] normalised scalars and v [B, N, n_v] normalised winds (complex, node gauges) on the mesh
        level of ``cx``; sinlat [N]."""
        B, N = s.shape[:2]
        src, dst = cx.edges[:, 0], cx.edges[:, 1]
        L = cx.h * EARTH_RADIUS  # metres per mesh unit
        U = v * self.v_scale.to(v.real.dtype)
        phi = s[..., self.iz] * self.s_std[self.iz] + self.s_mean[self.iz]  # geopotential [m^2/s^2]
        h = phi / G0  # equivalent depth [m]
        u_n, u_t = self.edge_components(U, cx)  # [B, E, n_v] m/s
        d_e, l_e = cx.length * L, cx.s1 * cx.length * L
        A_i = cx.s0 * L ** 2
        w = cx.flat(U)  # 1-form, mesh units x m/s (= length u_n)
        # vorticity at the triangles (dual vertices): circulation / area; absolute vorticity with f at the triangle
        zeta_f = cx.d1(w) / (cx.face_area * L)[None, :, None]
        f_f = 2 * OMEGA * sinlat.to(zeta_f.dtype).index_select(0, cx.faces.reshape(-1)).reshape(-1, 3).mean(1)
        h_f = h.index_select(1, cx.faces.reshape(-1)).reshape(B, -1, 3).mean(2)
        q_f = (zeta_f[..., self.kpv] + f_f[None]) / h_f  # [B, F] potential vorticity 1/(m s)
        ef = edge_faces(cx)
        q_e = 0.5 * (q_f.index_select(1, ef[:, 0]) + q_f.index_select(1, ef[:, 1]))
        h_e = 0.5 * (h.index_select(1, src) + h.index_select(1, dst))
        # normal component of -(eta k x u) = +eta u_t with t = k x n (so du_n/dt = pvflux - grad_n B)  [m/s^2]
        pvflux_e = q_e * h_e * u_t[..., self.kpv]
        # TRiSK kinetic energy at the cells from the edge normal components
        c = (l_e * d_e / 4)[None, :, None] * u_n ** 2
        K = (torch.zeros(B, N, self.n_v, dtype=c.dtype, device=c.device).index_add_(1, src, c).index_add_(1, dst, c)
             / A_i[None, :, None])
        gradB_e = cx.d0((K[..., self.kpv] + phi)[..., None])[..., 0] / d_e
        vi_e = pvflux_e - gradB_e
        gradK_e = cx.d0(K[..., self.others]) / d_e[None, :, None]
        edge_vals = torch.cat([torch.stack([pvflux_e, gradB_e, vi_e], -1), gradK_e], -1)
        vec = cx.sharp(edge_vals * cx.length[None, :, None]) / (2 * OMEGA * U_REF)
        # cell PV = triangle-area weighted mean of the vertex PV (face_to_vertex divides by the incident area)
        pv_i = cx.face_to_vertex((q_f * cx.face_area[None])[..., None]) * (H_REF / (2 * OMEGA))
        div_hu = -cx.delta1(h_e[..., None] * w[..., self.kpv:self.kpv + 1]) / (L * H_REF * DIV_REF)
        advs = []
        for it, ks in self.adv:  # -div(u T) with the centred edge value of the (normalised) tracer
            t_e = 0.5 * (s[..., it].index_select(1, src) + s[..., it].index_select(1, dst))
            w_k = w[..., ks].mean(-1)
            advs.append(cx.delta1((t_e * w_k)[..., None]) / (L * ADV_REF))
        scal = torch.cat([K / U_REF ** 2, pv_i, div_hu] + advs, -1)
        return Field(scal, vec)
