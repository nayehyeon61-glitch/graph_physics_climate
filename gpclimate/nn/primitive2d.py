"""Fixed resolved dynamics of the reduced (single-level, 2-D) primitive equations for the 11-variable ERA5 layout.

Layer-wise physical constraints (docs/design/Layerwise_Physical_Constraints_KO.md, stage S5a): the physics
guide's tendency head no longer predicts the whole 6 h tendency but only the *closure* ``S_theta`` that remains
after the resolved dynamics ``F(X)`` below have been evaluated by fixed, exactly SO(2)-equivariant operators on
the finest mesh (NeuralGCM-style "dynamical core + learned physics"):

    dX/dt = F(X, c) + S_theta(X, c),      F fixed (no parameters),  S_theta learned.

Only z500, t850, q700, t2m, msl and the winds at 850 hPa, 500 hPa and 10 m are available, so there is no vertical
column: ``F`` is the horizontal part of each equation and everything vertical or unresolved (vertical advection,
adiabatic heating ``kappa T omega / p``, friction, radiation, latent heat, ...) is left to the closure.

    u500 :  -(u.grad)u - f k x u - grad Phi500                      (Phi500 = z500 geopotential)
    u850 :  -(u.grad)u - f k x u - grad Phi850,   Phi850 = R_d Tbar ln(msl / 850 hPa), Tbar = (t2m + t850) / 2
    u10  :  -(u.grad)u - f k x u - R_d t2m grad ln msl              (surface pressure-gradient force)
    t850 :  -u850 . grad t850
    t2m  :  -u10 . grad t2m
    q700 :  -u700 . grad q700,   u700 = log-pressure interpolation of u850 and u500
    msl  :  -msl * sum_k w_k (div u_k + u_k . grad ln msl)          (mass continuity, 3-level column quadrature)
    z500 :  0                                                       (no column temperature: closure only)

Every term is evaluated in SI units per second on the mesh (gradients and Jacobians of
``gpclimate.nn.physics_ops.Stencil`` divided by the physical edge length) and converted to the guide's normalised
6 h tendency units.  Each term can be switched off (``terms``) and carries a fixed calibration coefficient
(``coef``, default 1): the instantaneous tendency is compared with a 6 h forward difference on a 5.6 deg grid, and
``scripts/resolved_fraction.py --fit`` measures per variable the least-squares coefficient that makes the fixed term
explain the most tendency variance (the Coriolis and pressure-gradient terms share one coefficient so that their
geostrophic cancellation is kept).  The coefficients are fitted once on the training set before Stage A, never learned.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from gpclimate.data.variables import VariableSpec
from gpclimate.geometry.graphs import EdgeSet, GraphView
from gpclimate.nn.gauge_conv import transport_interpolate
from gpclimate.nn.physics_ops import EARTH_RADIUS, OMEGA, Stencil
from gpclimate.nn.so2 import Field

R_DRY = 287.05  # J kg^-1 K^-1
P850, P700, P500 = 85000.0, 70000.0, 50000.0
DT_DATA = 6 * 3600.0  # seconds per data step
SCALARS = ("z500", "t850", "q700", "t2m", "msl")
VECTORS = (("u850", "v850"), ("u500", "v500"), ("u10", "v10"))
DEFAULT_TERMS = ("adv_u", "coriolis", "pgf", "adv_t850", "adv_t2m", "adv_q700", "mass")


def _dot(u: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """u . g for complex 2-vectors (gauge invariant)."""
    return (u.conj() * g).real


class ReducedPrimitiveTendencies(nn.Module):
    """F(X, c): fixed resolved tendencies of the reduced primitive equations (see module docstring).

    ``mass_weights`` are the sigma-layer thicknesses represented by the (850 hPa, 500 hPa, 10 m) winds in the
    column quadrature of the continuity equation.
    """

    def __init__(self, spec: VariableSpec, terms: Tuple[str, ...] = DEFAULT_TERMS,
                 mass_weights: Tuple[float, float, float] = (0.30, 0.65, 0.05)):
        super().__init__()
        if tuple(spec.scalars) != SCALARS or tuple(spec.vectors) != VECTORS:
            raise ValueError("ReducedPrimitiveTendencies is written for the default 11-variable layout")
        self.spec = spec
        self.terms = tuple(terms)
        unknown = set(self.terms) - set(DEFAULT_TERMS)
        if unknown:
            raise ValueError(f"unknown resolved terms {sorted(unknown)}")
        self.w850, self.w500, self.w10 = mass_weights
        self.w500_at_700 = math.log(P850 / P700) / math.log(P850 / P500)  # 0.365
        # normalisation of the state and of the guide's tendency target (copied from the Normalizer by the trainer)
        n_s, n_v = spec.n_s, spec.n_v
        self.register_buffer("s_mean", torch.zeros(n_s))
        self.register_buffer("s_std", torch.ones(n_s))
        self.register_buffer("v_scale", torch.ones(n_v))
        self.register_buffer("tend_s", torch.ones(n_s))
        self.register_buffer("tend_v", torch.ones(n_v))
        # fixed calibration coefficients (see module docstring): one per scalar term, and per wind level for the
        # self-advection and for the (Coriolis + pressure gradient) pair
        self.register_buffer("coef_s", torch.ones(n_s))  # z500 (unused), t850, q700, t2m, msl
        self.register_buffer("coef_adv_u", torch.ones(n_v))
        self.register_buffer("coef_ageo", torch.ones(n_v))

    def set_coefficients(self, coef: Optional[Dict[str, list]]) -> None:
        """``{"coef_s": [5], "coef_adv_u": [3], "coef_ageo": [3]}`` as written by scripts/resolved_fraction.py --fit."""
        if not coef:
            return
        for k in ("coef_s", "coef_adv_u", "coef_ageo"):
            if k in coef:
                getattr(self, k).copy_(torch.as_tensor(coef[k], dtype=torch.float32))

    def set_scales(self, s_mean, s_std, v_scale, tend_s, tend_v) -> None:
        for name, val in (("s_mean", s_mean), ("s_std", s_std), ("v_scale", v_scale), ("tend_s", tend_s),
                          ("tend_v", tend_v)):
            getattr(self, name).copy_(torch.as_tensor(val, dtype=torch.float32))

    def on(self, name: str) -> float:
        return 1.0 if name in self.terms else 0.0

    # ------------------------------------------------------------------ physical units, on a mesh level
    def physical(self, s: torch.Tensor, v: torch.Tensor, sinlat: torch.Tensor, edges: EdgeSet,
                 return_terms: bool = False):
        """s [B, N, 5] physical scalars, v [B, N, 3] complex winds (m/s) -> Field of tendencies per second."""
        st = Stencil(edges)
        L = st.h * EARTH_RADIUS  # metres per mesh length unit
        B, N = s.shape[:2]
        f = (2 * OMEGA) * sinlat.to(s.dtype)[None, :, None].expand(B, N, 1)
        z500, t850, q700, t2m, msl = s.unbind(-1)
        lnp = torch.log(msl.clamp(min=1e3))
        tbar = 0.5 * (t2m + t850)
        phi850 = R_DRY * tbar * (lnp - math.log(P850))
        grads = st.grad(torch.stack([z500, t850, q700, t2m, lnp, phi850], -1)) / L  # complex [B, N, 6], per metre
        g_z500, g_t850, g_q700, g_t2m, g_lnp, g_phi850 = grads.unbind(-1)
        J = st.jacobian(v) / L  # [B, N, 3, 2, 2] per second
        self_adv = torch.view_as_complex(torch.einsum("bncij,bncj->bnci", J, torch.view_as_real(v)).contiguous())
        div = J[..., 0, 0] + J[..., 1, 1]  # [B, N, 3]
        cor = 1j * f * v  # f k x u
        u850, u500, u10 = v.unbind(-1)
        pgf = torch.stack([g_phi850, g_z500, R_DRY * t2m * g_lnp], -1)  # pressure-gradient force (sign below)
        ca, cg = self.coef_adv_u.to(s.dtype), self.coef_ageo.to(s.dtype)
        F_u = -(self.on("adv_u") * ca * self_adv + cg * (self.on("coriolis") * cor + self.on("pgf") * pgf))
        u700 = (1 - self.w500_at_700) * u850 + self.w500_at_700 * u500
        c1, c2, c3, c4 = (self.coef_s[i].to(s.dtype) for i in (1, 2, 3, 4))
        F_t850 = -self.on("adv_t850") * c1 * _dot(u850, g_t850)
        F_t2m = -self.on("adv_t2m") * c3 * _dot(u10, g_t2m)
        F_q700 = -self.on("adv_q700") * c2 * _dot(u700, g_q700)
        col = (self.w850 * (div[..., 0] + _dot(u850, g_lnp)) + self.w500 * (div[..., 1] + _dot(u500, g_lnp))
               + self.w10 * (div[..., 2] + _dot(u10, g_lnp)))
        F_msl = -self.on("mass") * c4 * msl * col
        F = Field(torch.stack([torch.zeros_like(z500), F_t850, F_q700, F_t2m, F_msl], -1), F_u)
        if not return_terms:
            return F
        terms = {"adv_u": -self_adv, "coriolis": -cor, "pgf": -pgf, "ageo": -(cor + pgf),
                 "adv_t850": -_dot(u850, g_t850), "adv_t2m": -_dot(u10, g_t2m), "adv_q700": -_dot(u700, g_q700),
                 "mass": -msl * col}
        return F, terms

    # ------------------------------------------------------------------ normalised 6 h tendency, on mesh 0
    def to_normalised(self, F: Field) -> Field:
        return Field(F.s * DT_DATA / (self.s_std * self.tend_s), F.v * DT_DATA / (self.v_scale * self.tend_v))

    @staticmethod
    def pool(grid_field: Field, g: GraphView) -> Field:
        """Grid -> mesh 0 by bilinear sampling with transport (falls back to the Gaussian pooling of old graphs)."""
        return transport_interpolate(grid_field, g.extras.get("grid2mesh_lin", g.grid2mesh))

    def mesh_forward(self, state_norm: Field, g: GraphView) -> Field:
        """Normalised grid state -> resolved part of the normalised 6 h tendency on mesh 0 (exactly equivariant).

        The closure head of the guide and the PDE-residual loss live on mesh 0 as well, so no mesh-to-grid
        interpolation (an inverse-distance average that is ~9% noisy for gradient fields) enters the constraint.
        """
        phys = Field(state_norm.s * self.s_std + self.s_mean, state_norm.v * self.v_scale)
        m = self.pool(phys, g)
        return self.to_normalised(self.physical(m.s, m.v, g.mesh_sinlat[0], g.mesh_edges[0]))

    def forward(self, state_norm: Field, g: GraphView) -> Field:
        """Same as ``mesh_forward`` but interpolated back to the grid (diagnostics only)."""
        return transport_interpolate(self.mesh_forward(state_norm, g), g.mesh2grid)

    def normalised_terms(self, state_norm: Field, g: GraphView) -> Dict[str, Field]:
        """Each resolved term separately as a normalised mesh-0 tendency (for the resolved-fraction diagnostic)."""
        phys = Field(state_norm.s * self.s_std + self.s_mean, state_norm.v * self.v_scale)
        m = self.pool(phys, g)
        _, terms = self.physical(m.s, m.v, g.mesh_sinlat[0], g.mesh_edges[0], return_terms=True)
        B, N = m.s.shape[:2]
        zs = m.s.new_zeros(B, N)
        zv = torch.zeros(B, N, 3, dtype=m.v.dtype, device=m.v.device)
        out = {}
        for name, t in terms.items():
            if name in ("adv_u", "coriolis", "pgf", "ageo"):
                F = Field(torch.stack([zs] * 5, -1), t)
            else:
                col = {"adv_t850": 1, "adv_q700": 2, "adv_t2m": 3, "mass": 4}[name]
                s_list = [zs] * 5
                s_list[col] = t
                F = Field(torch.stack(s_list, -1), zv)
            out[name] = self.to_normalised(F)
        return out


def resolved_fraction(target: Field, resolved: Field, area_w: torch.Tensor) -> Dict[str, Optional[float]]:
    """Per-channel var(target - resolved) / var(target): < 1 means the resolved dynamics explain tendency variance."""
    w = area_w[None, :, None]

    def var(x):
        return (x.abs() ** 2 * w).sum((0, 1)) / w.expand_as(x.real).sum((0, 1))

    rs = (var(target.s - resolved.s) / var(target.s).clamp(min=1e-12)).tolist()
    rv = (var(target.v - resolved.v) / var(target.v).clamp(min=1e-12)).tolist()
    names = list(SCALARS) + [a[1:] for a, _ in VECTORS]
    return dict(zip(names, rs + rv))
