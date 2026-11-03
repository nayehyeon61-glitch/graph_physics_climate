"""Fixed connection / curvature operators of the flow for the physics guide (stage11a, "extended graph").

Added next to ``PhysicsOperators`` on the finest mesh (no parameters, exactly SO(2) equivariant):

scalars (invariants), per wind pair k
* ``zeta_dec``   relative vorticity from the exact face circulation (curvature of the flow connection)
* ``div_dec``    divergence from the exact codifferential
* ``zeta_abs``   absolute vorticity (zeta_dec in units of 2 Omega + sin(lat), the planetary curvature term)
* ``strain``     |S| = sqrt((du/dx - dv/dy)^2 + (du/dy + dv/dx)^2), the symmetric traceless part of grad u
* ``okubo_weiss``  W = |S|^2 - zeta^2 (strain- vs rotation-dominated flow; the misalignment of the material frame
                 rotation (zeta/2) and the geometric transport is what W measures)
* ``psi``, ``chi`` Helmholtz potentials (streamfunction, velocity potential) from the mesh Poisson solves -- the
                 non-local part the local message passing cannot form

vectors (frequency 1)
* ``u_psi`` = i grad psi (rotational wind), ``u_chi`` = grad chi (divergent wind) per pair
* ``bracket_kl`` = [u_k, u_l] = (u_k . grad) u_l - (u_l . grad) u_k  for every pair of wind levels -- the Lie
  bracket of the two flow generators (non-commutativity of the layers' transports; thermal-wind shear structure)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from gpclimate.geometry.connection import MeshComplex
from gpclimate.geometry.graphs import EdgeSet
from gpclimate.nn.physics_ops import EARTH_RADIUS, OMEGA, Stencil
from gpclimate.nn.so2 import Field


class ConnectionCurvatureOperators(nn.Module):
    def __init__(self, n_v: int, v_scale_mps: Optional[torch.Tensor] = None):
        super().__init__()
        self.n_v = n_v
        self.register_buffer("v_scale", torch.ones(n_v) if v_scale_mps is None else torch.as_tensor(v_scale_mps).float())
        self.pairs = [(k, l) for k in range(n_v) for l in range(k + 1, n_v)]

    @property
    def out_dims(self) -> Tuple[int, int]:
        return 7 * self.n_v, 2 * self.n_v + len(self.pairs)

    @property
    def scalar_names(self) -> Tuple[str, ...]:
        names = []
        for key in ("zeta_dec", "div_dec", "zeta_abs", "strain", "okubo_weiss", "psi", "chi"):
            names += [f"{key}{k}" for k in range(self.n_v)]
        return tuple(names)

    @property
    def vector_names(self) -> Tuple[str, ...]:
        return tuple([f"u_psi{k}" for k in range(self.n_v)] + [f"u_chi{k}" for k in range(self.n_v)]
                     + [f"bracket{k}{l}" for k, l in self.pairs])

    def forward(self, v: torch.Tensor, sinlat: torch.Tensor, edges: EdgeSet, cx: MeshComplex) -> Field:
        """v: wind pairs [B, N, n_v] complex (normalised units) on the mesh level of ``cx``."""
        st = Stencil(edges)
        B, N = v.shape[:2]
        zeta = cx.vorticity(v)
        div = cx.divergence(v)
        kappa = (self.v_scale / (EARTH_RADIUS * cx.h * 2 * OMEGA)).to(zeta.dtype)
        f = sinlat.to(zeta.dtype)[None, :, None].expand(B, N, 1)
        zeta_abs = zeta * kappa + f
        J = st.jacobian(v)  # [B, N, n_v, 2, 2]
        s_n, s_s = J[..., 0, 0] - J[..., 1, 1], J[..., 0, 1] + J[..., 1, 0]
        zeta_st = J[..., 1, 0] - J[..., 0, 1]
        strain2 = s_n ** 2 + s_s ** 2
        strain = strain2.clamp_min(1e-12).sqrt()
        okubo = strain2 - zeta_st ** 2
        psi, chi = cx.potentials(v)
        u_psi = 1j * st.grad(psi)
        u_chi = st.grad(chi)
        ur = torch.view_as_real(v)  # [B, N, n_v, 2]
        brackets = []
        for k, l in self.pairs:
            adv_kl = torch.einsum("bnij,bnj->bni", J[..., l, :, :], ur[..., k, :])  # (u_k . grad) u_l
            adv_lk = torch.einsum("bnij,bnj->bni", J[..., k, :, :], ur[..., l, :])
            brackets.append(torch.view_as_complex((adv_kl - adv_lk).contiguous()))
        vec = [u_psi, u_chi] + ([torch.stack(brackets, -1)] if brackets else [])
        scal = torch.cat([zeta, div, zeta_abs, strain, okubo, psi, chi], -1)
        return Field(scal, torch.cat(vec, -1))
