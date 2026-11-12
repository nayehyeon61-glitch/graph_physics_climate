"""stage19 GPINN: physics residuals of the relations that actually hold for ERA5 at 5.6 deg / 6 h.

``scripts/pinn_equation_check.py`` measured, on mesh 0 of the training years, var(L - R) / var(L) of candidate
equations (extratropics):  geostrophic balance at 500 hPa 0.03 and at 850 hPa 0.09, the balanced 6 h change at
500 hPa 0.30, thermal wind 0.31 (calibrated), hypsometric 6 h change 0.35 (calibrated, also in the tropics) and the
500 hPa barotropic vorticity equation with the midpoint rate 0.39 -- while the reduced momentum equation fails (1.97),
which is why the hard-coded dynamics of stage5a weakened the guide.  Only the relations that hold are used here, as
soft PINN losses (Raissi et al. 2019; GPINN, arXiv 2306.09792): the mean squared residual of each relation over its
region, divided by var(L) of ERA5 (``fit_scales``), so a forecast that is as balanced as the analysis scores about the
ratio above and an unbalanced one scores up to O(1).

State relations (one state X):            geo500, geo850, thermal
Change relations (X0 -> X1 over 6 h):     geo500_tend, hyps_tend, bve500 (midpoint rate)

Everything runs on mesh 0 in physical units with the equivariant stencil of ``gpclimate.nn.physics_ops`` after the
transport pooling of the grid state, so residual magnitudes are gauge invariant (random gauge augmentation is fine).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn

from gpclimate.geometry.graphs import GraphView
from gpclimate.nn.gauge_conv import transport_interpolate
from gpclimate.nn.physics_ops import EARTH_RADIUS, OMEGA, Stencil
from gpclimate.nn.so2 import Field

R_DRY = 287.05
P850, P700, P500 = 85000.0, 70000.0, 50000.0
LN85 = math.log(P850 / P500)
DT = 6 * 3600.0
STATE_TERMS = ("geo500", "geo850", "thermal")
CHANGE_TERMS = ("geo500_tend", "hyps_tend", "bve500")
ALL_TERMS = STATE_TERMS + CHANGE_TERMS
# least-squares calibration of the relations with an approximate layer temperature (pinn_equation_check, global);
# defaults only: ``PINNResiduals.fit_scales`` re-fits them on the run's own training pairs (preprocessing step)
ALPHA = {"thermal": 0.77, "hyps_tend": 0.76}
CALIBRATED = tuple(ALPHA)
GLOBAL_TERMS = ("hyps_tend",)  # holds in the tropics as well; the balance / vorticity relations use |lat| > lat_min


def physics_fields(m: Field, sinlat: torch.Tensor, st: Stencil) -> Dict[str, torch.Tensor]:
    """Mesh-0 physical state (base11 layout) -> the quantities of the relations (SI units)."""
    L = st.h * EARTH_RADIUS
    z500, t850, q700, t2m, msl = m.s.unbind(-1)
    u850, u500, _ = m.v.unbind(-1)
    f = (2 * OMEGA) * sinlat.to(m.s.dtype)[None, :]
    # balance relations divide by f; near the equator (masked out of those relations) keep it finite
    f_min = 2 * OMEGA * math.sin(math.radians(5.0))
    f_safe = torch.where(f.abs() < f_min, torch.where(f < 0, -f_min, f_min) * torch.ones_like(f), f)
    phi850 = R_DRY * 0.5 * (t2m + t850) * (torch.log(msl.clamp(min=1e3)) - math.log(P850))
    J = st.jacobian(torch.stack([u850, u500], -1)) / L  # [B, N, 2, 2, 2]  d u_i / d x_j
    vort = J[..., 1, 0] - J[..., 0, 1]
    div = J[..., 0, 0] + J[..., 1, 1]
    grads = st.grad(torch.stack([z500, t850, q700, phi850, vort[..., 1] + f.expand_as(z500)], -1)) / L
    g_z500, g_t850, g_q700, g_phi850, g_absv500 = grads.unbind(-1)
    return dict(z500=z500, t850=t850, q700=q700, phi850=phi850, u850=u850, u500=u500, f=f, f_safe=f_safe,
                vort500=vort[..., 1],
                div500=div[..., 1], g_z500=g_z500, g_t850=g_t850, g_q700=g_q700, g_phi850=g_phi850,
                g_absv500=g_absv500)


def state_relations(q: Dict[str, torch.Tensor], alpha: Optional[Dict[str, float]] = None
                    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """(L, R) of the balance relations of one state; L - R is the residual."""
    a = ALPHA if alpha is None else alpha
    f = q["f_safe"]
    return {"geo500": (q["u500"], 1j * q["g_z500"] / f),
            "geo850": (q["u850"], 1j * q["g_phi850"] / f),
            "thermal": (q["u500"] - q["u850"], a["thermal"] * 1j * R_DRY * LN85 * q["g_t850"] / f)}


def change_relations(q0, q1, qm, alpha: Optional[Dict[str, float]] = None) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """(L, R) of the 6 h change relations (``qm`` = fields of the midpoint state)."""
    a = ALPHA if alpha is None else alpha
    f = q0["f_safe"]
    return {"geo500_tend": (q1["u500"] - q0["u500"], 1j * (q1["g_z500"] - q0["g_z500"]) / f),
            "hyps_tend": (q1["z500"] - q0["z500"],
                          a["hyps_tend"] * ((q1["phi850"] - q0["phi850"]) + R_DRY * LN85 * (q1["t850"] - q0["t850"]))),
            "bve500": (q1["vort500"] - q0["vort500"], -DT * (qm["u500"].conj() * qm["g_absv500"]).real)}


class PINNResiduals(nn.Module):
    """Normalised PINN residual losses on mesh 0.  ``scale`` = var(L) of ERA5 per relation (``fit_scales``)."""

    def __init__(self, s_mean, s_std, v_scale, mesh_lat_deg, terms: Sequence[str] = ALL_TERMS, lat_min: float = 20.0):
        super().__init__()
        unknown = set(terms) - set(ALL_TERMS)
        if unknown:
            raise ValueError(f"unknown PINN relations {sorted(unknown)}")
        self.terms = tuple(terms)
        self.register_buffer("s_mean", torch.as_tensor(s_mean, dtype=torch.float32).clone())
        self.register_buffer("s_std", torch.as_tensor(s_std, dtype=torch.float32).clone())
        self.register_buffer("v_scale", torch.as_tensor(v_scale, dtype=torch.float32).clone())
        lat = torch.as_tensor(mesh_lat_deg, dtype=torch.float32).abs()
        self.register_buffer("mask_x", (lat > lat_min).float())
        self.register_buffer("scale", torch.ones(len(ALL_TERMS)))
        self.register_buffer("alpha", torch.tensor([ALPHA[k] for k in CALIBRATED]))
        self._st: Dict[int, Stencil] = {}
        self.last_fit: Dict[str, Dict[str, float]] = {}

    def _stencil(self, g: GraphView) -> Stencil:
        key = id(g.mesh_edges[0])
        if key not in self._st:
            self._st = {key: Stencil(g.mesh_edges[0])}  # one view at a time (Stage A changes the gauge every step)
        return self._st[key]

    def fields(self, x_norm: Field, g: GraphView) -> Dict[str, torch.Tensor]:
        phys = Field(x_norm.s * self.s_std + self.s_mean, x_norm.v * self.v_scale)
        m = transport_interpolate(phys, g.extras.get("grid2mesh_lin", g.grid2mesh))
        return physics_fields(m, g.mesh_sinlat[0], self._stencil(g))

    def alpha_dict(self) -> Dict[str, float]:
        return {k: float(a) for k, a in zip(CALIBRATED, self.alpha)}

    def relations(self, x0: Field, x1: Optional[Field], g: GraphView, calibrated: bool = True
                  ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """(L, R) per relation; ``calibrated=False`` uses alpha = 1 (the raw right-hand side, for fitting alpha)."""
        a = self.alpha_dict() if calibrated else {k: 1.0 for k in CALIBRATED}
        q0 = self.fields(x0, g)
        out = {k: v for k, v in state_relations(q0, a).items() if k in self.terms}
        if x1 is not None and any(t in self.terms for t in CHANGE_TERMS):
            q1 = self.fields(x1, g)
            qm = self.fields(Field(0.5 * (x0.s + x1.s), 0.5 * (x0.v + x1.v)), g)
            out.update({k: v for k, v in change_relations(q0, q1, qm, a).items() if k in self.terms})
        return out

    def _region(self, name: str) -> torch.Tensor:
        return torch.ones_like(self.mask_x) if name in GLOBAL_TERMS else self.mask_x

    def _mean(self, name: str, t: torch.Tensor) -> torch.Tensor:
        m = self._region(name)
        return torch.where(m > 0, t.abs() ** 2, torch.zeros_like(t.real)).sum(-1).mean() / m.sum()

    def _cross(self, name: str, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Region mean of Re(conj(a) b)."""
        m = self._region(name)
        return torch.where(m > 0, (a.conj() * b).real, torch.zeros_like(a.real)).sum(-1).mean() / m.sum()

    def losses(self, x0: Field, x1: Optional[Field], g: GraphView) -> Dict[str, torch.Tensor]:
        """Per relation: mean |L - R|^2 over its region / var(L) of ERA5."""
        return {k: self._mean(k, L - R) / self.scale[ALL_TERMS.index(k)]
                for k, (L, R) in self.relations(x0, x1, g).items()}

    def ratios(self, x0: Field, x1: Optional[Field], g: GraphView) -> Dict[str, float]:
        """var(L - R) / var(L) of these states themselves (the pinn_equation_check number), for logs."""
        return {k: float(self._mean(k, L - R) / self._mean(k, L).clamp_min(1e-30))
                for k, (L, R) in self.relations(x0, x1, g).items()}

    @torch.no_grad()
    def fit_scales(self, pairs, g: GraphView, fit_alpha: bool = True) -> Dict[str, float]:
        """Preprocessing fitted on the run's training data.  ``pairs``: iterable of normalised (X0, X1) truth pairs.
        Sets ``scale`` = var(L) per relation and, with ``fit_alpha``, the calibration of the thermal-wind and
        hypsometric relations by least squares, alpha = <L, R> / <R, R> over the relation's region (clipped to
        [0.3, 1.5]; kept at the default when R vanishes).  ``self.last_fit`` holds alpha and var(L - R)/var(L)."""
        acc = {k: [0.0, 0.0, 0.0, 0.0, 0] for k in self.terms}  # var L, <L,R>, <R,R>, var(L - R) (calibrated), n
        pairs = list(pairs)
        for x0, x1 in pairs:
            for k, (L, R) in self.relations(x0, x1, g, calibrated=False).items():
                a = acc[k]
                a[0] += float(self._mean(k, L))
                a[1] += float(self._cross(k, L, R))
                a[2] += float(self._mean(k, R))
                a[4] += 1
        for k, (vl, lr, rr, _, n) in acc.items():
            if n == 0:
                continue
            self.scale[ALL_TERMS.index(k)] = max(vl / n, 1e-30)
            if fit_alpha and k in CALIBRATED and rr > 1e-12 * max(vl, 1e-30):
                self.alpha[CALIBRATED.index(k)] = min(max(lr / rr, 0.3), 1.5)
        for x0, x1 in pairs:  # the fitted relations' own residual ratio (the pinn_equation_check number)
            for k, (L, R) in self.relations(x0, x1, g).items():
                acc[k][3] += float(self._mean(k, L - R))
        self.last_fit = {"alpha": self.alpha_dict(),
                         "ratio": {k: v[3] / max(v[0], 1e-30) for k, v in acc.items() if v[4]}}
        return {k: float(self.scale[ALL_TERMS.index(k)]) for k in self.terms}
