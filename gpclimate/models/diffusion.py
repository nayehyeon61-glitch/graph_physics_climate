"""Stage C (skeleton) - conditional graph diffusion over the residual of the deterministic forecast.

    R_0 = (X^{ERA5}_{t+dt} - X_bar_{t+dt}) / sigma_R            residual in normalised units
    R_s = sqrt(abar_s) R_0 + sqrt(1 - abar_s) eps
    eps_hat = D_psi(R_s, s | X_bar, C_t),   C_t = (X_t, {P_l}, c_t)
    X^{(k)} = P_C(X_bar + sigma_R * R^{(k)}; X_bar)              ensemble assembly (+ feasibility projection, S5c)

``s`` is the artificial diffusion time and is kept separate from the
atmospheric time ``t`` everywhere (argument names, logs).

Gauge equivariance: the denoiser is built from the same SO(2) layers; vector
noise is isotropic complex Gaussian, whose law is rotation invariant, so the
*distribution* of samples is equivariant (individual samples match only when the
seed noise is rotated as well, design doc section 11.2).

Stage S5c (layer-wise physical constraints, Noise layer): ``FeasibilityProjection`` is the projection P_C onto the
admissible set (q700 >= 0, total mass and total water equal to those of the deterministic forecast); the training
loss adds a feasibility penalty on the denoiser's x0-prediction and an unbiased ensemble-mean penalty on final,
projected samples (design doc docs/design/Layerwise_Physical_Constraints_KO.md, S5c).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import nn

from gpclimate.data.variables import VariableSpec
from gpclimate.geometry.graphs import GraphView
from gpclimate.models.lie_physical_gnn import GridLift, broadcast_batch
from gpclimate.nn.gauge_conv import GaugeBlock, GaugeTransfer, Merge
from gpclimate.nn.so2 import EqLinear, Field
from gpclimate.training.losses import area_mean


@dataclass
class DiffusionConfig:
    cs: int = 48
    cv: int = 12
    depth: int = 4
    n_basis: int = 8
    hidden: int = 64
    t_embed: int = 32
    n_train_steps: int = 1000  # number of diffusion steps s in training
    sample_steps: int = 20  # respaced ancestral (DDIM, eta=1) steps at inference


def timestep_embedding(s: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=s.device, dtype=torch.float32) / half)
    a = s.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


class ResidualDenoiser(nn.Module):
    def __init__(self, spec: VariableSpec, cfg: DiffusionConfig, guide_dims=(48, 12)):
        super().__init__()
        self.spec, self.cfg = spec, cfg
        cs, cv, nb, hd = cfg.cs, cfg.cv, cfg.n_basis, cfg.hidden
        n_in_s = 3 * spec.n_s + spec.n_cond_s  # R_s, X_bar, X_t, c_t
        n_in_v = 3 * spec.n_v + spec.n_cond_v
        self.lift = GridLift(n_in_s, n_in_v, cs, cv, nb, hd)
        pcs, pcv = guide_dims
        self.merge_p = Merge(cs, cv, pcs, pcv, cs, cv)  # inject physics features P0* of the frozen guide
        self.t_mlp = nn.Sequential(nn.Linear(cfg.t_embed, hd), nn.SiLU(), nn.Linear(hd, hd))
        self.t_proj = nn.ModuleList([nn.Linear(hd, cs) for _ in range(cfg.depth)])
        self.blocks = nn.ModuleList([GaugeBlock(cs, cv, nb, hd) for _ in range(cfg.depth)])
        self.dec = GaugeTransfer(cs, cv, cs, cv, nb, hd, extra_cs=cs, extra_cv=cv)
        self.out = EqLinear(cs, cv, spec.n_s, spec.n_v)

    def forward(self, r_s: Field, s: torch.Tensor, ctx: Dict[str, Field], g: GraphView) -> Field:
        B = r_s.s.shape[0]
        cond = broadcast_batch(ctx["cond"], B)
        xb, xt = ctx["xbar"], ctx["x"]
        x = Field(torch.cat([r_s.s, xb.s, xt.s, cond.s], -1), torch.cat([r_s.v, xb.v, xt.v, cond.v], -1))
        grid_emb, m = self.lift(x, g)
        if "P0u" in ctx and ctx["P0u"] is not None:
            m = self.merge_p(m, broadcast_batch(ctx["P0u"], B))
        temb = self.t_mlp(timestep_embedding(s, self.cfg.t_embed))  # [B, hd] invariant
        for blk, proj in zip(self.blocks, self.t_proj):
            m = Field(m.s + proj(temb)[:, None, :], m.v)
            m = blk(m, g.mesh_edges[0])
        return self.out(self.dec(m, g.mesh2grid, extra=grid_emb))


def cosine_alpha_bar(n: int, s0: float = 0.008) -> torch.Tensor:
    t = torch.linspace(0, n, n + 1, dtype=torch.float64) / n
    f = torch.cos((t + s0) / (1 + s0) * math.pi / 2) ** 2
    abar = (f / f[0])[1:]
    return abar.clamp(1e-5, 1.0).float()


class FeasibilityProjection(nn.Module):
    """P_C(mu): projection of a generated state onto the physically admissible set (stage S5c).

    Acts on normalised grid Fields ``[B, G, C]`` (scalars only, so it commutes with gauge changes).  For the
    11-variable layout the admissible set C(mu) is

      q700 >= 0                 clamp at the normalised threshold (0 - mean_q) / std_q
      M(X) = M(mu)   (mass)     msl shifted by one constant so that its area mean equals that of mu.  The total
                                mass is proportional to the area integral of the surface pressure, i.e. linear in
                                msl, so a uniform shift is the exact minimum-norm projection onto M(X) = M(mu).
      W(X) = W(mu)   (water)    q700 rescaled by one positive factor so that the p_s-weighted column water
                                integral  W = int msl * q700 dA  equals that of mu (rescaling keeps q700 >= 0).
      T > 0, msl > 0            never violated at atmospheric values; not enforced.

    No hydrostatic diagnosis of z500 is possible from the layout (one geopotential level, no column), so z500
    stays prognostic.
    """

    def __init__(self, spec: VariableSpec, s_mean: torch.Tensor, s_std: torch.Tensor, water: bool = True):
        super().__init__()
        sc = list(spec.scalars)
        self.iq, self.im = sc.index("q700"), sc.index("msl")
        self.register_buffer("s_mean", s_mean.clone().float())
        self.register_buffer("s_std", s_std.clone().float())
        self.register_buffer("water", torch.tensor(bool(water)))

    @property
    def q_min(self) -> torch.Tensor:
        return -self.s_mean[self.iq] / self.s_std[self.iq]  # q700 = 0 in normalised units

    def forward(self, x: Field, mu: Field, area_w: torch.Tensor) -> Field:
        cols = list(x.s.unbind(-1))
        iq, im = self.iq, self.im
        q = cols[iq].clamp(min=float(self.q_min))
        m = cols[im] - area_mean((cols[im] - mu.s[..., im])[..., None], area_w)[..., None, 0]
        if bool(self.water):
            mq, sq, mm, sm = self.s_mean[iq], self.s_std[iq], self.s_mean[im], self.s_std[im]
            qp, msl = q * sq + mq, m * sm + mm
            w_x = area_mean((msl * qp)[..., None], area_w)[..., 0]  # [B]
            w_mu = area_mean(((mu.s[..., im] * sm + mm) * (mu.s[..., iq] * sq + mq))[..., None], area_w)[..., 0]
            factor = w_mu / w_x.clamp(min=1e-12)
            q = (qp * factor[:, None] - mq) / sq
        cols[iq], cols[im] = q, m
        return Field(torch.stack(cols, -1), x.v)

    @torch.no_grad()
    def violations(self, x: Field, mu: Field, area_w: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Per-sample diagnostics in physical units: negative-q700 area fraction, mass and water drift w.r.t. mu."""
        iq, im = self.iq, self.im
        mq, sq, mm, sm = self.s_mean[iq], self.s_std[iq], self.s_mean[im], self.s_std[im]
        q, msl = x.s[..., iq] * sq + mq, x.s[..., im] * sm + mm
        q_mu, msl_mu = mu.s[..., iq] * sq + mq, mu.s[..., im] * sm + mm
        neg = area_mean((q < 0).float()[..., None], area_w)[..., 0]
        mass = area_mean((msl - msl_mu)[..., None], area_w)[..., 0]  # Pa
        w_x = area_mean((msl * q)[..., None], area_w)[..., 0]
        w_mu = area_mean((msl_mu * q_mu)[..., None], area_w)[..., 0]
        return {"neg_q_fraction": neg, "mass_drift_pa": mass, "water_drift_rel": w_x / w_mu - 1}


class GraphResidualDiffusion(nn.Module):
    def __init__(self, denoiser: ResidualDenoiser, n_train_steps: int = 1000,
                 project: Optional[FeasibilityProjection] = None):
        super().__init__()
        self.denoiser = denoiser
        self.project = project  # stage S5c: P_C applied to every generated state (None = unconstrained, S1..S5b)
        self.n = n_train_steps
        self.register_buffer("abar", cosine_alpha_bar(n_train_steps))
        self.register_buffer("res_scale_s", torch.ones(denoiser.spec.n_s))
        self.register_buffer("res_scale_v", torch.ones(denoiser.spec.n_v))

    @staticmethod
    def _noise_like(f: Field, generator: Optional[torch.Generator] = None) -> Field:
        es = torch.randn(f.s.shape, generator=generator, device=f.s.device, dtype=f.s.dtype)
        er = torch.randn(f.v.shape, generator=generator, device=f.s.device, dtype=f.s.dtype)
        ei = torch.randn(f.v.shape, generator=generator, device=f.s.device, dtype=f.s.dtype)
        return Field(es, torch.complex(er, ei))  # isotropic in each tangent plane

    def to_residual(self, truth: Field, xbar: Field) -> Field:
        return Field((truth.s - xbar.s) / self.res_scale_s, (truth.v - xbar.v) / self.res_scale_v)

    def losses(self, r0: Field, ctx: Dict[str, Field], g: GraphView, area_w: torch.Tensor) -> Dict[str, torch.Tensor]:
        """``dsm``: denoising score matching on the observed residual.  ``feas`` (stage S5c, zero without a
        projection): soft feasibility penalty on the denoiser's x0-prediction, in residual units and weighted by
        abar(s) -- q700 below zero and the mass drift of the implied state X_bar + sigma_R x0_hat."""
        B = r0.s.shape[0]
        s = torch.randint(0, self.n, (B,), device=r0.s.device)
        a = self.abar[s].reshape(B, 1, 1)
        eps = self._noise_like(r0)
        r_s = Field(a.sqrt() * r0.s + (1 - a).sqrt() * eps.s, a.sqrt() * r0.v + (1 - a).sqrt() * eps.v)
        pred = self.denoiser(r_s, s, ctx, g)
        w = area_w[None, :, None]
        num = ((pred.s - eps.s) ** 2 * w).sum() + ((pred.v - eps.v).abs() ** 2 * w).sum()
        den = w.expand_as(pred.s).sum() + 2 * w.expand(B, -1, pred.v.shape[-1]).sum()
        out = {"dsm": num / den}
        if self.project is not None:
            x0_hat = (r_s.s - (1 - a).sqrt() * pred.s) / a.sqrt()
            xbar = ctx["xbar"].s
            state_s = xbar + self.res_scale_s * x0_hat
            iq, im = self.project.iq, self.project.im
            neg_q = torch.relu(float(self.project.q_min) - state_s[..., iq]) / self.res_scale_s[iq]
            pen_q = area_mean((neg_q ** 2)[..., None], area_w)[..., 0]  # [B]
            drift = area_mean((state_s[..., im] - xbar[..., im])[..., None], area_w)[..., 0] / self.res_scale_s[im]
            out["feas"] = (a.reshape(B) * (pen_q + drift ** 2)).mean()
        else:
            out["feas"] = torch.zeros((), device=r0.s.device)
        return out

    def loss(self, r0: Field, ctx: Dict[str, Field], g: GraphView, area_w: torch.Tensor) -> torch.Tensor:
        return self.losses(r0, ctx, g, area_w)["dsm"]

    def _sample_residuals(self, ctx: Dict[str, Field], g: GraphView, n_members: int, steps: int,
                          gen: torch.Generator) -> List[Field]:
        ts = torch.linspace(self.n - 1, 0, steps).round().long().tolist()
        members = []
        for _ in range(n_members):
            x = self._noise_like(ctx["xbar"], gen)
            B = x.s.shape[0]
            for k, t in enumerate(ts):
                a_t = self.abar[t]
                a_prev = self.abar[ts[k + 1]] if k + 1 < len(ts) else torch.tensor(1.0, device=a_t.device)
                eps = self.denoiser(x, torch.full((B,), t, device=x.s.device), ctx, g)
                x0 = Field((x.s - (1 - a_t).sqrt() * eps.s) / a_t.sqrt(), (x.v - (1 - a_t).sqrt() * eps.v) / a_t.sqrt())
                mag = x0.v.abs()
                x0 = Field(x0.s.clamp(-10, 10), x0.v * (mag.clamp(max=10.0) / mag.clamp(min=1e-12)))
                if k + 1 == len(ts):
                    x = x0
                    break
                # DDIM with eta = 1 (ancestral)
                sig = ((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)).clamp(min=0).sqrt()
                c = (1 - a_prev - sig ** 2).clamp(min=0).sqrt()
                z = self._noise_like(x, gen)
                x = Field(a_prev.sqrt() * x0.s + c * eps.s + sig * z.s, a_prev.sqrt() * x0.v + c * eps.v + sig * z.v)
            members.append(Field(x.s * self.res_scale_s, x.v * self.res_scale_v))
        return members

    def assemble(self, xbar: Field, residuals: List[Field], area_w: torch.Tensor, project: bool = True) -> List[Field]:
        """X^{(k)} = P_C(X_bar + R^{(k)}; X_bar) -- the projection is the identity for unconstrained runs."""
        members = assemble_ensemble(xbar, residuals)
        if project and self.project is not None:
            members = [self.project(m, xbar, area_w) for m in members]
        return members

    def mean_loss(self, ctx: Dict[str, Field], g: GraphView, area_w: torch.Tensor, n_members: int = 2,
                  steps: int = 4, gen: Optional[torch.Generator] = None) -> torch.Tensor:
        """Stage S5c ensemble-mean constraint  E[X^{(k)} | c] = X_bar  on final (projected) samples, with gradient.

        The squared error of a K-member mean contains the member variance / K, so minimising it would also shrink
        the spread.  The U-statistic over distinct pairs,  (1 / K(K-1)) sum_{k != l} <X_k - X_bar, X_l - X_bar>_W,
        is an unbiased estimate of ||bias||^2 alone (members are conditionally independent).  Residual units.
        """
        gen = gen or torch.Generator(device=ctx["xbar"].s.device)
        xbar = ctx["xbar"]
        members = self.assemble(xbar, self._sample_residuals(ctx, g, n_members, steps, gen), area_w)
        return pairwise_bias(members, xbar, self.res_scale_s, self.res_scale_v, area_w)

    @torch.no_grad()
    def sample(self, ctx: Dict[str, Field], g: GraphView, n_members: int, steps: int = 20,
               seed: int = 0) -> List[Field]:
        """Independent residual members R^{(k)} (respaced ancestral sampling); see ``assemble`` for the states."""
        gen = torch.Generator(device=ctx["xbar"].s.device).manual_seed(seed)
        return self._sample_residuals(ctx, g, n_members, steps, gen)


def pairwise_bias(members: List[Field], xbar: Field, scale_s: torch.Tensor, scale_v: torch.Tensor,
                  area_w: torch.Tensor) -> torch.Tensor:
    """Unbiased estimate of ||E[X^{(k)}] - X_bar||^2 (residual units, area weighted, per real component)."""
    K = len(members)
    ds = [(m.s - xbar.s) / scale_s for m in members]
    dv = [(m.v - xbar.v) / scale_v for m in members]
    w = area_w[None, :, None]
    den = w.expand_as(ds[0]).sum() + 2 * w.expand(dv[0].shape[0], -1, dv[0].shape[-1]).sum()
    tot = 0.0
    for k in range(K):
        for l in range(K):
            if k != l:
                tot = tot + ((ds[k] * ds[l] * w).sum() + ((dv[k].conj() * dv[l]).real * w).sum()) / den
    return tot / (K * (K - 1))


def assemble_ensemble(xbar: Field, residuals: List[Field]) -> List[Field]:
    """X^{(k)} = X_bar + R^{(k)} (no forced zero-mean: residual means may carry useful bias correction)."""
    return [xbar + r for r in residuals]
