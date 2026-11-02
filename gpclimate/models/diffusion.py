"""Stage C (skeleton) - conditional graph diffusion over the residual of the deterministic forecast.

    R_0 = (X^{ERA5}_{t+dt} - X_bar_{t+dt}) / sigma_R            residual in normalised units
    R_s = sqrt(abar_s) R_0 + sqrt(1 - abar_s) eps
    eps_hat = D_psi(R_s, s | X_bar, C_t),   C_t = (X_t, {P_l}, c_t)
    X^{(k)} = X_bar + sigma_R * R^{(k)}                          ensemble assembly

``s`` is the artificial diffusion time and is kept separate from the
atmospheric time ``t`` everywhere (argument names, logs).

Gauge equivariance: the denoiser is built from the same SO(2) layers; vector
noise is isotropic complex Gaussian, whose law is rotation invariant, so the
*distribution* of samples is equivariant (individual samples match only when the
seed noise is rotated as well, design doc section 11.2).
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


class GraphResidualDiffusion(nn.Module):
    def __init__(self, denoiser: ResidualDenoiser, n_train_steps: int = 1000):
        super().__init__()
        self.denoiser = denoiser
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

    def loss(self, r0: Field, ctx: Dict[str, Field], g: GraphView, area_w: torch.Tensor) -> torch.Tensor:
        B = r0.s.shape[0]
        s = torch.randint(0, self.n, (B,), device=r0.s.device)
        a = self.abar[s].reshape(B, 1, 1)
        eps = self._noise_like(r0)
        r_s = Field(a.sqrt() * r0.s + (1 - a).sqrt() * eps.s, a.sqrt() * r0.v + (1 - a).sqrt() * eps.v)
        pred = self.denoiser(r_s, s, ctx, g)
        w = area_w[None, :, None]
        num = ((pred.s - eps.s) ** 2 * w).sum() + ((pred.v - eps.v).abs() ** 2 * w).sum()
        den = w.expand_as(pred.s).sum() + 2 * w.expand(B, -1, pred.v.shape[-1]).sum()
        return num / den

    @torch.no_grad()
    def sample(self, ctx: Dict[str, Field], g: GraphView, n_members: int, steps: int = 20,
               seed: int = 0) -> List[Field]:
        """Independent members; each starts from its own Gaussian seed (respaced ancestral sampling)."""
        gen = torch.Generator(device=ctx["xbar"].s.device).manual_seed(seed)
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


def assemble_ensemble(xbar: Field, residuals: List[Field]) -> List[Field]:
    """X^{(k)} = X_bar + R^{(k)} (no forced zero-mean: residual means may carry useful bias correction)."""
    return [xbar + r for r in residuals]
