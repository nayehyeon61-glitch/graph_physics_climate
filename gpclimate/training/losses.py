"""Losses and diagnostics for the three training stages."""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

import torch

from gpclimate.geometry.graphs import GraphView, SphereGraphs
from gpclimate.nn.so2 import Field


def field_mse(pred: Field, target: Field, area_w: torch.Tensor,
              ch_w: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
    """Area-weighted MSE per real component (scalars and both vector components weighted equally).

    ``ch_w = (w_s [n_s], w_v [n_v])`` (stage10b) weights the channels: the numerator and the denominator are both
    weighted, so with all weights 1 the value is the plain area-weighted MSE, and down-weighting a channel group
    (e.g. the fcn21 channels outside base11) makes the loss ≈ the MSE of the remaining channels."""
    w = area_w[None, :, None]
    if ch_w is None:
        num = ((pred.s - target.s) ** 2 * w).sum() + ((pred.v - target.v).abs() ** 2 * w).sum()
        den = w.expand_as(pred.s).sum() + 2 * w.expand(pred.v.shape[0], -1, pred.v.shape[-1]).sum()
        return num / den
    ws, wv = ch_w[0].to(pred.s.dtype)[None, None, :], ch_w[1].to(pred.s.dtype)[None, None, :]
    num = ((pred.s - target.s) ** 2 * w * ws).sum() + ((pred.v - target.v).abs() ** 2 * w * wv).sum()
    den = (w * ws).expand_as(pred.s).sum() + 2 * (w * wv).expand(pred.v.shape[0], -1, pred.v.shape[-1]).sum()
    return num / den


def area_mean(x: torch.Tensor, area_w: torch.Tensor) -> torch.Tensor:
    """Global area mean over the node axis (-2) of [B, G, C]."""
    return (x * area_w[None, :, None]).sum(-2) / area_w.sum()


def global_tendency_consistency(pred_delta_s: torch.Tensor, true_delta_s: torch.Tensor,
                                area_w: torch.Tensor) -> torch.Tensor:
    """Weak budget term: the area-integrated change of each scalar should match the observed one.

    It does *not* force the integral to zero (observed budgets are not closed: sources, sinks,
    numerics, assimilation; design doc section 4.5).
    """
    return ((area_mean(pred_delta_s, area_w) - area_mean(true_delta_s, area_w)) ** 2).mean()


def random_gauges(graphs: SphereGraphs, generator: Optional[torch.Generator] = None):
    two_pi = 2 * math.pi
    tg = torch.rand(graphs.grid.n, generator=generator) * two_pi
    tm = [torch.rand(m.n, generator=generator) * two_pi for m in graphs.mesh]
    return tg, tm


def perturb(x: Field, sigma: torch.Tensor, generator: Optional[torch.Generator] = None) -> Field:
    """X~ = X + sigma * L * eps with L = I in normalised units; isotropic complex noise for vectors.

    Noise is drawn on the field's device; a ``generator``, if given, must live on that device.
    """
    B = x.s.shape[0]
    dev, dt = x.s.device, x.s.dtype
    sg = sigma.to(device=dev, dtype=dt).reshape(B, 1, 1)
    es = torch.randn(x.s.shape, generator=generator, device=dev, dtype=dt)
    ev = torch.complex(torch.randn(x.v.shape, generator=generator, device=dev, dtype=dt),
                       torch.randn(x.v.shape, generator=generator, device=dev, dtype=dt))
    return Field(x.s + sg * es, x.v + sg * ev)


def sample_sigma(B: int, sigma_max: float, p_zero: float = 0.25,
                 generator: Optional[torch.Generator] = None) -> torch.Tensor:
    s = torch.rand(B, generator=generator) * sigma_max
    zero = torch.rand(B, generator=generator) < p_zero
    return torch.where(zero, torch.zeros_like(s), s)


def relative_error(a: Field, b: Field) -> float:
    num = (a.s - b.s).pow(2).sum() + (a.v - b.v).abs().pow(2).sum()
    den = b.s.pow(2).sum() + b.v.abs().pow(2).sum() + 1e-30
    return float(torch.sqrt(num / den))


def gauge_equivariance_report(fn: Callable[[Field, Field, GraphView], Dict[str, Field]], x: Field, cond: Field,
                              graphs: SphereGraphs, level_of: Dict[str, str], seed: int = 0) -> Dict[str, float]:
    """Relative error || f(T_g x) - T_g f(x) || / || f(x) || for random per-node gauge rotations.

    ``level_of`` maps each output key to 'grid' or a mesh level index ('0', '1', '2').
    """
    gen = torch.Generator().manual_seed(seed)
    tg, tm = random_gauges(graphs, gen)
    g0 = graphs.view()
    g1 = graphs.gauge_transformed(tm, tg)
    with torch.no_grad():
        out0 = fn(x, cond, g0)
        out1 = fn(x.gauge(tg), cond.gauge(tg), g1)
    rep = {}
    for k, f0 in out0.items():
        lvl = level_of[k]
        theta = tg if lvl == "grid" else tm[int(lvl)]
        rep[k] = relative_error(out1[k], f0.gauge(theta))
    return rep
