"""Adapter to the PG-Flow (physical_guide_flowmatching) / KAI ERA5 I/O contract.

Contract (``pgcfm/contract.py`` on branch ``experiment/step13-spherical-ode`` of
nayehyeon61-glitch/physical_guide_flowmatching, KAI_ERA5_IO_FORMAT 2026-10-22):

* tensors ``[B, 67, 121, 240]``: 1.5 deg grid with both poles, latitude ascending,
  longitude 0..358.5;
* channels ``z,q,t,u,v`` x 12 levels (50..1000 hPa) followed by
  ``t2m, sp, tisr, tcwv, orog, msl, skt``; ``orog`` is static, ``tisr`` is forcing;
* z-score normalisation ``(x - mean) / (std + 1e-8)`` per channel.

KAI normalises ``u`` and ``v`` with *different* means and stds, which breaks the
rotation rule of a wind vector.  ``KAIAdapter`` therefore undoes the KAI
normalisation of each wind pair and re-scales it with one common, mean-free
scale per level before forming the complex tangent vector ``u + i v``; the
inverse mapping is exact, so model outputs go back into the KAI layout.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from gpclimate.data.variables import VariableSpec
from gpclimate.nn.so2 import Field

LEVELS_HPA = [50, 100, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
UPPER_AIR_VARS = ["z", "q", "t", "u", "v"]
SURFACE_VARS = ["t2m", "sp", "tisr", "tcwv", "orog", "msl", "skt"]
CHANNEL_NAMES: List[str] = [f"{v}{p}" for v in UPPER_AIR_VARS for p in LEVELS_HPA] + SURFACE_VARS
KAI_STD_EPS = 1e-8
N_LAT, N_LON = 121, 240


def kai_grid(n_lat: int = N_LAT, n_lon: int = N_LON) -> Tuple[np.ndarray, np.ndarray]:
    return np.linspace(-90.0, 90.0, n_lat), np.arange(n_lon) * (360.0 / n_lon)


def kai_spec() -> VariableSpec:
    scalars = tuple(f"{v}{p}" for v in ("z", "q", "t") for p in LEVELS_HPA) + ("t2m", "sp", "tcwv", "msl", "skt")
    vectors = tuple((f"u{p}", f"v{p}") for p in LEVELS_HPA)
    return VariableSpec(scalars=scalars, vectors=vectors, static_scalars=("orog",), static_vectors=(),
                        forcing_scalars=("tisr",))


def reference_stats() -> Tuple[np.ndarray, np.ndarray]:
    """KAI REFERENCE_MEAN / REFERENCE_STD from ``pgcfm.contract`` if PG-Flow is importable."""
    from pgcfm import contract  # type: ignore

    return np.asarray(contract.REFERENCE_MEAN), np.asarray(contract.REFERENCE_STD)


class KAIAdapter:
    def __init__(self, mean: Sequence[float], std: Sequence[float], spec: Optional[VariableSpec] = None):
        self.spec = spec or kai_spec()
        self.mean = torch.as_tensor(np.asarray(mean, np.float64), dtype=torch.float64)
        self.std = torch.as_tensor(np.asarray(std, np.float64), dtype=torch.float64)
        self.idx: Dict[str, int] = {n: i for i, n in enumerate(CHANNEL_NAMES)}
        self.s_idx = [self.idx[n] for n in self.spec.scalars]
        self.u_idx = [self.idx[a] for a, _ in self.spec.vectors]
        self.v_idx = [self.idx[b] for _, b in self.spec.vectors]
        su, sv = self.std[self.u_idx], self.std[self.v_idx]
        self.vec_scale = torch.sqrt((su ** 2 + sv ** 2) / 2)  # one common scale per wind pair

    def _phys(self, x: torch.Tensor, idx: List[int]) -> torch.Tensor:
        m = self.mean[idx].to(x)[None, :, None, None]
        s = (self.std[idx] + KAI_STD_EPS).to(x)[None, :, None, None]
        return x[:, idx] * s + m

    def to_internal(self, x_kai: torch.Tensor) -> Tuple[Field, Field]:
        """KAI-normalised [B, 67, H, W] -> (state Field, condition Field) on the grid (float32)."""
        x = x_kai.to(torch.float64)
        B = x.shape[0]
        s = x[:, self.s_idx]  # scalars keep the KAI z-score
        vs = self.vec_scale.to(x)[None, :, None, None]
        u = self._phys(x, self.u_idx) / vs
        v = self._phys(x, self.v_idx) / vs
        c_idx = [self.idx[n] for n in self.spec.static_scalars + self.spec.forcing_scalars]
        cs = x[:, c_idx]

        def flat(t):
            return t.flatten(2).transpose(1, 2)

        state = Field(flat(s).float(), flat(torch.complex(u, v)).to(torch.complex64))
        cond = Field(flat(cs).float(), torch.zeros(B, s.shape[-2] * s.shape[-1], 0, dtype=torch.complex64,
                                                     device=x.device))
        return state, cond

    def to_kai(self, state: Field, template: torch.Tensor) -> torch.Tensor:
        """Internal state -> KAI-normalised channels written into a copy of ``template`` [B, 67, H, W]."""
        out = template.clone().to(torch.float64)
        B, _, H, W = out.shape

        def grid(t):
            return t.transpose(1, 2).reshape(B, -1, H, W).to(torch.float64)

        out[:, self.s_idx] = grid(state.s)
        vs = self.vec_scale.to(out)[None, :, None, None]
        mean, std = self.mean.to(out), self.std.to(out)
        u = grid(state.v.real) * vs
        v = grid(state.v.imag) * vs
        for k, (iu, iv) in enumerate(zip(self.u_idx, self.v_idx)):
            out[:, iu] = (u[:, k] - mean[iu]) / (std[iu] + KAI_STD_EPS)
            out[:, iv] = (v[:, k] - mean[iv]) / (std[iv] + KAI_STD_EPS)
        return out.to(template.dtype)
