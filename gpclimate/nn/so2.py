"""SO(2)-equivariant feature fields and pointwise layers.

A field lives on a node set and has two parts:

* ``s``: real scalars ``[..., N, Cs]`` (frequency 0, gauge invariant)
* ``v``: complex tangent vectors ``[..., N, Cv]`` (frequency 1); under a gauge
  rotation ``theta_i`` of node ``i`` they become ``exp(-i theta_i) v``.

Equivariant pointwise maps:

* linear: real matrix on ``s``; *complex* matrix on ``v`` (the commutant of the
  frequency-1 irrep is C, so complex weights are the most general equivariant
  linear map; ``i * v`` is the 90 degree rotation ``k x v`` that appears in
  geostrophic balance and the Coriolis term),
* norms / gates: functions of invariants (``s`` and ``|v|``) only.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Sequence

import torch
from torch import nn


class Field(NamedTuple):
    s: torch.Tensor
    v: torch.Tensor

    @property
    def cs(self) -> int:
        return self.s.shape[-1]

    @property
    def cv(self) -> int:
        return self.v.shape[-1]

    def __add__(self, other: "Field") -> "Field":  # type: ignore[override]
        return Field(self.s + other.s, self.v + other.v)

    def scale(self, a_s, a_v) -> "Field":
        return Field(self.s * a_s, self.v * a_v)

    def gauge(self, theta: torch.Tensor) -> "Field":
        """Coordinates after rotating every node frame by theta [N]."""
        return Field(self.s, self.v * torch.exp(-1j * theta.to(self.s))[..., None])

    def detach(self) -> "Field":
        return Field(self.s.detach(), self.v.detach())


def cat_fields(fields: Sequence[Field]) -> Field:
    return Field(torch.cat([f.s for f in fields], -1), torch.cat([f.v for f in fields], -1))


def pack(f: Field) -> torch.Tensor:
    """Real tensor [..., Cs + 2Cv] for message-passing libraries that are real-only."""
    return torch.cat([f.s, torch.view_as_real(f.v).flatten(-2)], -1)


def unpack(x: torch.Tensor, cs: int, cv: int) -> Field:
    v = x[..., cs:cs + 2 * cv].reshape(*x.shape[:-1], cv, 2)
    return Field(x[..., :cs], torch.view_as_complex(v.contiguous()))


def phase(angle: torch.Tensor) -> torch.Tensor:
    return torch.polar(torch.ones_like(angle), angle)


class EqLinear(nn.Module):
    """s' = W_s s + b,  v' = W_v v  (W_v complex, no bias: a constant vector is not equivariant)."""

    def __init__(self, cs_in: int, cv_in: int, cs_out: int, cv_out: int, bias: bool = True):
        super().__init__()
        self.cs_out, self.cv_out = cs_out, cv_out
        self.lin_s = nn.Linear(cs_in, cs_out, bias=bias) if cs_in > 0 and cs_out > 0 else None
        self.bias_only = nn.Parameter(torch.zeros(cs_out)) if (cs_in == 0 and cs_out > 0 and bias) else None
        if cv_in > 0 and cv_out > 0:
            std = 1.0 / math.sqrt(2 * cv_in)
            self.w_re = nn.Parameter(torch.randn(cv_out, cv_in) * std)
            self.w_im = nn.Parameter(torch.randn(cv_out, cv_in) * std)
        else:
            self.w_re = self.w_im = None

    def forward(self, f: Field) -> Field:
        lead = f.s.shape[:-1]
        if self.lin_s is not None:
            s = self.lin_s(f.s)
        else:
            s = f.s.new_zeros(*lead, self.cs_out)
            if self.bias_only is not None:
                s = s + self.bias_only
        if self.w_re is not None:
            w = torch.complex(self.w_re, self.w_im)
            v = f.v @ w.transpose(0, 1)
        else:
            v = f.v.new_zeros(*lead, self.cv_out)
        return Field(s, v)


class EqNorm(nn.Module):
    """LayerNorm on scalars; RMS-norm of vector magnitudes (no centring, equivariant)."""

    def __init__(self, cs: int, cv: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(cs) if cs > 0 else None
        self.gain_v = nn.Parameter(torch.ones(cv)) if cv > 0 else None
        self.eps = eps

    def forward(self, f: Field) -> Field:
        s = self.ln(f.s) if self.ln is not None else f.s
        if self.gain_v is not None:
            rms = torch.sqrt((f.v.abs() ** 2).mean(-1, keepdim=True) + self.eps)
            v = f.v / rms * self.gain_v
        else:
            v = f.v
        return Field(s, v)


class Gate(nn.Module):
    """SiLU on scalars, sigmoid gates (from invariants) on vectors (e3nn-style gated nonlinearity)."""

    def __init__(self, cs: int, cv: int):
        super().__init__()
        self.cv = cv
        self.gate = nn.Linear(cs + cv, cv) if cv > 0 else None

    def forward(self, f: Field) -> Field:
        s = torch.nn.functional.silu(f.s)
        if self.gate is None:
            return Field(s, f.v)
        g = torch.sigmoid(self.gate(torch.cat([f.s, f.v.abs()], -1)))
        return Field(s, f.v * g)


class EqFFN(nn.Module):
    def __init__(self, cs: int, cv: int, mult: int = 2, cs_out: int = None, cv_out: int = None):
        super().__init__()
        cs_out = cs if cs_out is None else cs_out
        cv_out = cv if cv_out is None else cv_out
        self.l1 = EqLinear(cs, cv, mult * cs, mult * cv)
        self.act = Gate(mult * cs, mult * cv)
        self.l2 = EqLinear(mult * cs, mult * cv, cs_out, cv_out)

    def forward(self, f: Field) -> Field:
        return self.l2(self.act(self.l1(f)))


def invariants(f: Field) -> torch.Tensor:
    """Gauge-invariant summary of a field: [s, |v|]."""
    return torch.cat([f.s, f.v.abs()], -1)
