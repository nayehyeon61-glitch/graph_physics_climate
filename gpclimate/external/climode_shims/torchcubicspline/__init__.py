"""Minimal stand-in for ``torchcubicspline`` (patrick-kidger), enough for ClimODE's ``get_delta_u``.

ClimODE's vendored ``utils.py`` imports ``natural_cubic_spline_coeffs`` / ``NaturalCubicSpline`` and only
uses ``spline.derivative(point)`` of a natural cubic spline through the last three 6-hourly frames.
This shim evaluates the same natural cubic spline with ``scipy.interpolate.CubicSpline(bc_type="natural")``
along the time axis, so no git install is needed in the cloud environment.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import CubicSpline


def natural_cubic_spline_coeffs(t: torch.Tensor, x: torch.Tensor):
    """``t`` [T], ``x`` [..., T, C] (time is the second-to-last axis, as in torchcubicspline)."""
    return (t.detach().cpu().double().numpy(), x.detach().cpu().double().numpy())


class NaturalCubicSpline:
    def __init__(self, coeffs):
        t, x = coeffs
        self._shape = x.shape
        self._spline = CubicSpline(t, np.moveaxis(x, -2, 0), axis=0, bc_type="natural")

    def evaluate(self, point) -> torch.Tensor:
        p = float(point)
        return torch.as_tensor(self._spline(p), dtype=torch.float32)

    def derivative(self, point, order: int = 1) -> torch.Tensor:
        p = float(point)
        return torch.as_tensor(self._spline(p, nu=order), dtype=torch.float32)
