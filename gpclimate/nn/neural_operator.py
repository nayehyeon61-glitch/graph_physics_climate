"""stage13: a *continuous* (function-space) layer after the discrete mesh GNN — a Fourier / spherical neural operator
from the ``neuraloperator`` library (Kossaifi et al. 2024; FNO Li et al. 2021; SFNO Bonev et al. 2023) acting on the
lat-lon grid.

Position in the pipeline (감자, 2026-11-04 09:28 KST): Lie-group guide (gauge-equivariant physics features) ->
discrete mesh Dynamics V-Net -> **continuous operator** on the grid -> forecast.  The block reads the input state,
the conditions and the V-Net's grid delta, applies ``layers`` FNO blocks (global spectral convolution = a
resolution-independent kernel integral, the operator is defined on the function not on the 64x32 sample) and returns a
residual correction of the delta.  Its projection is zero-initialised, so a freshly built model equals the stage3
model exactly (persistence of the recipe; the layer only has to earn its keep).

``conv="fft"``: ``neuralop.layers.SpectralConv`` (2-D FFT, periodic in longitude; latitude treated periodically as in
AFNO/FourCastNet).  ``conv="sht"``: ``neuralop.layers.SphericalConv`` (spherical harmonic transform, needs
``torch_harmonics``; the linear part is SO(3)-equivariant because the multipliers depend on the degree l only).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from gpclimate.nn.so2 import Field


def _conv_module(conv: str):
    if conv == "sht":
        from neuralop.layers.spherical_convolution import SphericalConv  # requires torch_harmonics
        return SphericalConv
    if conv == "fft":
        from neuralop.layers.spectral_convolution import SpectralConv
        return SpectralConv
    raise ValueError(f"neural operator conv must be 'fft' or 'sht', got {conv!r}")


def field_to_image(f: Field, H: int, W: int) -> torch.Tensor:
    """Grid Field (row-major lat-slow, lon-fast nodes) -> [B, C_s + 2 C_v, H, W] (vector = east, north components)."""
    B = f.s.shape[0]
    chans = [f.s]
    if f.v.shape[-1] > 0:
        chans += [f.v.real, f.v.imag]
    img = torch.cat(chans, -1)  # [B, H*W, C]
    return img.view(B, H, W, -1).permute(0, 3, 1, 2)


def image_to_field(img: torch.Tensor, n_s: int, n_v: int) -> Field:
    B = img.shape[0]
    flat = img.permute(0, 2, 3, 1).reshape(B, -1, img.shape[1])
    s = flat[..., :n_s]
    v = torch.complex(flat[..., n_s:n_s + n_v], flat[..., n_s + n_v:n_s + 2 * n_v]) if n_v > 0 \
        else torch.zeros(B, flat.shape[1], 0, dtype=torch.complex64, device=img.device)
    return Field(s, v)


class ContinuousOperatorBlock(nn.Module):
    """Residual neural-operator correction of the V-Net delta on the lat-lon grid.

    Inputs per grid point: normalised state X_t (n_s + 2 n_v), conditions (n_cond_s + 2 n_cond_v), the V-Net delta
    (n_s + 2 n_v) and three sphere coordinates (sin phi, cos phi cos lambda, cos phi sin lambda).  Output: a delta
    correction (n_s + 2 n_v), zero at initialisation.
    """

    def __init__(self, n_s: int, n_v: int, n_cond_s: int, n_cond_v: int, hidden: int = 16, layers: int = 2,
                 modes: Tuple[int, int] = (16, 32), conv: str = "fft"):
        super().__init__()
        from neuralop.layers.fno_block import FNOBlocks
        self.n_s, self.n_v, self.layers, self.conv = n_s, n_v, int(layers), conv
        if conv == "sht" and int(modes[1]) < 2 * int(modes[0]):
            # neuralop's SphericalConv keeps l < modes[0] and m < modes[1] // 2 and needs mmax >= lmax
            raise ValueError(f"spherical modes must satisfy modes[1] >= 2 * modes[0] (lmax, 2 mmax); got {tuple(modes)}")
        self.n_delta = n_s + 2 * n_v
        in_ch = 2 * self.n_delta + n_cond_s + 2 * n_cond_v + 3
        self.lift = nn.Conv2d(in_ch, hidden, 1)
        self.blocks = FNOBlocks(hidden, hidden, tuple(int(m) for m in modes), n_layers=self.layers, norm=None,
                                use_channel_mlp=True, channel_mlp_expansion=0.5, conv_module=_conv_module(conv))
        self.proj = nn.Sequential(nn.Conv2d(hidden, hidden, 1), nn.GELU(), nn.Conv2d(hidden, self.n_delta, 1))
        with torch.no_grad():  # no-op at initialisation
            self.proj[-1].weight.zero_()
            self.proj[-1].bias.zero_()

    def coords(self, lat_deg: torch.Tensor, W: int, device, dtype) -> torch.Tensor:
        phi = torch.deg2rad(lat_deg.to(device=device, dtype=dtype))[:, None]  # [H, 1]
        lam = (2 * torch.pi / W) * torch.arange(W, device=device, dtype=dtype)[None, :]  # [1, W]
        return torch.stack([torch.sin(phi).expand(-1, W), torch.cos(phi) * torch.cos(lam),
                            torch.cos(phi) * torch.sin(lam)], 0)  # [3, H, W]

    def forward(self, x: Field, delta: Field, H: int, W: int, lat_deg: torch.Tensor,
                gauge: Optional[torch.Tensor] = None) -> Field:
        """``x`` = state + conditions concatenated (B or broadcast batch), ``delta`` = V-Net delta [B, H*W, .].

        The operator works in the canonical geographic frame (east, north): grid vectors given in rotated node
        frames (``gauge`` = frame angle per grid node, from ``GraphView.extras['grid_gauge']``) are rotated back
        first and the output is rotated into those frames again, so the whole model stays gauge-equivariant.
        """
        if gauge is not None:
            x, delta = x.gauge(-gauge), delta.gauge(-gauge)
        B = delta.s.shape[0]
        xi = field_to_image(x, H, W)
        if xi.shape[0] != B:
            xi = xi.expand(B, -1, -1, -1)
        img = torch.cat([xi, field_to_image(delta, H, W),
                         self.coords(lat_deg, W, xi.device, xi.dtype)[None].expand(B, -1, -1, -1)], 1)
        h = self.lift(img)
        for i in range(self.layers):
            h = self.blocks(h, i)
        out = image_to_field(self.proj(h), self.n_s, self.n_v)
        return out.gauge(gauge) if gauge is not None else out
