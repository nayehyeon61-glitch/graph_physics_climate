"""stage14b: initial-increment nudging (Newtonian relaxation of the first forecast step).

NWP nudging adds a relaxation term (X_ref - X) / tau to the model tendency.  Here the reference is built from the
only extra information available at the initial time: the model's *own* error over the previous lead,

    e_t = X_t - M(X_{t-lead})          (the "analysis increment" of a pre-forecast started one lead earlier),

and the relaxation coefficient alpha = dt / tau in (0, 1) is a learned, state-dependent gate per variable and grid
node.  The nudged change of the first step is

    delta' = delta_net + alpha(F, e_t) * e_t      (all in delta-scale units),

i.e. the part of the last observed development that the model failed to predict is relaxed into the next step
(flow-dependent error feedback; Danforth & Kalnay 2008 style tendency correction, but learned and local).  Later
rollout steps have no new observation, so e = 0 and the term vanishes (incremental-analysis-update style window of
one lead).  Gauge: alpha is computed from invariants [s, |v|] and multiplies e, which is covariant, so the block is
SO(2)-equivariant.  With e = None the block is skipped, so flag-off runs are unchanged.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from gpclimate.nn.so2 import Field, invariants


class InitialNudging(nn.Module):
    def __init__(self, cs: int, cv: int, n_s: int, n_v: int, hidden: int = 32, alpha_init: float = 0.12):
        super().__init__()
        self.n_s, self.n_v = n_s, n_v
        self.net = nn.Sequential(nn.Linear(cs + cv + n_s + n_v, hidden), nn.SiLU(), nn.Linear(hidden, n_s + n_v))
        with torch.no_grad():  # start from a small, uniform relaxation coefficient
            self.net[-1].weight.zero_()
            self.net[-1].bias.fill_(float(torch.logit(torch.tensor(alpha_init))))
        self.last_alpha: Optional[torch.Tensor] = None  # mean coefficient per channel of the last call (diagnostic)
        # the term just added = the model-error estimate carried into the next rollout step (Evensen 2009 Ch. 12:
        # red-noise model error q_k = rho q_{k-1}, with the learned alpha in the role of rho); None if nothing was added
        self.last_term: Optional[Field] = None

    def alpha(self, emb: Field, inc: Field) -> torch.Tensor:
        """Relaxation coefficients in (0, 1): [B, N, n_s + n_v] from the grid embedding and the increment."""
        return torch.sigmoid(self.net(torch.cat([invariants(emb), invariants(inc)], -1)))

    def forward(self, emb: Field, inc: Field) -> Field:
        a = self.alpha(emb, inc)
        self.last_alpha = a.detach().mean((0, 1))
        self.last_term = Field(a[..., :self.n_s] * inc.s, a[..., self.n_s:] * inc.v)
        return self.last_term
