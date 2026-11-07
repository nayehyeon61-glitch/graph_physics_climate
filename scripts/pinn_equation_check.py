#!/usr/bin/env python
"""stage18 data check: which physical equations actually hold for ERA5 at 5.6 deg / 6 h?

A PINN-style residual loss (stage18, GPINN guide) only helps when the equation it enforces is close to true on the
data; a residual that is mostly discretisation error teaches the guide that error (stage5a: the reduced momentum
equations explained < 30 % of the 6 h wind tendency).  For a sample of training times this script evaluates, on
mesh 0 and in physical units, the residual  L - R  of candidate equations and reports

  ratio      = var(L - R) / var(L)        (< 1: the equation explains variance of L)
  alpha      = argmin_a var(L - a R)       (1 = calibrated)
  ratio(a)   = var(L - alpha R) / var(L)

Tendency equations compare the observed 6 h change L = Y(t+6h) - Y(t) with R = 6 h * rate, the rate evaluated at
X(t) ("fwd", first order, as in stage5a) or at the midpoint (X(t) + X(t+6h)) / 2 ("mid", second order):

  bve500       absolute vorticity at 500 hPa:   dzeta/dt = -u.grad(zeta + f)
  vort500      + stretching:                    dzeta/dt = -u.grad(zeta + f) - (zeta + f) div u
  adv_t850     dT850/dt = -u850.grad T850
  adv_q700     dq700/dt = -u700.grad q700
  mom500       du500/dt = -(u.grad)u - f k x u - grad Phi500     (stage5a's momentum equation)
  hyps_tend    dPhi500/dt = dPhi850/dt + R ln(850/500) dTbar/dt,  Phi850 = R (t2m+t850)/2 ln(msl/850 hPa),
               Tbar = t850 (layer-mean temperature proxy; alpha absorbs the 850->500 hPa lapse)

Balance (diagnostic) equations compare a field with its balanced estimate at the same time:

  geo500       u500 = (1/f) k x grad Phi500                         (extratropics only)
  geo850       u850 = (1/f) k x grad Phi850
  geo500_tend  du500 = (1/f) k x grad dPhi500                        (balanced 6 h change, extratropics)
  thermal      u500 - u850 = (R/f) ln(850/500) k x grad T850        (thermal wind, extratropics)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gpclimate.nn.physics_ops import EARTH_RADIUS, OMEGA, Stencil  # noqa: E402
from gpclimate.nn.primitive2d import P500, P700, P850, R_DRY, ReducedPrimitiveTendencies  # noqa: E402
from gpclimate.nn.so2 import Field  # noqa: E402
from train_eval import Run  # noqa: E402

DT = 6 * 3600.0
LN85 = math.log(P850 / P500)


def physics_fields(m: Field, sinlat: torch.Tensor, st: Stencil):
    """Mesh-0 physical state -> dict of the quantities the candidate equations need (SI units)."""
    L = st.h * EARTH_RADIUS
    z500, t850, q700, t2m, msl = m.s.unbind(-1)
    u850, u500, u10 = m.v.unbind(-1)
    f = (2 * OMEGA) * sinlat.to(m.s.dtype)[None, :]
    phi850 = R_DRY * 0.5 * (t2m + t850) * (torch.log(msl.clamp(min=1e3)) - math.log(P850))
    w = math.log(P850 / P700) / LN85
    u700 = (1 - w) * u850 + w * u500
    J = st.jacobian(torch.stack([u850, u500], -1)) / L  # [B, N, 2, 2, 2]  d u_i / d x_j
    vort = J[..., 1, 0] - J[..., 0, 1]
    div = J[..., 0, 0] + J[..., 1, 1]
    grads = st.grad(torch.stack([z500, t850, q700, phi850, vort[..., 1] + f.expand_as(z500)], -1)) / L
    g_z500, g_t850, g_q700, g_phi850, g_absv500 = grads.unbind(-1)
    self_adv500 = torch.view_as_complex(torch.einsum("bnij,bnj->bni", J[:, :, 1], torch.view_as_real(u500)).contiguous())
    return dict(z500=z500, t850=t850, q700=q700, phi850=phi850, u850=u850, u500=u500, u700=u700, f=f,
                vort500=vort[..., 1], div500=div[..., 1], g_z500=g_z500, g_t850=g_t850, g_q700=g_q700,
                g_phi850=g_phi850, g_absv500=g_absv500, self_adv500=self_adv500)


def dot(a, b):
    return (a.conj() * b).real


def rates(q):
    """Instantaneous right-hand sides (per second) of the tendency equations."""
    f = q["f"]
    adv_absv = -dot(q["u500"], q["g_absv500"])
    return {
        "bve500": adv_absv,
        "vort500": adv_absv - (q["vort500"] + f) * q["div500"],
        "adv_t850": -dot(q["u850"], q["g_t850"]),
        "adv_q700": -dot(q["u700"], q["g_q700"]),
        "mom500": -(q["self_adv500"] + 1j * f * q["u500"] + q["g_z500"]),
    }


def lhs(q0, q1):
    return {"bve500": q1["vort500"] - q0["vort500"], "vort500": q1["vort500"] - q0["vort500"],
            "adv_t850": q1["t850"] - q0["t850"], "adv_q700": q1["q700"] - q0["q700"], "mom500": q1["u500"] - q0["u500"]}


def balance(q0, q1):
    """(L, R) pairs of the balance relations and of the hypsometric tendency."""
    f = q0["f"]
    out = {
        "geo500": (q0["u500"], 1j * q0["g_z500"] / f),
        "geo850": (q0["u850"], 1j * q0["g_phi850"] / f),
        "geo500_tend": (q1["u500"] - q0["u500"], 1j * (q1["g_z500"] - q0["g_z500"]) / f),
        "thermal": (q0["u500"] - q0["u850"], 1j * R_DRY * LN85 * q0["g_t850"] / f),
        "hyps_tend": (q1["z500"] - q0["z500"], (q1["phi850"] - q0["phi850"]) + R_DRY * LN85 * (q1["t850"] - q0["t850"])),
    }
    return out


def stats(Lv: torch.Tensor, Rv: torch.Tensor):
    Lv, Rv = Lv.flatten(), Rv.flatten()
    vl = float((Lv.abs() ** 2).mean())
    vr = float((Rv.abs() ** 2).mean())
    a = float((Lv.conj() * Rv).real.mean() / max(vr, 1e-30))
    return {"ratio": float(((Lv - Rv).abs() ** 2).mean()) / vl, "alpha": a,
            "ratio_alpha": float(((Lv - a * Rv).abs() ** 2).mean()) / vl, "rhs_over_lhs_var": vr / vl}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "runs/era5_64x32_s9b_merged"))
    ap.add_argument("--data", default=None)
    ap.add_argument("--n-times", type=int, default=160)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "runs/s18_prep/pinn_equation_check.json"))
    args, _ = ap.parse_known_args()
    torch.set_num_threads(args.threads)
    cfg = json.load(open(Path(args.run) / "config.json"))
    ns = argparse.Namespace(run=args.run, data=args.data or str(ROOT / cfg["data"]), **{k: None for k in (
        "mesh", "lead", "cs", "cv", "depth", "heads", "ds", "dv", "hidden", "n_basis", "guide_cs", "guide_cv",
        "diff_cs", "diff_cv", "diff_depth", "physics_ops", "tend_head", "fusion", "pigt_layers", "pigt_ops",
        "closure_tend", "resolved_terms", "resolved_coef")})
    run = Run(ns)
    g, ds = run.g, run.train
    st = Stencil(g.mesh_edges[0])
    sinlat = g.mesh_sinlat[0]
    ids = list(np.linspace(0, len(ds) - 2, args.n_times).round().astype(int))
    acc = {}

    def add(key, Lv, Rv):
        acc.setdefault(key, ([], []))
        acc[key][0].append(Lv)
        acc[key][1].append(Rv)

    with torch.no_grad():
        for i in range(0, len(ids), 8):
            b = ids[i:i + 8]
            _, _, _, x0 = run.prep(ds, b)
            x1 = ds.next_state(b)
            m0, m1 = ReducedPrimitiveTendencies.pool(x0, g), ReducedPrimitiveTendencies.pool(x1, g)  # physical units
            mm = Field(0.5 * (m0.s + m1.s), 0.5 * (m0.v + m1.v))
            q0, q1, qm = (physics_fields(m, sinlat, st) for m in (m0, m1, mm))
            Ls, rf, rm = lhs(q0, q1), rates(q0), rates(qm)
            for k in Ls:
                add(f"{k}|fwd", Ls[k], DT * rf[k])
                add(f"{k}|mid", Ls[k], DT * rm[k])
            for k, (Lv, Rv) in balance(q0, q1).items():
                add(k, Lv, Rv)
    lat = torch.as_tensor(np.abs(run.graphs.mesh[0].lat))
    masks = {"global": lat >= 0, "extratropics": lat > 20, "tropics": lat <= 20}
    rep = {"run": args.run, "n_times": len(ids), "results": {}}
    for k, (Ls, Rs) in acc.items():
        Lc, Rc = torch.cat(Ls), torch.cat(Rs)
        rep["results"][k] = {mn: stats(Lc[:, m], Rc[:, m]) for mn, m in masks.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(args.out, "w"), indent=1)
    lines = [f"# PINN equation check ({len(ids)} training times, mesh 0)", "",
             "| equation | global | extratropics | tropics |", "|---|---|---|---|"]
    for k, r in rep["results"].items():
        cells = [f"{r[m]['ratio']:.2f} / α {r[m]['alpha']:.2f} → {r[m]['ratio_alpha']:.2f}" for m in masks]
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    md = "\n".join(lines) + "\n\n값: var(L−R)/var(L) / 최소제곱 α → var(L−αR)/var(L) (작을수록 식이 잘 성립)\n"
    Path(args.out).with_suffix(".md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
