"""Lag-1 autocorrelation of a trained model's own lead-time error (stage14b prior and check).

Evensen (2009, *Data Assimilation: The Ensemble Kalman Filter*, 2nd ed.) represents model error as red noise,
q_k = rho q_{k-1} + sqrt(1 - rho^2) w_k with rho = 1 - dt/tau (eqs. 11.33, 12.3), and, with the state augmented
by q, the model error estimated at an analysis is carried into the next forecast through rho (Sect. 12.1.4).  The
nudging coefficient alpha of stage14b plays the role of rho, so rho can be *measured* before training:

    e_prev = X_t - M(X_{t-lead}),   e_next = X_{t+lead} - M(X_t)      (calibration period, normalised units)
    rho   = <e_prev e_next> / sqrt(<e_prev^2> <e_next^2>)            (per variable, per latitude band)
    beta  = <e_prev e_next> / <e_prev^2>                              (optimal linear carry coefficient)
    gain  = rho^2                                                     (fraction of next-step MSE removed by beta e_prev)

Usage: python scripts/model_error_autocorr.py --run runs/era5_64x32_physops_roll --threads 1 --n-cases 120
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_eval import Run  # noqa: E402

BANDS = {"global": (0.0, 90.0), "tropics": (0.0, 30.0), "midlat": (30.0, 60.0), "polar": (60.0, 90.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "runs/era5_64x32_physops_roll"))
    ap.add_argument("--data", default=str(ROOT / "data/era5_wb2_64x32_2016_2019.npz"))
    ap.add_argument("--n-cases", type=int, default=120)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=None)
    args, _ = ap.parse_known_args()
    torch.set_num_threads(args.threads)
    ns = argparse.Namespace(run=args.run, data=args.data, **{k: None for k in (
        "mesh", "lead", "cs", "cv", "depth", "heads", "ds", "dv", "hidden", "n_basis", "guide_cs", "guide_cv",
        "diff_cs", "diff_cv", "diff_depth", "physics_ops", "tend_head", "fusion", "pigt_layers", "pigt_ops",
        "closure_tend", "resolved_terms", "resolved_coef")})
    run = Run(ns)
    g, ds, lead = run.g, run.calib, int(run.cfg["lead"])
    model = run.forecaster(True)
    run.load_guide(model)
    run.load_vnet(model, "vnet")
    model.eval()
    lat = g.extras["grid_lat"].abs().repeat_interleave(int(g.extras["grid_hw"][1]))  # per grid node (lat-slow, lon-fast)
    masks = {b: ((lat >= lo) & (lat <= hi)).to(run.w) * run.w for b, (lo, hi) in BANDS.items()}
    n_s, n_v = run.spec.n_s, run.spec.n_v
    acc = {b: torch.zeros(3, n_s + n_v, dtype=torch.float64) for b in BANDS}  # <pn>, <pp>, <nn>
    idx = np.linspace(lead, len(ds) - 1, args.n_cases).round().astype(int)
    with torch.no_grad():
        for i in range(0, len(idx), 8):
            ii = idx[i:i + 8]
            x0n, x1n, cn, _ = run.prep(ds, ii)
            xp = run.norm.state(ds.prev_state(ii))
            cp = run.norm.cond(ds.cond_field(torch.as_tensor(ii - lead)))
            fp, _ = model(xp, cp, g, prev=run.prev(ds, ii - lead))
            f1, _ = model(x0n, cn, g, prev=run.prev(ds, ii))
            ep_s, ep_v = x0n.s - fp.s, x0n.v - fp.v
            en_s, en_v = x1n.s - f1.s, x1n.v - f1.v
            for b, w in masks.items():
                W = w[None, :, None].double()
                pn = torch.cat([(W * ep_s.double() * en_s.double()).sum((0, 1)),
                                (W * (ep_v.to(torch.complex128) * en_v.to(torch.complex128).conj()).real).sum((0, 1))])
                pp = torch.cat([(W * ep_s.double() ** 2).sum((0, 1)), (W * ep_v.abs().double() ** 2).sum((0, 1))])
                nn = torch.cat([(W * en_s.double() ** 2).sum((0, 1)), (W * en_v.abs().double() ** 2).sum((0, 1))])
                acc[b] += torch.stack([pn, pp, nn])
    names = list(run.spec.scalars) + ["/".join(p) for p in run.spec.vectors]
    out = {"run": args.run, "n_cases": int(len(idx)), "lead_steps": lead, "bands_abs_lat_deg": BANDS, "channels": names,
           "rho": {}, "beta": {}, "mse_fraction_removed": {}}
    for b, a in acc.items():
        pn, pp, nn = a
        rho = pn / torch.sqrt(pp * nn)
        out["rho"][b] = rho.tolist()
        out["beta"][b] = (pn / pp).tolist()
        out["mse_fraction_removed"][b] = (rho ** 2).tolist()
        out.setdefault("rho_mean", {})[b] = float(rho.mean())
        out.setdefault("rho_mse_weighted", {})[b] = float((pn.sum()) / torch.sqrt(pp.sum() * nn.sum()))
    o = Path(args.out or (Path(args.run) / "model_error_autocorr.json"))
    json.dump(out, open(o, "w"), indent=2)
    lines = [f"# Lag-1 autocorrelation of the {6 * lead} h model error — `{args.run}`", "",
             f"{len(idx)} initial times of the calibration period; e_prev = X_t − M(X_{{t−{6 * lead}h}}), "
             f"e_next = X_{{t+{6 * lead}h}} − M(X_t); ρ = corr(e_prev, e_next) (area-weighted, per channel), "
             "β = optimal linear carry coefficient, ρ² = fraction of next-step MSE a linear carry β·e_prev would remove "
             "(Evensen 2009 Ch. 12: red-noise model error q_k = ρ q_{k−1} + …, ρ = 1 − Δt/τ).", "",
             "| channel | " + " | ".join(f"ρ {b}" for b in BANDS) + " | β global | ρ² global |",
             "|---|" + "---|" * (len(BANDS) + 2)]
    for k, nme in enumerate(names):
        lines.append(f"| {nme} | " + " | ".join(f"{out['rho'][b][k]:+.3f}" for b in BANDS)
                     + f" | {out['beta']['global'][k]:+.3f} | {out['mse_fraction_removed']['global'][k]:.3f} |")
    lines.append("| **mean** | " + " | ".join(f"**{out['rho_mean'][b]:+.3f}**" for b in BANDS)
                 + f" | | {np.mean(out['mse_fraction_removed']['global']):.3f} |")
    lines.append("")
    lines.append("MSE-weighted ρ (all channels pooled): " + ", ".join(f"{b} {v:+.3f}" for b, v in out["rho_mse_weighted"].items()))
    rep = Path(args.report or (ROOT / "reports" / f"model_error_autocorr_{Path(args.run).name}.md"))
    rep.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {o} and {rep}")


if __name__ == "__main__":
    main()
