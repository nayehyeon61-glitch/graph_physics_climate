#!/usr/bin/env python
"""Diagnose how much a trained Dynamics V-Net uses the physics guide and what the guide encodes.

1. P-dependence: forecast error of the trained V-Net with the true physics features,
   with features zeroed, and with features taken from another time (shuffled batch).
2. Linear probe on the finest mesh: ridge regression of the normalised 24 h tendency of
   the scalar channels from (a) the current state at the node, (b) the guide features,
   (c) both.  R^2 on held-out times measures information beyond the local state.
3. Fusion gates beta_l and the attention entropy per level.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gpclimate.nn.gauge_conv import transport_interpolate  # noqa: E402
from gpclimate.nn.primitive2d import resolved_fraction  # noqa: E402
from gpclimate.nn.so2 import Field, invariants  # noqa: E402
from gpclimate.training.losses import field_mse  # noqa: E402
from train_eval import Run  # noqa: E402


def ridge_r2(Xtr, ytr, Xte, yte, lam=1e-2):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Xtr = torch.cat([Xtr, torch.ones(len(Xtr), 1)], 1)
    Xte = torch.cat([Xte, torch.ones(len(Xte), 1)], 1)
    A = Xtr.T @ Xtr + lam * len(Xtr) * torch.eye(Xtr.shape[1])
    W = torch.linalg.solve(A, Xtr.T @ ytr)
    pred = Xte @ W
    ss_res = ((yte - pred) ** 2).sum(0)
    ss_tot = ((yte - yte.mean(0)) ** 2).sum(0)
    return (1 - ss_res / ss_tot).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "runs/era5_64x32"))
    ap.add_argument("--data", default=str(ROOT / "data/era5_wb2_64x32_2016_2019.npz"))
    ap.add_argument("--n-cases", type=int, default=48)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    args, _ = ap.parse_known_args()
    torch.set_num_threads(args.threads)
    # reuse Run's loaders with the run's own config
    ns = argparse.Namespace(run=args.run, data=args.data, **{k: None for k in (
        "mesh", "lead", "cs", "cv", "depth", "heads", "ds", "dv", "hidden", "n_basis", "guide_cs", "guide_cv",
        "diff_cs", "diff_cv", "diff_depth", "physics_ops", "tend_head", "fusion", "pigt_layers", "pigt_ops",
        "closure_tend", "resolved_terms", "resolved_coef")})
    run = Run(ns)
    g, w, test = run.g, run.w, run.test
    model = run.forecaster(True)
    run.load_guide(model)
    run.load_vnet(model, "vnet")
    model.eval()
    rep = {"run": args.run}

    # ---------------------------------------------------------------- 1. P-dependence
    idx = list(np.linspace(0, len(test) - 1, args.n_cases).round().astype(int))
    mse = {"true_P": 0.0, "zero_P": 0.0, "shuffled_P": 0.0, "persistence": 0.0}
    with torch.no_grad():
        for i in range(0, len(idx), 8):
            x0n, x1n, cn, _ = run.prep(test, idx[i:i + 8])
            prev = run.prev(test, idx[i:i + 8])  # stage14
            nud = run.increment(model, test, idx[i:i + 8], x0n)  # stage14b (increment from the unmodified model)
            cl = run.col(test, idx[i:i + 8])  # stage15a column inputs (initial time)
            B = x0n.s.shape[0]
            P = model.physics(x0n, cn, g)
            perm = torch.roll(torch.arange(B), 1)
            variants = {
                "true_P": P,
                "zero_P": {k: Field(torch.zeros_like(f.s), torch.zeros_like(f.v)) for k, f in P.items()},
                "shuffled_P": {k: Field(f.s[perm], f.v[perm]) for k, f in P.items()},
            }
            for name, PP in variants.items():
                if model.vnet.cfg.integrator == "ode":  # stage8: apply the variant at every sub-step of the integration
                    p_map = {"true_P": None,
                             "zero_P": lambda P: {k: Field(torch.zeros_like(f.s), torch.zeros_like(f.v)) for k, f in P.items()},
                             "shuffled_P": lambda P: {k: Field(f.s[perm], f.v[perm]) for k, f in P.items()}}[name]
                    fc = model.integrate(x0n, cn, g, p_map)
                else:
                    d = model.vnet(x0n, cn, PP, g, prev=prev, nudge=nud, col=cl)
                    fc = Field(x0n.s + d.s * model.delta_scale_s, x0n.v + d.v * model.delta_scale_v)
                mse[name] += float(field_mse(fc, x1n, w)) * B
            mse["persistence"] += float(field_mse(x0n, x1n, w)) * B
    mse = {k: v / len(idx) for k, v in mse.items()}
    rep["p_dependence_mse"] = mse
    rep["p_dependence_relative"] = {k: mse[k] / mse["true_P"] for k in mse}

    # ---------------------------------------------------------------- 2. linear probe on mesh 0
    has_ops = model.guide.phys_ops is not None

    def collect(ds, ids):
        feats_x, feats_p, feats_o, targets = [], [], [], []
        with torch.no_grad():
            for i in range(0, len(ids), 8):
                x0n, x1n, cn, _ = run.prep(ds, ids[i:i + 8])
                P = model.physics(x0n, cn, g)["P0u"]
                xm = transport_interpolate(x0n, g.grid2mesh)
                dm = transport_interpolate(Field(x1n.s - x0n.s, x1n.v - x0n.v), g.grid2mesh)
                feats_x.append(invariants(xm).flatten(0, 1))
                feats_p.append(invariants(P).flatten(0, 1))
                if has_ops:  # raw fixed operator terms (no learned parameters)
                    feats_o.append(invariants(model.guide.physics_terms(x0n, cn, g)).flatten(0, 1))
                targets.append((dm.s / run.norm.delta_s).flatten(0, 1))
        return (torch.cat(feats_x), torch.cat(feats_p), torch.cat(feats_o) if has_ops else None,
                torch.cat(targets))

    tr_ids = list(np.linspace(0, len(run.calib) - 1, 96).round().astype(int))
    te_ids = list(np.linspace(0, len(test) - 1, 48).round().astype(int))
    Xtr, Ptr, Otr, Ytr = collect(run.calib, tr_ids)
    Xte, Pte, Ote, Yte = collect(test, te_ids)
    probe = {
        "state_only": ridge_r2(Xtr, Ytr, Xte, Yte),
        "guide_only": ridge_r2(Ptr, Ytr, Pte, Yte),
        "state_plus_guide": ridge_r2(torch.cat([Xtr, Ptr], 1), Ytr, torch.cat([Xte, Pte], 1), Yte),
    }
    if has_ops:
        probe["physics_ops_only"] = ridge_r2(Otr, Ytr, Ote, Yte)
        probe["state_plus_physics_ops"] = ridge_r2(torch.cat([Xtr, Otr], 1), Ytr, torch.cat([Xte, Ote], 1), Yte)
    rep["linear_probe_r2_24h_scalar_tendency"] = {"channels": list(run.spec.scalars), **probe}
    rep["linear_probe_r2_mean"] = {k: float(np.mean(v)) for k, v in probe.items()}

    # ---------------------------------------------------------------- 3. gates
    if model.vnet.cfg.fusion == "pigt":
        with torch.no_grad():
            x0n, _, cn, _ = run.prep(test, idx[:8])
            model(x0n, cn, g, prev=run.prev(test, idx[:8]), nudge=run.increment(model, test, idx[:8], x0n),
                  col=run.col(test, idx[:8]))
        rep["pigt_attention_mass"] = {"sources": model.vnet.fusion[0].source_names, **model.vnet.attention_mass()}
    else:
        rep["fusion_gates_mean_abs"] = [float(f.beta_s.abs().mean()) for f in model.vnet.fusion]
        rep["fusion_gates_max_abs"] = [float(f.beta_s.abs().max()) for f in model.vnet.fusion]
    # ---------------------------------------------------------------- 4. layer-wise physics (stage S5): resolved
    # fraction of the 6 h tendency, closure quality, and the PDE residual of the 24 h forecasts (guided / guide-free)
    phys = {}
    if model.guide.closure_out is not None:
        guide, lead = model.guide, run.cfg["lead"]
        w_m = torch.ones(g.n_mesh[0])
        acc = {"tend": [], "F": [], "S": []}
        with torch.no_grad():
            for i in range(0, len(idx), 8):
                x0n, _, cn, x0 = run.prep(test, idx[i:i + 8])
                ids = idx[i:i + 8]
                acc["tend"].append(guide.resolved.pool(run.norm.tendency(x0, test.next_state(ids)), g))
                acc["F"].append(guide.resolved_tendency(x0n, g))
                acc["S"].append(guide.closure(guide(x0n, cn, torch.zeros(len(ids)), g)))
        cat = {k: Field(torch.cat([f.s for f in v]), torch.cat([f.v for f in v])) for k, v in acc.items()}
        phys["resolved_fraction_6h"] = resolved_fraction(cat["tend"], cat["F"], w_m)
        phys["closure_fraction_6h"] = resolved_fraction(cat["tend"], cat["F"] + cat["S"], w_m)
        phys["mse_6h_tendency"] = {"resolved_only": float(field_mse(cat["F"], cat["tend"], w_m)),
                                   "resolved_plus_closure": float(field_mse(cat["F"] + cat["S"], cat["tend"], w_m)),
                                   "zero": float(field_mse(Field(torch.zeros_like(cat["tend"].s),
                                                                 torch.zeros_like(cat["tend"].v)), cat["tend"], w_m))}

        def pde_residual(fm, x0n, fc, cn):
            tend_hat = Field((fc.s - x0n.s) / (lead * run.norm.tend_s), (fc.v - x0n.v) / (lead * run.norm.tend_v))
            xbar = Field(0.5 * (x0n.s + fc.s), 0.5 * (x0n.v + fc.v))
            feats = guide(xbar, cn, torch.zeros(x0n.s.shape[0]), g)
            return float(field_mse(guide.resolved.pool(tend_hat, g), guide.tendency_mesh(feats, xbar, g), w_m))

        models = {"lie_guided": model}
        if (Path(args.run) / "vnet_noguide.pt").exists():
            m2 = run.forecaster(False)
            run.load_vnet(m2, "vnet_noguide")
            m2.eval()
            models["no_guide"] = m2
        res = {k: 0.0 for k in models}
        res["truth"] = 0.0
        with torch.no_grad():
            for i in range(0, len(idx), 8):
                x0n, x1n, cn, _ = run.prep(test, idx[i:i + 8])
                B = x0n.s.shape[0]
                for k, m in models.items():
                    fc, _ = m(x0n, cn, g, prev=run.prev(test, idx[i:i + 8]),
                              nudge=run.increment(m, test, idx[i:i + 8], x0n), col=run.col(test, idx[i:i + 8]))
                    res[k] += pde_residual(m, x0n, fc, cn) * B
                res["truth"] += pde_residual(model, x0n, x1n, cn) * B
        phys["pde_residual_24h"] = {k: v / len(idx) for k, v in res.items()}
        phys["pde_residual_24h"]["note"] = ("mean squared normalised residual (X_hat - X_t)/lead - [F + S](X_bar) on "
                                           "mesh 0; 'truth' = the same residual of the observed X_{t+lead}")
        rep["layerwise_physics"] = phys
    out = args.out or str(Path(args.run) / "diagnose_guide.json")
    json.dump(rep, open(out, "w"), indent=2)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
