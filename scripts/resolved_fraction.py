#!/usr/bin/env python
"""Stage S5a data check: how much of the observed 6 h tendency do the fixed resolved terms explain?

For a sample of training times the normalised 6 h tendency ``T = (X_{t+6h} - X_t)`` is pooled to mesh 0 and compared
with the fixed resolved dynamics ``F(X_t)`` of ``gpclimate.nn.primitive2d`` (all terms and each term alone):

  ratio      = var(T - F) / var(T)      per variable  (< 1: the term explains tendency variance, > 1: it adds noise)
  alpha      = argmin_a var(T - a F)    the least-squares coefficient of the term (1 = perfectly calibrated)
  ratio(alpha) = var(T - alpha F) / var(T)

Also split into tropics (|lat| < 20) and extra-tropics.  The result decides which terms are kept in the closure
structure (``--resolved-terms``) and is recorded in the stage ledger.
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

from gpclimate.nn.primitive2d import DEFAULT_TERMS, SCALARS, VECTORS, ReducedPrimitiveTendencies  # noqa: E402
from gpclimate.nn.so2 import Field  # noqa: E402
from train_eval import Run  # noqa: E402

NAMES = list(SCALARS) + [a[1:] for a, _ in VECTORS]


def stats(T: Field, F: Field, mask: torch.Tensor):
    """Per-channel ratio, alpha and ratio(alpha) over the masked mesh nodes."""
    out = {}
    for name, t, f in list(zip(SCALARS, T.s.unbind(-1), F.s.unbind(-1))) + list(
            zip([a[1:] for a, _ in VECTORS], T.v.unbind(-1), F.v.unbind(-1))):
        t, f = t[:, mask].flatten(), f[:, mask].flatten()
        vt = float((t.abs() ** 2).mean())
        vf = float((f.abs() ** 2).mean())
        if vf < 1e-20 or vt < 1e-20:
            out[name] = {"ratio": 1.0, "alpha": 0.0, "ratio_alpha": 1.0, "term_var_over_tend_var": vf / max(vt, 1e-20)}
            continue
        alpha = float((t.conj() * f).real.mean() / vf)
        out[name] = {"ratio": float(((t - f).abs() ** 2).mean()) / vt, "alpha": alpha,
                     "ratio_alpha": float(((t - alpha * f).abs() ** 2).mean()) / vt, "term_var_over_tend_var": vf / vt}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "runs/era5_64x32_physops_roll"), help="run dir (data/normaliser)")
    ap.add_argument("--data", default=str(ROOT / "data/era5_wb2_64x32_2016_2019.npz"))
    ap.add_argument("--n-times", type=int, default=160)
    ap.add_argument("--split", default="train", choices=["train", "calib", "test"])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=None)
    args, _ = ap.parse_known_args()
    torch.set_num_threads(args.threads)
    ns = argparse.Namespace(run=args.run, data=args.data, **{k: None for k in (
        "mesh", "lead", "cs", "cv", "depth", "heads", "ds", "dv", "hidden", "n_basis", "guide_cs", "guide_cv",
        "diff_cs", "diff_cv", "diff_depth", "physics_ops", "tend_head", "fusion", "pigt_layers", "pigt_ops",
        "closure_tend", "resolved_terms", "resolved_coef")})
    run = Run(ns)
    g, norm = run.g, run.norm
    ds = {"train": run.train, "calib": run.calib, "test": run.test}[args.split]
    rp = ReducedPrimitiveTendencies(run.spec)
    rp.set_scales(norm.s_mean, norm.s_std, norm.v_scale, norm.tend_s, norm.tend_v)
    ids = list(np.linspace(0, len(ds) - 1, args.n_times).round().astype(int))
    T_all, F_all, terms_all = [], [], {k: [] for k in list(DEFAULT_TERMS) + ['ageo']}
    with torch.no_grad():
        for i in range(0, len(ids), 8):
            x0n, _, _, x0 = run.prep(ds, ids[i:i + 8])
            T_all.append(rp.pool(norm.tendency(x0, ds.next_state(ids[i:i + 8])), g))
            F_all.append(rp.mesh_forward(x0n, g))
            for k, v in rp.normalised_terms(x0n, g).items():
                terms_all[k].append(v)
    cat = lambda fs: Field(torch.cat([f.s for f in fs]), torch.cat([f.v for f in fs]))  # noqa: E731
    T, F = cat(T_all), cat(F_all)
    terms = {k: cat(v) for k, v in terms_all.items()}
    lat = torch.as_tensor(np.abs(run.graphs.mesh[0].lat))
    masks = {"global": lat >= 0, "extratropics_|lat|>20": lat > 20, "tropics_|lat|<20": lat <= 20}
    rep = {"split": args.split, "n_times": len(ids), "tendency_var_normalised": {
        **{n: float((T.s[..., i].abs() ** 2).mean()) for i, n in enumerate(SCALARS)},
        **{a[1:]: float((T.v[..., i].abs() ** 2).mean()) for i, (a, _) in enumerate(VECTORS)}}}
    for mname, m in masks.items():
        rep[mname] = {"all_terms": stats(T, F, m), **{k: stats(T, v, m) for k, v in terms.items()}}
    # ---- joint least-squares calibration (global): winds <- (adv_u, ageo), scalars <- their single term
    def ls(t, fs):
        A = torch.tensor([[float((a.conj() * b).real.mean()) for b in fs] for a in fs], dtype=torch.float64)
        b = torch.tensor([float((t.conj() * f).real.mean()) for f in fs], dtype=torch.float64)
        x = torch.linalg.solve(A + 1e-12 * torch.eye(len(fs), dtype=torch.float64), b)
        fit = sum(float(xi) * f for xi, f in zip(x, fs))
        return [float(v) for v in x], float(((t - fit).abs() ** 2).mean() / (t.abs() ** 2).mean())

    coef = {"coef_s": [1.0, 1.0, 1.0, 1.0, 1.0], "coef_adv_u": [1.0] * 3, "coef_ageo": [1.0] * 3}
    fitted = {}
    for i, (name, term) in enumerate([("t850", "adv_t850"), ("q700", "adv_q700"), ("t2m", "adv_t2m"), ("msl", "mass")]):
        col = {"t850": 1, "q700": 2, "t2m": 3, "msl": 4}[name]
        (a,), r = ls(T.s[..., col].flatten(), [terms[term].s[..., col].flatten()])
        coef["coef_s"][col] = a
        fitted[name] = {"terms": [term], "coef": [a], "ratio": r}
    for k, name in enumerate(["850", "500", "10"]):
        (a, b), r = ls(T.v[..., k].flatten(), [terms["adv_u"].v[..., k].flatten(), terms["ageo"].v[..., k].flatten()])
        coef["coef_adv_u"][k], coef["coef_ageo"][k] = a, b
        fitted[name] = {"terms": ["adv_u", "ageo"], "coef": [a, b], "ratio": r}
    # keep a term group only where it buys at least 2% of tendency variance somewhere
    gain = {}
    for name, f in fitted.items():
        for term, c in zip(f["terms"], f["coef"]):
            r1 = ls(T.s[..., {"t850": 1, "q700": 2, "t2m": 3, "msl": 4}[name]].flatten() if name in SCALARS
                    else T.v[..., ["850", "500", "10"].index(name)].flatten(),
                    [terms[term].s[..., {"t850": 1, "q700": 2, "t2m": 3, "msl": 4}[name]].flatten() if name in SCALARS
                     else terms[term].v[..., ["850", "500", "10"].index(name)].flatten()])[1]
            gain[term] = max(gain.get(term, 0.0), 1.0 - r1)
    kept = [t for t in ("adv_u", "adv_t850", "adv_t2m", "adv_q700", "mass") if gain.get(t, 0.0) >= 0.02]
    if gain.get("ageo", 0.0) >= 0.02:
        kept += ["coriolis", "pgf"]
    for t in ("adv_t850", "adv_t2m", "adv_q700", "mass"):  # zero the coefficient of dropped scalar terms
        if t not in kept:
            coef["coef_s"][{"adv_t850": 1, "adv_t2m": 3, "adv_q700": 2, "mass": 4}[t]] = 0.0
    rep["fit"] = {"coef": coef, "per_variable": fitted, "gain_per_term": gain, "kept_terms": kept}
    out = args.out or str(Path(args.run) / f"resolved_fraction_{args.split}.json")
    json.dump(rep, open(out, "w"), indent=1)
    json.dump({"coef": coef, "kept_terms": kept, "split": args.split, "n_times": len(ids), "fitted": fitted},
              open(Path(out).with_name("resolved_coef.json"), "w"), indent=1)
    # markdown summary: ratio / alpha / ratio(alpha) per variable, global and extratropics
    lines = [f"# resolved fraction ({args.split}, {len(ids)} times)", "",
             "| term | region | " + " | ".join(NAMES) + " |", "|---|---|" + "---|" * len(NAMES)]
    for k in ["all_terms"] + list(DEFAULT_TERMS) + ["ageo"]:
        for mname in ("global", "extratropics_|lat|>20"):
            r = rep[mname][k]
            cells = []
            for n in NAMES:
                x = r[n]
                cells.append("·" if x["term_var_over_tend_var"] < 1e-9 else
                             f"{x['ratio']:.2f} / α {x['alpha']:.2f} → {x['ratio_alpha']:.2f}")
            lines.append(f"| {k} | {mname} | " + " | ".join(cells) + " |")
    lines += ["", "## 결합 최소제곱 보정 (전구)", "", "| 변수 | 항 | 계수 | var(T−F_cal)/var(T) |", "|---|---|---|---|"]
    for name, f in fitted.items():
        lines.append(f"| {name} | {', '.join(f['terms'])} | {', '.join(f'{c:.3f}' for c in f['coef'])} | {f['ratio']:.3f} |")
    lines += ["", f"유지할 항(설명분산 이득 ≥ 2 %): `{','.join(kept)}`"]
    md = "\n".join(lines) + "\n\n값: var(T−F)/var(T) / 최소제곱 계수 α → var(T−αF)/var(T); '·' = 항이 그 변수에 기여하지 않음\n"
    Path(out).with_suffix(".md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
