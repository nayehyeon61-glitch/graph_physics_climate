#!/usr/bin/env python
"""Error-structure diagnostics of a trained run (improvement investigation, 2026-11-04).

For the deterministic forecaster of a run (and its guide-free V-Net when present), the same 90 init times and
5 leads as ``train_eval.py --stage eval``:
  1. RMSE per latitude band (tropics |lat| < 20, mid-latitudes 20-60, polar > 60) per variable and lead,
  2. systematic bias: RMS of the case-mean error map / RMSE, and the share of the MSE explained by the
     per-case area-mean error,
  3. activity ratio: area-weighted std of the forecast anomaly (w.r.t. the training climatology) over that of the
     truth (< 1 = damped / smoothed forecasts),
  4. zonal power spectrum ratio forecast / truth per wavenumber band (1-3, 4-7, 8-15, 16-32) in 30-60 deg of
     both hemispheres (small-scale smoothing),
  5. error growth RMSE(lead) / RMSE(24 h),
  6. the ClimODE forecasts of ``runs/climode_2019_trainmode`` on the shared variables for the same metrics,
  7. the Stage B validation curve (log_vnet*.jsonl): last-500-step improvement = step limitation.
Writes <run>/error_diagnostics.json and reports/error_diagnostics_<name>.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gpclimate.data.variables import merge_state  # noqa: E402
from train_eval import Run, log  # noqa: E402

BANDS = {"tropics(|lat|<20)": (0.0, 20.0), "midlat(20-60)": (20.0, 60.0), "polar(>60)": (60.0, 90.1)}
SPEC_BANDS = {"k1-3": (1, 3), "k4-7": (4, 7), "k8-15": (8, 15), "k16-32": (16, 32)}
CLIMODE_VARS = ["z500", "t850", "t2m", "u10", "v10"]


def area_w(lat: np.ndarray) -> np.ndarray:
    w = np.cos(np.deg2rad(lat))
    return w / w.mean()


def rmse(err: np.ndarray, w: np.ndarray) -> float:  # err [T, H, W]
    return float(np.sqrt(np.mean(np.mean(err ** 2 * w[None, :, None], axis=(1, 2)))))


def band_rmse(err: np.ndarray, lat: np.ndarray, w: np.ndarray, lo: float, hi: float) -> float:
    m = (np.abs(lat) >= lo) & (np.abs(lat) < hi)
    ww = w[m]
    return float(np.sqrt(np.mean(np.sum(err[:, m, :] ** 2 * ww[None, :, None], axis=1).mean(-1) / ww.sum())))


def bias_stats(err: np.ndarray, w: np.ndarray):
    mse = np.mean(np.mean(err ** 2 * w[None, :, None], axis=(1, 2)))
    sys_map = err.mean(0)  # case-mean error map
    sys_rms = np.sqrt(np.mean(sys_map ** 2 * w[:, None]))
    gm = np.mean(err * w[None, :, None], axis=(1, 2))  # per-case global mean error
    return {"systematic_rms_over_rmse": float(sys_rms / np.sqrt(mse)), "global_mean_share": float(np.mean(gm ** 2) / mse)}


def activity_ratio(fc: np.ndarray, tr: np.ndarray, clim: np.ndarray, w: np.ndarray) -> float:
    def std(a):
        an = a - clim[None]
        an = an - np.mean(an * w[None, :, None], axis=(1, 2), keepdims=True)
        return np.sqrt(np.mean(an ** 2 * w[None, :, None]))
    return float(std(fc) / std(tr))


def spectrum_ratio(fc: np.ndarray, tr: np.ndarray, lat: np.ndarray):
    m = (np.abs(lat) >= 30) & (np.abs(lat) <= 60)
    wl = np.cos(np.deg2rad(lat[m]))
    pf = np.abs(np.fft.rfft(fc[:, m, :], axis=-1)) ** 2  # [T, Hm, K]
    pt = np.abs(np.fft.rfft(tr[:, m, :], axis=-1)) ** 2
    pf = (pf * wl[None, :, None]).sum(1) / wl.sum()
    pt = (pt * wl[None, :, None]).sum(1) / wl.sum()
    out = {}
    for name, (a, b) in SPEC_BANDS.items():
        out[name] = float(pf[:, a:b + 1].mean() / pt[:, a:b + 1].mean())
    return out


def val_curve(path: Path):
    if not path.exists():
        return None
    rows = [json.loads(l) for l in open(path)]
    v = {r["step"]: (r["val_mse"], r.get("val_mse2")) for r in rows if "val_mse" in r}
    steps = sorted(v)
    out = {"val_mse": {str(s): round(v[s][0], 5) for s in steps}}
    if len(steps) >= 3:
        s_last, s_mid = steps[-1], steps[-3]
        out["last_improvement_pct"] = round(100 * (1 - v[s_last][0] / v[s_mid][0]), 2)
        out["last_improvement_steps"] = s_last - s_mid
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default=str(ROOT / "data/era5_wb2_64x32_2016_2019.npz"))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--eval-stride", type=int, default=16)
    ap.add_argument("--max-lead-steps", type=int, default=5)
    ap.add_argument("--climode", default=str(ROOT / "runs/climode_2019_trainmode/shard_0of1.npz"))
    args, _ = ap.parse_known_args()
    torch.set_num_threads(args.threads)
    ns = argparse.Namespace(run=args.run, data=args.data, **{k: None for k in (
        "mesh", "lead", "cs", "cv", "depth", "heads", "ds", "dv", "hidden", "n_basis", "guide_cs", "guide_cv",
        "diff_cs", "diff_cv", "diff_depth", "physics_ops", "tend_head", "fusion", "pigt_layers", "pigt_ops",
        "closure_tend", "resolved_terms", "resolved_coef")})
    run = Run(ns)
    g, test = run.g, run.test
    H, W, lead, K = len(test.lat), len(test.lon), run.cfg["lead"], args.max_lead_steps
    models = {}
    m = run.forecaster(True)
    run.load_guide(m)
    run.load_vnet(m, "vnet")
    models["lie_guided"] = m.eval()
    if (run.dir / "vnet_noguide.pt").exists():
        m2 = run.forecaster(False)
        run.load_vnet(m2, "vnet_noguide")
        models["no_guide"] = m2.eval()
    clim_phys = run.train.state.mean(0).numpy()  # [C, H, W]
    clim = {v: clim_phys[i] for i, v in enumerate(run.train.names)}
    last = len(test) - 1 - (K - 1) * lead
    cases = list(range(0, last, args.eval_stride))
    log(f"error diagnostics: {len(cases)} init times, {K} leads, models {list(models)}")
    V = run.eval_vars
    truth = {k: {v: [] for v in V} for k in range(1, K + 1)}
    preds = {n: {k: {v: [] for v in V} for k in range(1, K + 1)} for n in list(models) + ["persistence"]}

    def add(store, k, f):
        phys = merge_state(run.norm.state_inv(f), run.spec, H, W)
        for v in V:
            store[k][v].append(phys[v][0].numpy().copy())

    t0 = time.time()
    with torch.no_grad():
        for i in cases:
            x0n, _, _, _ = run.prep(test, [i])
            st = {n: x0n for n in models}
            pv = {n: run.prev(test, [i]) for n in models}  # stage14
            for k in range(1, K + 1):
                j = i + (k - 1) * lead
                cn = run.cond_at(test, [j])
                _, x1n, _, _ = run.prep(test, [j])
                add(truth, k, x1n)
                add(preds["persistence"], k, x0n)
                for n, mdl in models.items():
                    new, _ = mdl(st[n], cn, g, prev=pv[n])
                    pv[n], st[n] = st[n], new
                    add(preds[n], k, st[n])
    log(f"rollouts done in {(time.time() - t0) / 60:.1f} min")
    # ClimODE forecasts on the shared variables (same test split; align by case index)
    cp = Path(args.climode)
    if cp.exists():
        z = np.load(cp)
        cidx = {int(c): i for i, c in enumerate(z["cases"])}
        sel = [cidx[c] for c in cases if c in cidx]
        keep = [ci for ci, c in enumerate(cases) if c in cidx]
        P = z["pred"]  # load once: each z["pred"] access decompresses the whole array (OOM when sliced per case)
        if sel and P.shape[1] >= K:
            preds["ClimODE"] = {k: {v: [P[s, k - 1, c].copy() for s in sel] for c, v in enumerate(CLIMODE_VARS)}
                                for k in range(1, K + 1)}
            log(f"ClimODE forecasts aligned on {len(sel)} of {len(cases)} init times")
        else:
            keep = list(range(len(cases)))
    else:
        keep = list(range(len(cases)))
    lat, w = test.lat, area_w(test.lat)
    rep = {"run": args.run, "n_cases": len(cases), "lead_hours": {k: 6 * lead * k for k in range(1, K + 1)},
           "band_rmse": {}, "bias": {}, "activity": {}, "spectrum": {}, "rmse": {}, "growth": {}}
    for n, st in preds.items():
        for k in range(1, K + 1):
            for v in st[k]:
                if not st[k][v]:
                    continue
                fc = np.stack(st[k][v], 0)
                tr = np.stack(truth[k][v], 0)
                if n == "ClimODE":
                    tr = tr[keep]
                err = fc - tr
                key = f"{n}|{k}|{v}"
                rep["rmse"][key] = rmse(err, w)
                rep["band_rmse"][key] = {b: band_rmse(err, lat, w, lo, hi) for b, (lo, hi) in BANDS.items()}
                rep["bias"][key] = bias_stats(err, w)
                rep["activity"][key] = activity_ratio(fc, tr, clim[v], w)
                if v in ("z500", "t850", "u850", "q700", "msl"):
                    rep["spectrum"][key] = spectrum_ratio(fc, tr, lat)
        for v in st[1]:
            if st[1][v]:
                r1 = rep["rmse"][f"{n}|1|{v}"]
                rep["growth"][f"{n}|{v}"] = [round(rep["rmse"][f"{n}|{k}|{v}"] / r1, 3) for k in range(1, K + 1)]
    rep["val_curve"] = {n: val_curve(run.dir / f"log_{n}.jsonl") for n in ("vnet", "vnet_noguide")}
    rep["params"] = {"vnet": sum(p.numel() for p in models["lie_guided"].vnet.parameters()),
                     "guide": sum(p.numel() for p in models["lie_guided"].guide.parameters()) if models["lie_guided"].guide is not None else 0}
    out = run.dir / "error_diagnostics.json"
    json.dump(rep, open(out, "w"), indent=1)
    log(f"written {out}")
    # ---------------------------------------------------------------- markdown
    name = Path(args.run).name
    md = [f"# 오차 구조 진단: {name}\n", f"초기 시각 {len(cases)}개(stride {args.eval_stride}), lead {list(rep['lead_hours'].values())} h. 부호 규약: 비율·skill은 1 또는 + 쪽이 좋음.\n"]
    md.append("## 1. 위도대별 RMSE (24 h / 120 h)\n")
    md.append("| 변수 | 모델 | " + " | ".join(f"{b} 24h" for b in BANDS) + " | " + " | ".join(f"{b} 120h" for b in BANDS) + " |")
    md.append("|---|---|" + "---|" * (2 * len(BANDS)))
    for v in V:
        for n in preds:
            k1, k5 = f"{n}|1|{v}", f"{n}|{K}|{v}"
            if k1 not in rep["band_rmse"]:
                continue
            md.append(f"| {v} | {n} | " + " | ".join(f"{rep['band_rmse'][k1][b]:.4g}" for b in BANDS) + " | "
                      + " | ".join(f"{rep['band_rmse'][k5][b]:.4g}" for b in BANDS) + " |")
    md.append("\n## 2. 편향과 진폭 (24 h / 72 h / 120 h)\n")
    md.append("체계적 편향 = 사례평균 오차장의 RMS / RMSE (1에 가까울수록 오차가 편향 지배); 전구평균 오차 비중 = 사례별 전구평균 오차²/MSE; 진폭비 = 예보 아노말리 std / 관측 아노말리 std.\n")
    md.append("| 변수 | 모델 | 체계적 편향 24/72/120 | 전구평균 비중 24/72/120 | 진폭비 24/72/120 |")
    md.append("|---|---|---|---|---|")
    ks = [1, min(3, K), K]
    for v in V:
        for n in preds:
            if f"{n}|1|{v}" not in rep["bias"]:
                continue
            md.append(f"| {v} | {n} | " + "/".join(f"{rep['bias'][f'{n}|{k}|{v}']['systematic_rms_over_rmse']:.2f}" for k in ks)
                      + " | " + "/".join(f"{rep['bias'][f'{n}|{k}|{v}']['global_mean_share']:.2f}" for k in ks)
                      + " | " + "/".join(f"{rep['activity'][f'{n}|{k}|{v}']:.2f}" for k in ks) + " |")
    md.append("\n## 3. 동서 스펙트럼 비 (예보/관측 파워, 30–60° 양반구; 1 미만 = 평활화)\n")
    md.append("| 변수 | 모델 | lead | " + " | ".join(SPEC_BANDS) + " |")
    md.append("|---|---|---|" + "---|" * len(SPEC_BANDS))
    for v in ("z500", "t850", "u850"):
        for n in preds:
            for k in (1, K):
                key = f"{n}|{k}|{v}"
                if key in rep["spectrum"]:
                    md.append(f"| {v} | {n} | {rep['lead_hours'][k]} h | " + " | ".join(f"{rep['spectrum'][key][b]:.2f}" for b in SPEC_BANDS) + " |")
    md.append("\n## 4. 오차 증가율 RMSE(lead)/RMSE(24 h)\n")
    md.append("| 변수 | " + " | ".join(preds) + " |")
    md.append("|---|" + "---|" * len(preds))
    for v in V:
        md.append(f"| {v} | " + " | ".join("/".join(f"{x:.2f}" for x in rep["growth"][f"{n}|{v}"]) if f"{n}|{v}" in rep["growth"] else "—" for n in preds) + " |")
    md.append("\n## 5. 학습 곡선 (Stage B 검증 MSE)\n")
    for n, vc in rep["val_curve"].items():
        if vc:
            md.append(f"- {n}: " + ", ".join(f"{s}: {x}" for s, x in vc["val_mse"].items())
                      + (f" — 마지막 {vc['last_improvement_steps']} step 개선 {vc['last_improvement_pct']:+.2f} %" if "last_improvement_pct" in vc else ""))
    md.append(f"\n매개변수: V-Net {rep['params']['vnet']:,}, guide {rep['params']['guide']:,}.")
    Path(ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / f"error_diagnostics_{name}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
