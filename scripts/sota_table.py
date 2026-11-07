#!/usr/bin/env python
"""Markdown tables for the external benchmark (reports/SOTA_COMPARISON_KO.md): our runs, ClimODE and
WeatherBench2 forecasts scored on the same benchmark year.

Usage:
  python scripts/sota_table.py --ref stage9 \
      "stage9=runs/era5_64x32_s9_integrated/eval_2020.json:lie_guided" \
      "ClimODE=runs/climode_2020_trainmode/eval.json:climode" \
      "GraphCast=runs/wb2_graphcast_2020/eval.json:graphcast" ...

Prints (1) an RMSE table (variables x leads, q700 in g/kg, "—" where a model does not predict the variable),
(2) the mean RMSE ratio ref ÷ column over the variables both predict (> 1: the reference is worse) and
(3) the improvement 1 − ratio of the reference over every (or every --gain) column (+ = the reference is better, project sign convention).
All columns must be scored on the same init times; a mismatch is reported under the tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

VARS = ("z500", "t850", "q700", "t2m", "msl", "u850", "v850", "u500", "v500", "u10", "v10")
SHOW = ("z500", "t850", "q700", "t2m", "msl", "u10")


def load(spec: str) -> dict:
    label, rest = spec.split("=", 1)
    path, model = rest.rsplit(":", 1)
    p = Path(path)
    ev = json.load(open(p if p.suffix == ".json" else p / "eval.json"))
    return {"label": label, "model": model, "ev": ev, "path": str(p)}


def rmse(c: dict, lead: str, var: str):
    return c["ev"]["rmse"].get(lead, {}).get(c["model"], {}).get(var)


def fmt(var: str, x) -> str:
    if x is None:
        return "—"
    if var == "q700":
        x = 1000 * x
    return f"{x:.0f}" if abs(x) >= 10 else f"{x:.2f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cols", nargs="+", help="label=path[:model]")
    ap.add_argument("--ref", required=True, help="label of the reference column (our model)")
    ap.add_argument("--leads", default="1,3,5", help="lead indices of the RMSE table")
    ap.add_argument("--vars", default=",".join(SHOW))
    ap.add_argument("--gain", default="", help="labels of the improvement table (default: every column)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cols = [load(s) for s in a.cols]
    ref = next(c for c in cols if c["label"] == a.ref)
    hours = ref["ev"].get("lead_hours", {})
    leads = a.leads.split(",")
    all_leads = sorted(ref["ev"]["rmse"], key=int)
    lines = ["| 변수 | lead | " + " | ".join(c["label"] for c in cols) + " |",
             "|---|---|" + "---|" * len(cols)]
    for v in a.vars.split(","):
        for l in leads:
            lines.append(f"| {v} | {hours.get(l, l)} h | " + " | ".join(fmt(v, rmse(c, l, v)) for c in cols) + " |")
    lines += ["", f"평균 RMSE 비({a.ref} ÷ 상대, 두 모델이 모두 내는 변수 평균, 1보다 크면 {a.ref}가 나쁨):", "",
              "| 상대 | 공통 변수 | " + " | ".join(f"{hours.get(l, l)} h" for l in all_leads) + " |",
              "|---|---|" + "---|" * len(all_leads)]
    gains = []
    for c in cols:
        if c is ref:
            continue
        shared = [v for v in VARS if rmse(ref, all_leads[0], v) is not None and rmse(c, all_leads[0], v) is not None]
        ratios = [np.mean([rmse(ref, l, v) / rmse(c, l, v) for v in shared]) for l in all_leads]
        lines.append(f"| {c['label']} | {len(shared)} | " + " | ".join(f"{r:.2f}" for r in ratios) + " |")
        if not a.gain or c["label"] in a.gain.split(","):
            gains.append((c["label"], len(shared), [100 * (1 - r) for r in ratios]))
    lines += ["", f"{a.ref}의 개선률(공통 변수 평균 RMSE 비 기준, + = {a.ref}가 나음, − = 나쁨):", "",
              "| 상대 | " + " | ".join(f"{hours.get(l, l)} h" for l in all_leads) + " |",
              "|---|" + "---|" * len(all_leads)]
    for label, _, g in gains:
        lines.append(f"| {label} | " + " | ".join(f"{x:+.1f} %" for x in g) + " |")
    t0 = [str(t) for t in ref["ev"].get("init_times", [])]
    odd = [c["label"] for c in cols if c is not ref and c["ev"].get("init_times") is not None
           and [str(t) for t in c["ev"]["init_times"]] != t0]
    if odd:
        lines += ["", f"초기시각이 기준과 다른 열: {', '.join(odd)} (n = "
                  + ", ".join(str(c["ev"].get("n_cases")) for c in cols if c["label"] in odd) + ")"]
    text = "\n".join(lines) + "\n"
    if a.out:
        Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
