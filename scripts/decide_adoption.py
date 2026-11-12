#!/usr/bin/env python
"""Adoption rule of the stage programme, applied to two eval.json files of the same test protocol.

    python scripts/decide_adoption.py runs/<candidate>/eval.json runs/<baseline>/eval.json [--model lie_guided]

Per lead: improvement = 100 * (1 - mean_v RMSE_cand(v) / RMSE_base(v)) over the variables both score (+ = better).
Adopted when at least ``--min-leads`` (3) of the leads improve and the mean gain over the leads is at least
``--min-mean-gain`` percent (0 by default; chains use 0.5 so a change that only moves the error within noise, like
stage14b's +0.22…−0.37 %, is not merged).  Prints a JSON line and exits 0 when adopted,
1 when not (so chain scripts can branch on it).  The P=0 guide-dependence check stays a manual step.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def decide(cand: dict, base: dict, model: str = "lie_guided", min_leads: int = 3, min_mean_gain: float = 0.0) -> dict:
    if cand.get("init_times") and base.get("init_times") and cand["init_times"] != base["init_times"]:
        raise ValueError("candidate and baseline were scored on different init times")
    gains = {}
    for lead in sorted(base["rmse"], key=int):
        a, b = cand["rmse"][lead][model], base["rmse"][lead][model]
        shared = [v for v in b if v in a]
        gains[lead] = 100 * (1 - float(np.mean([a[v] / b[v] for v in shared])))
    n = sum(g > 0 for g in gains.values())
    mean = float(np.mean(list(gains.values())))
    return {"adopt": n >= min_leads and mean >= min_mean_gain, "improved_leads": n, "mean_gain_pct": round(mean, 2),
            "gain_pct": {k: round(v, 2) for k, v in gains.items()}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("baseline")
    ap.add_argument("--model", default="lie_guided")
    ap.add_argument("--min-leads", type=int, default=3)
    ap.add_argument("--min-mean-gain", type=float, default=0.0)
    a = ap.parse_args(argv)
    res = decide(json.load(open(a.candidate)), json.load(open(a.baseline)), a.model, a.min_leads, a.min_mean_gain)
    res.update(candidate=a.candidate, baseline=a.baseline)
    print(json.dumps(res))
    return 0 if res["adopt"] else 1


if __name__ == "__main__":
    sys.exit(main())
