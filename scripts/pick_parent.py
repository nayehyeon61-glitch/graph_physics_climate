"""Chain helper: among candidate runs, print the one with the largest mean RMSE gain that passes the adoption rule
against the base run (scripts/decide_adoption.py), else the base itself.

    python scripts/pick_parent.py --base runs/<base> runs/<cand1> runs/<cand2> ...

Candidates without an eval.json are skipped.  Used by stage22's chain to pick the stage19 sub-stage it builds on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from decide_adoption import decide


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--min-mean-gain", type=float, default=0.5)
    ap.add_argument("candidates", nargs="*")
    a = ap.parse_args()
    base_ev = json.load(open(Path(a.base) / "eval.json"))
    best, best_gain = a.base, None
    for c in a.candidates:
        p = Path(c) / "eval.json"
        if not p.exists():
            continue
        r = decide(json.load(open(p)), base_ev, min_mean_gain=a.min_mean_gain)
        if r["adopt"] and (best_gain is None or r["mean_gain_pct"] > best_gain):
            best, best_gain = c, r["mean_gain_pct"]
    print(best)


if __name__ == "__main__":
    main()
