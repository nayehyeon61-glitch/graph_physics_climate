"""stage21: physics metrics of a graph-tendency-core run from its eval.json (the metrics the physics-first stages are
judged on besides the RMSE adoption rule).

    python scripts/core_physics_summary.py runs/<run>/eval.json [runs/<other>/eval.json ...]

Per lead: physics share of the core (rms of transport + flux + diffusion / rms of the sub-step change, lie_guided),
guide-only skill (physics_only vs persistence, 11-variable mean RMSE improvement, + = better) and the guide gain
(lie_guided vs no_guide, same convention).
"""

from __future__ import annotations

import json
import sys


def gain(r: dict, cand: str, base: str) -> float:
    """11-variable mean RMSE improvement 100 (1 - cand / base) in %, + = better."""
    if cand not in r or base not in r:
        return float("nan")
    vals = [100.0 * (1.0 - r[cand][v] / r[base][v]) for v in r[base] if r[base][v] > 0]
    return sum(vals) / len(vals)


def summary(ev: dict) -> list:
    rows = []
    for k, lh in ev["lead_hours"].items():
        r = ev["rmse"][k]
        rows.append({"lead_h": lh, "physics_share": ev.get("physics_share", {}).get(k),
                     "physics_only_vs_persistence_pct": gain(r, "physics_only", "persistence"),
                     "guide_gain_pct": gain(r, "lie_guided", "no_guide")})
    return rows


def main():
    for path in sys.argv[1:]:
        ev = json.load(open(path))
        print(path)
        print("  lead   phys share   guide-only vs persistence   guide gain")
        for row in summary(ev):
            ps = "   --" if row["physics_share"] is None else f"{row['physics_share']:5.3f}"
            print(f"  {row['lead_h']:4d} h   {ps}        {row['physics_only_vs_persistence_pct']:+6.2f} %"
                  f"               {row['guide_gain_pct']:+6.2f} %")


if __name__ == "__main__":
    main()
