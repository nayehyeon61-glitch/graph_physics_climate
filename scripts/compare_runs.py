#!/usr/bin/env python
"""Side-by-side comparison of several runs (same data / test init times) in one markdown file.

Usage:
  python scripts/compare_runs.py --out reports/X.md \
      "기존 guide=runs/era5_64x32:lie_guided" "보강 guide=runs/era5_64x32_physops:lie_guided" ...

Each column is ``label=run_dir:model`` where ``model`` is a key of ``eval.json["rmse"][lead]``
(``lie_guided`` or ``no_guide``).  Guide-dependence rows are filled for columns whose run has a
``diagnose_guide.json`` and whose model is ``lie_guided``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VARS = ("z500", "t850", "t2m", "msl", "q700", "u850", "v850", "u500", "v500", "u10", "v10")
KEY_VARS = ("z500", "t850", "t2m", "msl", "u850")


def fmt(x: float) -> str:
    return f"{x:.4g}" if abs(x) < 1000 else f"{x:.0f}"


def pct(new: float, ref: float) -> str:
    return f"{100 * (new / ref - 1):+.1f}%"


def get(c: dict, lead: str, var: str):
    """RMSE of column ``c`` at ``lead`` for ``var`` or None when the run does not predict that variable."""
    return c["ev"]["rmse"].get(lead, {}).get(c["model"], {}).get(var)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("columns", nargs="+", help="label=run_dir:model")
    ap.add_argument("--ref", type=int, default=0, help="index of the reference column for relative changes")
    ap.add_argument("--title", default="run 비교")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cols = []
    for c in a.columns:
        label, rest = c.split("=", 1)
        run, model = rest.rsplit(":", 1)
        ev = json.load(open(Path(run) / "eval.json"))
        dg = Path(run) / "diagnose_guide.json"
        cols.append(dict(label=label, run=run, model=model, ev=ev,
                         dg=json.load(open(dg)) if (dg.exists() and model == "lie_guided") else None))
    ref = cols[a.ref]
    leads = sorted(ref["ev"]["rmse"], key=int)
    L = [f"# {a.title}\n", "열: " + "; ".join(f"**{c['label']}** = `{c['run']}` ({c['model']})" for c in cols) + "\n",
         f"2019년 테스트, 초기 시각 {ref['ev']['n_cases']}개, WeatherBench2 면적가중 RMSE(물리 단위). "
         f"괄호는 기준열 **{ref['label']}** 대비 상대 변화.\n"]
    for l in leads:
        L.append(f"## Lead {ref['ev']['lead_hours'][l]} h\n")
        L.append("| 변수 | " + " | ".join(c["label"] for c in cols) + " | persistence |")
        L.append("|---|" + "---|" * (len(cols) + 1))
        for v in VARS:
            cells = []
            for i, c in enumerate(cols):
                x = get(c, l, v)
                if x is None:
                    cells.append("—")
                else:
                    cells.append(fmt(x) if i == a.ref else f"{fmt(x)} ({pct(x, get(ref, l, v))})")
            L.append(f"| {v} | " + " | ".join(cells) + f" | {fmt(ref['ev']['rmse'][l]['persistence'][v])} |")
        L.append("")
    L.append("## 요약: 핵심 변수의 기준열 대비 상대 변화\n")
    L.append("| lead | 열 | " + " | ".join(KEY_VARS) + " |")
    L.append("|---|---|" + "---|" * len(KEY_VARS))
    for l in leads:
        for i, c in enumerate(cols):
            if i == a.ref:
                continue
            L.append(f"| {ref['ev']['lead_hours'][l]} h | {c['label']} | " + " | ".join(
                "—" if get(c, l, v) is None else pct(get(c, l, v), get(ref, l, v)) for v in KEY_VARS) + " |")
    if any(c["dg"] for c in cols):
        L.append("\n## guide 의존도 (diagnose_guide.py, 24 h, 테스트 48 시각)\n")
        gcols = [c for c in cols if c["dg"]]
        L.append("| 항목 | " + " | ".join(c["label"] for c in gcols) + " |")
        L.append("|---|" + "---|" * len(gcols))
        for k, lab in (("zero_P", "P를 0으로 했을 때 MSE 변화"), ("shuffled_P", "P를 다른 시각으로 교체")):
            L.append(f"| {lab} | " + " | ".join(f"{100 * (c['dg']['p_dependence_relative'][k] - 1):+.0f}%" for c in gcols) + " |")
        for k, lab in (("state_only", "probe R² 상태만"), ("guide_only", "probe R² guide만"),
                       ("state_plus_guide", "probe R² 상태+guide"), ("physics_ops_only", "probe R² 고정 연산자만")):
            L.append(f"| {lab} | " + " | ".join(
                "—" if c["dg"]["linear_probe_r2_mean"].get(k) is None else f"{c['dg']['linear_probe_r2_mean'][k]:.3f}"
                for c in gcols) + " |")
        L.append("| 융합 게이트 \\|β\\| 평균 (5레벨) | " + " | ".join(
            " / ".join(f"{x:.2f}" for x in c["dg"]["fusion_gates_mean_abs"]) if "fusion_gates_mean_abs" in c["dg"]
            else "— (PIGT)" for c in gcols) + " |")
        if any("pigt_attention_mass" in c["dg"] for c in gcols):
            L.append("| PIGT 어텐션 질량 H / P / O (5레벨, 첫 layer) | " + " | ".join(
                "; ".join(":".join(f"{x:.2f}" for x in c["dg"]["pigt_attention_mass"][lvl][0])
                          for lvl in ("P0", "P1", "P2", "P1u", "P0u"))
                if "pigt_attention_mass" in c["dg"] else "—" for c in gcols) + " |")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
