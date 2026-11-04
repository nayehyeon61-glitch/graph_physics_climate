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


def impr(ref: float, other: float) -> str:
    """Improvement rate of the reference column over ``other`` (+ = the reference has the lower error).
    Sign convention of the project (감자, 2026-11-04): + = better, − = worse, everywhere."""
    return f"{100 * (1 - ref / other):+.1f}%"


def get(c: dict, lead: str, var: str):
    """RMSE of column ``c`` at ``lead`` for ``var`` or None when the run does not predict that variable."""
    return c["ev"]["rmse"].get(lead, {}).get(c["model"], {}).get(var)


def pers(c: dict, lead: str, var: str) -> str:
    """Persistence RMSE stored next to the reference run, or "—" (WeatherBench2 forecast scores carry none)."""
    x = c["ev"]["rmse"].get(lead, {}).get("persistence", {}).get(var)
    return "—" if x is None else fmt(x)


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
        # run may name an eval json directly (e.g. runs/x/eval_2020.json for the external-year benchmark)
        ev_path = Path(run) if run.endswith(".json") else Path(run) / "eval.json"
        run = str(ev_path.parent)
        ev = json.load(open(ev_path))
        dg = Path(run) / "diagnose_guide.json"
        cols.append(dict(label=label, run=run, model=model, ev=ev,
                         dg=json.load(open(dg)) if (dg.exists() and model == "lie_guided" and ev_path.name == "eval.json")
                         else None))
    ref = cols[a.ref]
    leads = sorted(ref["ev"]["rmse"], key=int)
    # rows: the 11 base variables first, then any extra state channel of the reference run (stage10 feature-matrix
    # runs score 21); a column whose run does not predict a variable shows "—"
    ref_vars = list(ref["ev"]["rmse"][leads[0]][ref["model"]])
    rows = [v for v in VARS if v in ref_vars] + [v for v in ref_vars if v not in VARS]
    L = [f"# {a.title}\n", "열: " + "; ".join(f"**{c['label']}** = `{c['run']}` ({c['model']})" for c in cols) + "\n",
         f"{str(ref['ev'].get('init_times', ['2019'])[0])[:4]}년 테스트, 초기 시각 {ref['ev']['n_cases']}개, WeatherBench2 면적가중 RMSE(물리 단위). "
         f"괄호 = 기준열 **{ref['label']}**의 개선율 = 100·(1 − 기준열/해당 열): **+ = 기준열이 그 열보다 좋음, − = 나쁨**.\n"]
    for l in leads:
        L.append(f"## Lead {ref['ev']['lead_hours'][l]} h\n")
        L.append("| 변수 | " + " | ".join(c["label"] for c in cols) + " | persistence |")
        L.append("|---|" + "---|" * (len(cols) + 1))
        for v in rows:
            cells = []
            for i, c in enumerate(cols):
                x = get(c, l, v)
                if x is None:
                    cells.append("—")
                elif x != x:  # NaN: the model's forecast diverged for some init times at this lead
                    cells.append("발산")
                else:
                    cells.append(fmt(x) if i == a.ref else f"{fmt(x)} ({impr(get(ref, l, v), x)})")
            L.append(f"| {v} | " + " | ".join(cells) + f" | {pers(ref, l, v)} |")
        L.append("")
    L.append(f"## 요약: 기준열 **{ref['label']}**의 핵심 변수 개선율 (+ = 기준열이 비교 대상보다 좋음)\n")
    L.append("| lead | 비교 대상 | " + " | ".join(KEY_VARS) + " |")
    L.append("|---|---|" + "---|" * len(KEY_VARS))
    for l in leads:
        for i, c in enumerate(cols):
            if i == a.ref:
                continue
            L.append(f"| {ref['ev']['lead_hours'][l]} h | {c['label']} | " + " | ".join(
                "—" if get(c, l, v) is None else ("발산" if get(c, l, v) != get(c, l, v) else impr(get(ref, l, v), get(c, l, v)))
                for v in KEY_VARS) + " |")
    if any(c["dg"] for c in cols):
        L.append("\n## guide 의존도 (diagnose_guide.py, 24 h, 테스트 48 시각; 진단값 — 부호는 좋고 나쁨이 아니라 MSE 증가량)\n")
        gcols = [c for c in cols if c["dg"]]
        L.append("| 항목 | " + " | ".join(c["label"] for c in gcols) + " |")
        L.append("|---|" + "---|" * len(gcols))
        for k, lab in (("zero_P", "P를 0으로 했을 때 MSE 증가"), ("shuffled_P", "P를 다른 시각으로 교체했을 때 MSE 증가")):
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
