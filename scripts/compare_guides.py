#!/usr/bin/env python
"""Compare two runs that differ only in the physics guide (same data, V-Net, steps, normaliser).

Reads ``eval.json``, ``diagnose_guide.json`` and the Stage-A/B logs of a base run and a new run and
writes one Korean markdown report with: test RMSE per lead (old guide / new guide / no guide /
persistence), the relative change, guide-dependence counterfactuals, linear-probe R^2 and gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VARS = ("z500", "t850", "t2m", "msl", "q700", "u850", "v850", "u500", "v500", "u10", "v10")
KEY_VARS = ("z500", "t850", "t2m", "msl", "u850", "v850")


def fmt(x: float) -> str:
    if x == 0:
        return "0"
    if abs(x) < 0.01:
        return f"{x:.4g}"
    return f"{x:.4g}" if abs(x) < 1000 else f"{x:.0f}"


def pct(new: float, ref: float) -> str:
    return f"{100 * (new / ref - 1):+.1f}%"


def last_val(path: Path, key: str):
    v = None
    if path.exists():
        for line in open(path):
            rec = json.loads(line)
            if key in rec:
                v = rec[key]
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="runs/era5_64x32")
    ap.add_argument("--new", default="runs/era5_64x32_physops")
    ap.add_argument("--out", default="reports/GUIDE_STRENGTHENING_KO.md")
    a = ap.parse_args()
    base, new = Path(a.base), Path(a.new)
    eb, en = json.load(open(base / "eval.json")), json.load(open(new / "eval.json"))
    db = json.load(open(base / "diagnose_guide.json")) if (base / "diagnose_guide.json").exists() else None
    dn = json.load(open(new / "diagnose_guide.json")) if (new / "diagnose_guide.json").exists() else None
    cfg = json.load(open(new / "config.json"))
    leads = sorted(en["rmse"], key=int)
    L = []
    L.append("# 물리 가이드 보강 결과: 기존 guide / 보강 guide / guide 없음\n")
    L.append(f"기준 run `{base}` (기존 guide) 와 새 run `{new}` (보강 guide). 두 run은 데이터(2016–17 학습, 2018 보정, "
             f"2019 테스트), 정규화, V-Net 구성과 학습 step(Stage A 1000 / B 1500, batch 4, cosine LR 1e-3), "
             f"평가 초기 시각({en['n_cases']}개, stride 16×6 h)이 같고 guide만 다르다. `no_guide`는 guide와 무관하므로 "
             f"기준 run의 것을 그대로 재사용했다 (두 run의 no_guide 수치가 같은지 아래에서 확인).\n")
    L.append("보강 내용: (1) 고정 등변 물리 연산자(기울기·발산/회전·스칼라/벡터 이류·코리올리·절대와도 이류/신장·라플라시안)를 "
             "guide의 최상위 메시 임베딩에 입력, (2) Stage A에 6 h 경향 예측 head 추가(가중치 1.0). "
             f"guide 크기(cs {cfg['guide_cs']}/cv {cfg['guide_cv']})와 융합 게이트 초기값(0.1)은 그대로다.\n")

    # ---------------------------------------------------------------- Stage A / B logs
    L.append("## 1. 학습 곡선 (검증 2018)\n")
    L.append("| 항목 | 기존 guide | 보강 guide |")
    L.append("|---|---|---|")
    L.append(f"| Stage A 최종 val 복원 MSE | {fmt(last_val(base / 'log_a.jsonl', 'val_rec'))} | "
             f"{fmt(last_val(new / 'log_a.jsonl', 'val_rec'))} |")
    L.append(f"| Stage A 최종 val 와도/발산 MSE | {fmt(last_val(base / 'log_a.jsonl', 'val_aux'))} | "
             f"{fmt(last_val(new / 'log_a.jsonl', 'val_aux'))} |")
    vt = last_val(new / 'log_a.jsonl', 'val_tend')
    L.append(f"| Stage A 최종 val 6 h 경향 MSE (정규화, 1 = 분산) | — | {fmt(vt)} (R² ≈ {1 - vt:.2f}) |")
    L.append(f"| Stage B 최종 val 24 h MSE (정규화) | {fmt(last_val(base / 'log_vnet.jsonl', 'val_mse'))} | "
             f"{fmt(last_val(new / 'log_vnet.jsonl', 'val_mse'))} |")
    L.append(f"| Stage B guide 없음 val 24 h MSE | {fmt(last_val(base / 'log_vnet_noguide.jsonl', 'val_mse'))} | (재사용) |")
    steps = {}
    for nm, p in (("기존", base / "log_vnet.jsonl"), ("보강", new / "log_vnet.jsonl"), ("없음", base / "log_vnet_noguide.jsonl")):
        steps[nm] = {int(json.loads(l)["step"]): json.loads(l)["val_mse"] for l in open(p) if "val_mse" in l}
    common = sorted(set(steps["기존"]) & set(steps["보강"]) & set(steps["없음"]))
    L.append("\nStage B 검증 MSE의 step별 추이:\n")
    L.append("| step | 기존 guide | 보강 guide | guide 없음 |")
    L.append("|---|---|---|---|")
    for s in common:
        L.append(f"| {s} | {steps['기존'][s]:.4f} | {steps['보강'][s]:.4f} | {steps['없음'][s]:.4f} |")

    # ---------------------------------------------------------------- test RMSE
    L.append("\n## 2. 2019년 테스트 RMSE (WeatherBench2 면적가중, 물리 단위)\n")
    same = all(abs(eb["rmse"][l]["no_guide"][v] - en["rmse"][l]["no_guide"][v]) <= 1e-6 * abs(eb["rmse"][l]["no_guide"][v])
               for l in leads for v in VARS)
    L.append(f"두 run의 `no_guide` 수치 일치: {'예' if same else '아니오 (아래는 새 run의 값)'}\n")
    for l in leads:
        rb, rn = eb["rmse"][l], en["rmse"][l]
        L.append(f"### Lead {en['lead_hours'][l]} h\n")
        L.append("| 변수 | 기존 guide | 보강 guide | guide 없음 | persistence | 보강 vs 기존 | 보강 vs 없음 |")
        L.append("|---|---|---|---|---|---|---|")
        for v in VARS:
            L.append(f"| {v} | {fmt(rb['lie_guided'][v])} | {fmt(rn['lie_guided'][v])} | {fmt(rn['no_guide'][v])} | "
                     f"{fmt(rn['persistence'][v])} | {pct(rn['lie_guided'][v], rb['lie_guided'][v])} | "
                     f"{pct(rn['lie_guided'][v], rn['no_guide'][v])} |")
        L.append("")
    L.append("### 요약: 핵심 변수의 상대 변화\n")
    L.append("| lead | " + " | ".join(f"{v} 보강 vs 기존" for v in KEY_VARS) + " | " +
             " | ".join(f"{v} 보강 vs 없음" for v in KEY_VARS) + " |")
    L.append("|---|" + "---|" * (2 * len(KEY_VARS)))
    for l in leads:
        rb, rn = eb["rmse"][l], en["rmse"][l]
        L.append(f"| {en['lead_hours'][l]} h | " + " | ".join(pct(rn['lie_guided'][v], rb['lie_guided'][v]) for v in KEY_VARS)
                 + " | " + " | ".join(pct(rn['lie_guided'][v], rn['no_guide'][v]) for v in KEY_VARS) + " |")

    # ---------------------------------------------------------------- guide dependence
    if db and dn:
        L.append("\n## 3. guide 의존도 진단 (scripts/diagnose_guide.py)\n")
        L.append("학습된 V-Net의 24 h 예측 MSE(정규화, 테스트 48 시각)를 guide 특징 P를 그대로 / 0으로 / 다른 시각 것으로 바꿔 측정:\n")
        L.append("| P 처리 | 기존 guide | 보강 guide |")
        L.append("|---|---|---|")
        for k, lab in (("true_P", "그대로"), ("zero_P", "0으로"), ("shuffled_P", "다른 시각으로 교체"), ("persistence", "(참고) persistence")):
            L.append(f"| {lab} | {db['p_dependence_mse'][k]:.4f} ({100 * (db['p_dependence_relative'][k] - 1):+.0f}%) | "
                     f"{dn['p_dependence_mse'][k]:.4f} ({100 * (dn['p_dependence_relative'][k] - 1):+.0f}%) |")
        L.append("\n24 h 스칼라 경향에 대한 선형(ridge) probe R² (최상위 메시, 2018 적합 → 2019 평가, 채널 평균):\n")
        L.append("| 입력 | 기존 run | 보강 run |")
        L.append("|---|---|---|")
        keys = [("state_only", "현재 상태만"), ("guide_only", "guide 특징만"), ("state_plus_guide", "상태 + guide"),
                ("physics_ops_only", "고정 물리 연산자만 (학습 없음)"), ("state_plus_physics_ops", "상태 + 연산자")]
        for k, lab in keys:
            b = db["linear_probe_r2_mean"].get(k)
            n = dn["linear_probe_r2_mean"].get(k)
            L.append(f"| {lab} | {'—' if b is None else f'{b:.3f}'} | {'—' if n is None else f'{n:.3f}'} |")
        L.append("\n채널별 R² (보강 run, guide만): " + ", ".join(
            f"{c} {r:.2f}" for c, r in zip(dn["linear_probe_r2_24h_scalar_tendency"]["channels"],
                                           dn["linear_probe_r2_24h_scalar_tendency"]["guide_only"])))
        L.append(f"\n융합 게이트 |β| 평균 (레벨 0,1,2,1*,0*): 기존 {['%.3f' % x for x in db['fusion_gates_mean_abs']]}, "
                 f"보강 {['%.3f' % x for x in dn['fusion_gates_mean_abs']]}\n")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
