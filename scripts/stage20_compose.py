#!/usr/bin/env python
"""stage20 integrated version (감자 2026-11-07 23:10 KST: "이후 이점들을 모아서 stage 20으로 해서 통합 버젼 실험"):
choose the configuration that collects the advantages of stage17–19.

Every stage17–19 run was trained with the same paired fine-tuning from stage9b (1000 steps), was decided against the
stage right before it, and -- because each chain builds on the member the previous chain adopted -- its config.json
already carries the components adopted along its path.

Selection rule (decision card 2026-11-08 08:10 KST, 감자 chose "계속 + stage20 규칙 완화"): stage17a (the ODE dynamic
guide every later stage builds on) lost -3 % against the same-budget delta control stage17-ctl at 1000 steps, which
may be under-training of the ODE rather than the structure.  So the configuration is chosen inside the ODE family:
the run with the largest mean gain against stage17a among the runs that pass the adoption rule against stage17a
(``decide_adoption``: >= 3 of 5 leads and mean >= 0.5 %); with none passing, stage17a itself.  stage20 then trains it
with the larger budget (budget.env, 3000 steps) and the final verdict is against the same-budget (3000-step) delta
control.  Every run's gain against stage17-ctl is still recorded (``vs_control``) for the report.

    python scripts/stage20_compose.py --out runs/era5_64x32_s20_integrated

writes ``<out>/compose.json`` (every run's gain vs stage17a and vs the control, the choice, the files and Stage B flags
to reuse) and prints shell assignments (SRC=..., BFLAGS=...) for ``chain_s20.sh``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from decide_adoption import decide  # noqa: E402

CTL = "runs/era5_64x32_s17_ctl"
BASE = "stage17a"  # the ODE dynamic guide every stage17c-19c run builds on
# stage -> run directory, in chain order
RUNS = {
    "stage17a": "runs/era5_64x32_s17a_dyn",
    "stage17c": "runs/era5_64x32_s17c_transport",
    "stage18": "runs/era5_64x32_s18_core",
    "stage18b": "runs/era5_64x32_s18b_prevpath",
    "stage18c": "runs/era5_64x32_s18c_guidepath",
    "stage18d": "runs/era5_64x32_s18d_laplace",
    "stage18e": "runs/era5_64x32_s18e_tv",
    "stage18f": "runs/era5_64x32_s18f_edge",
    "stage18g": "runs/era5_64x32_s18g_geom",
    "stage21a": "runs/era5_64x32_s21a_physfirst",
    "stage21b": "runs/era5_64x32_s21b_guidecore",
    "stage19a": "runs/era5_64x32_s19a_gpinn",
    "stage19b": "runs/era5_64x32_s19b_gpinn",
    "stage19c": "runs/era5_64x32_s19c_gpinn",
    "stage22": "runs/era5_64x32_s22_trisk",
}
# Stage B options a stage used beyond its config.json (the guide-head fine-tuning is detected from guide_finetuned.pt)
EXTRA_B = {"stage19c": "--pinn-weight-b 0.02", "stage21a": "--source-penalty 0.1", "stage21b": "--source-penalty 0.1"}
# files of the chosen run reused as they are (the guide's Stage A and its preprocessing outputs)
GUIDE_FILES = ("guide.pt", "a_done.json", "log_a.jsonl", "pinn_scales.json", "graph_embed.npy")


def compose(root: Path, ctl: str = CTL, runs=None, min_mean_gain: float = 0.5, base: str = BASE) -> dict:
    runs = RUNS if runs is None else runs
    ctl_eval = root / ctl / "eval.json"
    if not ctl_eval.exists():
        raise FileNotFoundError(f"control {ctl_eval} missing")
    evals = {s: json.load(open(root / d / "eval.json")) for s, d in runs.items() if (root / d / "eval.json").exists()}
    ctl_ev = json.load(open(ctl_eval))
    table = {s: decide(ev, ctl_ev, min_mean_gain=min_mean_gain) for s, ev in evals.items()}
    out = {"rule": f"best run vs {base} (adoption rule), else {base}; final verdict vs the same-budget control",
           "base": base, "control": ctl, "vs_control": table, "vs_base": {}, "chosen": None}
    if base not in evals:
        return out
    vs_base = {s: decide(ev, evals[base], min_mean_gain=min_mean_gain) for s, ev in evals.items() if s != base}
    out["vs_base"] = vs_base
    passing = {s: r for s, r in vs_base.items() if r["adopt"]}
    stage = max(passing, key=lambda s: passing[s]["mean_gain_pct"]) if passing else base
    d = root / runs[stage]
    cfg = json.load(open(d / "config.json"))
    bflags = []
    if (d / "guide_finetuned.pt").exists():
        bflags.append("--guide-lr 3e-5 --guide-heads-only 1")
    if stage in EXTRA_B:
        bflags.append(EXTRA_B[stage])
    if (d / "extra_b.txt").exists():  # stage22: Stage B options inherited from the parent it was built on
        extra = (d / "extra_b.txt").read_text().strip()
        if extra:
            bflags.append(extra)
    out.update(chosen=stage, src=runs[stage], bflags=" ".join(bflags),
               files=[f for f in GUIDE_FILES if (d / f).exists()],
               components={k: cfg.get(k) for k in ("integrator", "ode_steps", "ode_path_forcing", "inter_times",
                                                    "flow_head", "ode_physics", "ode_drift", "tendency_core",
                                                    "core_tokens", "prev_path", "core_laplace", "graph_embed")})
    if (d / "pinn_scales.json").exists():
        out["components"]["pinn_stage_a"] = True
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--min-mean-gain", type=float, default=0.5)
    a = ap.parse_args(argv)
    res = compose(Path(a.root), min_mean_gain=a.min_mean_gain)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "compose.json", "w"), indent=1)
    print(f"SRC={res.get('src') or ''}")
    print(f"CHOSEN={res.get('chosen') or ''}")
    print(f"BFLAGS=\"{res.get('bflags', '')}\"")
    print(f"FILES=\"{' '.join(res.get('files', []))}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
