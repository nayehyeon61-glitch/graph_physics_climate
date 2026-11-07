#!/usr/bin/env python
"""stage20 integrated version (감자 2026-11-07 23:10 KST: "이후 이점들을 모아서 stage 20으로 해서 통합 버젼 실험"):
choose the configuration that collects the advantages of stage17–19.

Every stage17–19 run was trained with the same paired fine-tuning from stage9b (1000 steps), was decided against the
stage right before it, and -- because each chain builds on the member the previous chain adopted -- its config.json
already carries the components adopted along its path.  The integrated configuration is therefore the run with the
largest mean gain against the common same-budget control (stage17-ctl: stage9b fine-tuned 1000 steps without the new
structure) among the runs that pass the adoption rule against that control (``decide_adoption``: >= 3 of 5 leads and
mean >= 0.5 %).  With none passing, stage20 has no new structure and the chain only reports that.

    python scripts/stage20_compose.py --out runs/era5_64x32_s20_integrated

writes ``<out>/compose.json`` (every run's gain vs the control, the choice, the files and Stage B flags to reuse) and
prints shell assignments (SRC=..., BFLAGS=...) for ``chain_s20.sh``.
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
# stage -> run directory, in chain order
RUNS = {
    "stage17a": "runs/era5_64x32_s17a_dyn",
    "stage17c": "runs/era5_64x32_s17c_transport",
    "stage18": "runs/era5_64x32_s18_core",
    "stage18b": "runs/era5_64x32_s18b_prevpath",
    "stage18c": "runs/era5_64x32_s18c_guidepath",
    "stage18d": "runs/era5_64x32_s18d_laplace",
    "stage19a": "runs/era5_64x32_s19a_gpinn",
    "stage19b": "runs/era5_64x32_s19b_gpinn",
    "stage19c": "runs/era5_64x32_s19c_gpinn",
}
# Stage B options a stage used beyond its config.json (the guide-head fine-tuning is detected from guide_finetuned.pt)
EXTRA_B = {"stage19c": "--pinn-weight-b 0.02"}
# files of the chosen run reused as they are (the guide's Stage A and its preprocessing outputs)
GUIDE_FILES = ("guide.pt", "a_done.json", "log_a.jsonl", "pinn_scales.json", "graph_embed.npy")


def compose(root: Path, ctl: str = CTL, runs=None, min_mean_gain: float = 0.5) -> dict:
    runs = RUNS if runs is None else runs
    ctl_eval = root / ctl / "eval.json"
    if not ctl_eval.exists():
        raise FileNotFoundError(f"control {ctl_eval} missing")
    base = json.load(open(ctl_eval))
    table = {}
    for stage, d in runs.items():
        ev = root / d / "eval.json"
        if ev.exists():
            table[stage] = decide(json.load(open(ev)), base, min_mean_gain=min_mean_gain)
    passing = {s: r for s, r in table.items() if r["adopt"]}
    out = {"control": ctl, "vs_control": table, "chosen": None}
    if passing:
        stage = max(passing, key=lambda s: passing[s]["mean_gain_pct"])
        d = root / runs[stage]
        cfg = json.load(open(d / "config.json"))
        bflags = []
        if (d / "guide_finetuned.pt").exists():
            bflags.append("--guide-lr 3e-5 --guide-heads-only 1")
        if stage in EXTRA_B:
            bflags.append(EXTRA_B[stage])
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
