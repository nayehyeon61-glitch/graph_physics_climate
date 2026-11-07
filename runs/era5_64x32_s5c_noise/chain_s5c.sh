#!/usr/bin/env bash
# Stage chain S5c (2026-11-03): after the S5a/S5b chain, pick the deterministic run with the better test RMSE
# (S5a vs S5b, mean relative RMSE over 11 variables x 5 leads w.r.t. S3) and put a Noise layer on it:
#   S5c-0  runs/era5_64x32_s5c0_diffusion  Stage C residual diffusion without constraints (control)
#   S5c    runs/era5_64x32_s5c_noise       Stage C with feasibility projection + L_feas + L_mean
# Both reuse the chosen run's guide / V-Nets (GUIDE_FROM, VNET_FROM) so the Noise layer is the only change.
cd "$(dirname "$0")/../.."
LOG=runs/era5_64x32_s5c_noise/chain_s5c.log
until grep -q CHAIN_DONE runs/era5_64x32_s5a_closure/chain.log 2>/dev/null; do sleep 60; done
SRC=$(python - <<'PY'
import json
ref = json.load(open("runs/era5_64x32_physops_roll/eval.json"))["rmse"]
best, score = None, None
for run in ("runs/era5_64x32_s5a_closure", "runs/era5_64x32_s5b_pde"):
    try:
        r = json.load(open(run + "/eval.json"))["rmse"]
    except Exception:
        continue
    rel = [r[L]["lie_guided"][v] / ref[L]["lie_guided"][v] - 1 for L in ref for v in ref[L]["lie_guided"]]
    m = sum(rel) / len(rel)
    if score is None or m < score:
        best, score = run, m
print(best or "runs/era5_64x32_s5a_closure")
PY
)
echo "[$(date -u +%H:%M:%S)] S5a/S5b chain finished -> forecaster for S5c: $SRC" >> $LOG
for RUN in runs/era5_64x32_s5c0_diffusion runs/era5_64x32_s5c_noise; do
  mkdir -p $RUN
  cp "$SRC/diagnose_guide.json" "$RUN/" 2>/dev/null  # deterministic models are identical: guide diagnostics carry over
done
RUN=runs/era5_64x32_s5c0_diffusion BASE=$SRC GUIDE_FROM=$SRC VNET_FROM=$SRC STAGE_C=1 NOISE_CONSTRAINTS=0 \
  N_ENS_CASES=45 DIAGNOSE=0 bash scripts/run_guide_ablation.sh > runs/era5_64x32_s5c0_diffusion/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] S5c-0 exit $? -> S5c" >> $LOG
RUN=runs/era5_64x32_s5c_noise BASE=$SRC GUIDE_FROM=$SRC VNET_FROM=$SRC STAGE_C=1 NOISE_CONSTRAINTS=1 \
  N_ENS_CASES=45 DIAGNOSE=0 bash scripts/run_guide_ablation.sh > runs/era5_64x32_s5c_noise/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] S5c exit $? CHAIN_S5C_DONE" >> $LOG
