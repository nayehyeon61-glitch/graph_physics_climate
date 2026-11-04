#!/usr/bin/env bash
# Stage stage7c (2026-11-04 18:30 KST): step-budget ceiling on the 10-year data.  stage7b (Stage B 4500 steps) beat
# stage7/stage3 by +17.6…+7.3 % and ClimODE at every lead, and its validation MSE was still falling at 4500 steps
# (3500 -> 4499: 0.0594 -> 0.0578).  stage7c changes ONE thing vs stage7b: Stage B 4500 -> 9000 steps (rollout start
# scaled 1500 -> 3000, same fraction), guide reused from stage7 (GUIDE_FROM, Stage A skipped); the guide-free baseline
# is retrained with the same 9000 steps (TRAIN_NOGUIDE=1).  Compare with stage7b (+ stage3, ClimODE).
# Queued after the stage14b chain (end of the improvement queue); ~8.7 h (2.2 s/step + 1.15 s/step baseline).
cd "$(dirname "$0")/../.."
S7=runs/era5_64x32_s7_data10y
RUN=runs/era5_64x32_s7c_steps6x
LOG=$RUN/chain_s7c.log
until grep -q CHAIN_S14B_DONE runs/era5_64x32_s14b_nudge/chain_s14b.log 2>/dev/null; do sleep 60; done
echo "[$(date -u +%H:%M:%S)] stage14b chain finished -> stage7c (Stage B 9000 steps on 10 y)" >> $LOG
RUN=$RUN BASE=$S7 GUIDE_FROM=$S7 DATA=data/era5_wb2_64x32_2008_2019.npz STEPS_B=9000 TRAIN_NOGUIDE=1 \
  ROLLOUT_STEPS=2 ROLLOUT_START=3000 bash scripts/run_guide_ablation.sh > $RUN/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] stage7c pipeline exit $? CHAIN_S7C_DONE" >> $LOG
