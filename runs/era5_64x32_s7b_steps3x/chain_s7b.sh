#!/usr/bin/env bash
# Stage stage7b (2026-11-03 22:55 KST): step-budget stage on the 10-year data.  stage7 (10 y, A 1000 / B 1500) matched
# stage3 step-for-step, so the training is step-limited, not data-limited.  stage7b changes ONE thing: Stage B
# 1500 -> 4500 steps (rollout start scaled 500 -> 1500, same fraction), guide reused from stage7 (GUIDE_FROM, Stage A
# skipped); the guide-free baseline is retrained with the same 4500 steps (TRAIN_NOGUIDE=1).  Compare with stage7.
# Queued after the stage11a chain, moved behind stage12a/12b on 2026-11-04 09:00 KST (improvement stages first); ~6.5 h.
cd "$(dirname "$0")/../.."
S7=runs/era5_64x32_s7_data10y
RUN=runs/era5_64x32_s7b_steps3x
LOG=$RUN/chain_s7b.log
until grep -q CHAIN_S12B_DONE runs/era5_64x32_s12b_width64/chain_s12b.log 2>/dev/null; do sleep 60; done
echo "[$(date -u +%H:%M:%S)] stage12b chain finished -> stage7b (Stage B 4500 steps on 10 y)" >> $LOG
RUN=$RUN BASE=$S7 GUIDE_FROM=$S7 DATA=data/era5_wb2_64x32_2008_2019.npz STEPS_B=4500 TRAIN_NOGUIDE=1 \
  ROLLOUT_STEPS=2 ROLLOUT_START=1500 bash scripts/run_guide_ablation.sh > $RUN/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] stage7b pipeline exit $? CHAIN_S7B_DONE" >> $LOG
