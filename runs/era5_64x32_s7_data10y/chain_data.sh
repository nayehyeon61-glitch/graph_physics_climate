#!/usr/bin/env bash
# Stage S7 (2026-11-03): data stage.  Same model and training recipe as S3 (runs/era5_64x32_physops_roll: gated
# fusion, physics operators, tendency head, forecast-state perturbation in Stage A, 2-step rollout loss in Stage B,
# Stage A 1000 / Stage B 1500 steps) but trained on 10 years (2008-2017, 14,612 times) instead of 2 (2016-2017).
# Hypothesis: training is step/data limited (6000 samples ~ 2 epochs of 2 years, validation loss still falling).
# Order: (1) guide-free V-Net first -- creates config.json + normaliser from the 10-year file and provides the
# perturbation model for Stage A (S3 used the 2-year guide-free V-Net), (2) Stage A, (3) Stage B, (4) eval, (5) diagnose.
cd "$(dirname "$0")/../.."
RUN=runs/era5_64x32_s7_data10y
DATA=data/era5_wb2_64x32_2008_2019.npz
LOG=$RUN/chain_data.log
until grep -q CHAIN_S5C_DONE runs/era5_64x32_s5c_noise/chain_s5c.log 2>/dev/null; do sleep 60; done
echo "[$(date -u +%H:%M:%S)] S5c chain finished -> S7 data stage ($DATA)" >> $LOG
python -W ignore scripts/train_eval.py --run $RUN --data $DATA --threads 4 --batch 4 --lr 1e-3 --physics-ops 1 \
  --tend-head 1 --stage b_noguide --steps 1500 --val-every 250 --rollout-steps 2 --rollout-start 500 \
  --rollout-weight 1.0 > $RUN/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] S7 guide-free V-Net exit $? -> Stage A/B/eval/diagnose" >> $LOG
RUN=$RUN BASE=$RUN DATA=$DATA PERTURB_MODEL=$RUN/vnet_noguide.pt PERTURB_PROB=0.3 ROLLOUT_STEPS=2 ROLLOUT_START=500 \
  bash scripts/run_guide_ablation.sh >> $RUN/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] S7 pipeline exit $? CHAIN_S7_DONE" >> $LOG
