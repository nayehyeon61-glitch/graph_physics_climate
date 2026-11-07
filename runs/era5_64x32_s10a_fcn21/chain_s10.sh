#!/usr/bin/env bash
# Stage chain stage10 (2026-11-03): physical-guide feature matrix (FourCastNet / SFNO informed; 감자 2026-11-03).
#   stage10a runs/era5_64x32_s10a_fcn21  state feature matrix base11 -> fcn21 (FourCastNet's 20 channels + q700 =
#            13 scalars + 4 wind pairs), everything else the stage3 recipe (gated fusion, physics operators, tendency
#            head, Stage A forecast-state perturbation, Stage B 2-step rollout from 500, A 1000 / B 1500 steps,
#            2016-2017 training).  Design: docs/design/Physical_Guide_Feature_Matrix_KO.md.
# Order as in the stage7 chain: (1) guide-free V-Net first -- creates config.json (spec=fcn21) and the 21-channel
# normaliser and is the perturbation model for Stage A, (2) Stage A, (3) Stage B, (4) eval (21 variables),
# (5) diagnose.  Compare with stage3 on the 11 common variables (+ ClimODE on its 5).  run_guide_ablation.sh needs no
# --spec: later stages read the spec from the run's config.json.
cd "$(dirname "$0")/../.."
RUN=runs/era5_64x32_s10a_fcn21
DATA=data/era5_wb2_64x32_fcn21_2016_2019.npz
LOG=$RUN/chain_s10.log
until grep -q CHAIN_S8_DONE runs/era5_64x32_s8a_ode/chain_s8.log 2>/dev/null; do sleep 60; done
until grep -q FETCH_DONE data/fetch_fcn21_2016_2019.log 2>/dev/null; do sleep 60; done
echo "[$(date -u +%H:%M:%S)] stage8 chain finished -> stage10a feature matrix fcn21 ($DATA)" >> $LOG
python -W ignore scripts/train_eval.py --run $RUN --data $DATA --spec fcn21 --threads 4 --batch 4 --lr 1e-3 \
  --physics-ops 1 --tend-head 1 --stage b_noguide --steps 1500 --val-every 250 --rollout-steps 2 --rollout-start 500 \
  --rollout-weight 1.0 > $RUN/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] stage10a guide-free V-Net exit $? -> Stage A/B/eval/diagnose" >> $LOG
RUN=$RUN BASE=$RUN DATA=$DATA PERTURB_MODEL=$RUN/vnet_noguide.pt PERTURB_PROB=0.3 ROLLOUT_STEPS=2 ROLLOUT_START=500 \
  bash scripts/run_guide_ablation.sh >> $RUN/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] stage10a pipeline exit $? CHAIN_S10_DONE" >> $LOG
