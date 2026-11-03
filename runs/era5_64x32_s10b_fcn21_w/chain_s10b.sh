#!/usr/bin/env bash
# Stage stage10b (2026-11-04 08:40 KST): fcn21 feature matrix with the training losses focused on the base11 channels
# (weights: base11 state channels 1.0, the 10 new channels 0.2; Stage A reconstruction + tendency head, Stage B one-step
# + rollout, guide-free baseline).  Everything else = stage10a (same data file, normaliser, recipe A 1000 / B 1500).
# Separates "21-channel loss dilution" (stage10a's main suspect) from "value of the column information for the guide".
# Stage A perturbation model = stage10a's guide-free V-Net (unweighted loss; it only perturbs the Stage A inputs).
# Queued after stage7b (CPU is shared); ≈2.3 h.  Compare with stage10a and stage3 (+ ClimODE).
cd "$(dirname "$0")/../.."
S10A=runs/era5_64x32_s10a_fcn21
RUN=runs/era5_64x32_s10b_fcn21_w
DATA=data/era5_wb2_64x32_fcn21_2016_2019.npz
LOG=$RUN/chain_s10b.log
until grep -q CHAIN_S7B_DONE runs/era5_64x32_s7b_steps3x/chain_s7b.log 2>/dev/null; do sleep 60; done
echo "[$(date -u +%H:%M:%S)] stage7b chain finished -> stage10b (fcn21, base11-focused loss 0.2)" >> $LOG
python - "$S10A/config.json" "$RUN/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c.update(loss_focus="base11", loss_focus_weight=0.2)
json.dump(c, open(sys.argv[2], "w"), indent=2)
PY
cp "$S10A/normalizer.pt" "$RUN/"
RUN=$RUN BASE=$S10A DATA=$DATA PERTURB_MODEL=$S10A/vnet_noguide.pt PERTURB_PROB=0.3 ROLLOUT_STEPS=2 ROLLOUT_START=500 \
  TRAIN_NOGUIDE=1 bash scripts/run_guide_ablation.sh > $RUN/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] stage10b pipeline exit $? CHAIN_S10B_DONE" >> $LOG
