#!/usr/bin/env bash
# Stage stage11b (2026-11-04 10:05 KST): simplicial vertex-edge-face message passing in the guide (SimplicialBlock x2,
# `simplicial=2`) on top of stage11a (connection/curvature operators), which beat stage3 at 24-72 h (+1.5/+0.9/+0.4 %)
# and is therefore the baseline.  Everything else = stage11a / stage3: same data 2016-2017, stage3 normaliser,
# Stage A retrained (the blocks live inside the guide; perturbation model = stage3's guide-free V-Net), Stage B 1500
# steps with rollout 2 from 500, guide-free V-Net copied from stage3 (unchanged by this stage).
# Queued after stage10b (CPU is shared); ~3 h.  Compare with stage11a (+ stage3, ClimODE).
cd "$(dirname "$0")/../.."
RUN=runs/era5_64x32_s11b_simp
S11A=runs/era5_64x32_s11a_curv
S3=runs/era5_64x32_physops_roll
LOG=$RUN/chain_s11b.log
until grep -q CHAIN_S10B_DONE runs/era5_64x32_s10b_fcn21_w/chain_s10b.log 2>/dev/null; do sleep 60; done
echo "[$(date -u +%H:%M:%S)] stage10b chain finished -> stage11b simplicial blocks (baseline stage11a)" >> $LOG
[ -f $RUN/config.json ] || python - $S11A/config.json $RUN/config.json <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c.update(simplicial=2)
json.dump(c, open(sys.argv[2], "w"), indent=2)
PY
[ -f $RUN/normalizer.pt ] || cp $S3/normalizer.pt $RUN/
RUN=$RUN BASE=$S3 PERTURB_MODEL=$S3/vnet_noguide.pt PERTURB_PROB=0.3 ROLLOUT_STEPS=2 ROLLOUT_START=500 \
  bash scripts/run_guide_ablation.sh > $RUN/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] stage11b pipeline exit $? CHAIN_S11B_DONE" >> $LOG
