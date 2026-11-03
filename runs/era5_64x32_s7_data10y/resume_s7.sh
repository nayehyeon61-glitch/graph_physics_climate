#!/usr/bin/env bash
# Stage S7 resume (2026-11-03 21:10 KST): the container restarted at 12:04 UTC while Stage B was at step 1100/1500,
# killing every chain.  Stage A (guide.pt, log_a.jsonl) and the guide-free V-Net survived; train_eval.py has no
# Stage B resume, so Stage B is re-run from scratch (the partial vnet.pt / log_vnet.jsonl are kept in partial_restart/).
# GUIDE_FROM=$RUN makes run_guide_ablation.sh skip Stage A (all files already present); vnet_noguide.pt exists, so the
# BASE copy is skipped too.  Ends with the same CHAIN_S7_DONE marker chain_data.sh would have written.
cd "$(dirname "$0")/../.."
RUN=runs/era5_64x32_s7_data10y
DATA=data/era5_wb2_64x32_2008_2019.npz
LOG=$RUN/chain_data.log
echo "[$(date -u +%H:%M:%S)] container restart at 12:04 UTC killed Stage B (step 1100) -> resume: Stage B/eval/diagnose" >> $LOG
RUN=$RUN BASE=$RUN GUIDE_FROM=$RUN DATA=$DATA PERTURB_MODEL=$RUN/vnet_noguide.pt PERTURB_PROB=0.3 ROLLOUT_STEPS=2 ROLLOUT_START=500 \
  bash scripts/run_guide_ablation.sh >> $RUN/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] S7 pipeline exit $? CHAIN_S7_DONE" >> $LOG
