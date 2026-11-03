#!/usr/bin/env bash
# Stage chain for the night of 2026-11-03: wait for the PIGT pipeline, then S5a (closure-structured guide), then
# S5b (PDE-residual loss with the S5a guide).  Each stage is its own run directory (one change per stage).
cd "$(dirname "$0")/../.."
until grep -q PIPELINE_DONE runs/era5_64x32_pigt/pipeline.log 2>/dev/null || ! pgrep -f "run_pigt.sh" >/dev/null; do sleep 30; done
echo "[$(date -u +%H:%M:%S)] PIGT pipeline finished -> S5a" >> runs/era5_64x32_s5a_closure/chain.log
RUN=runs/era5_64x32_s5a_closure BASE=runs/era5_64x32_physops_roll CLOSURE_TEND=1 \
  RESOLVED_TERMS=adv_u,adv_t850,adv_q700,coriolis,pgf RESOLVED_COEF=runs/era5_64x32_s5a_closure/resolved_coef.json \
  PERTURB_MODEL=runs/era5_64x32_physops_roll/vnet_noguide.pt PERTURB_PROB=0.3 ROLLOUT_STEPS=2 ROLLOUT_START=500 \
  bash scripts/run_guide_ablation.sh > runs/era5_64x32_s5a_closure/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] S5a exit $? -> S5b" >> runs/era5_64x32_s5a_closure/chain.log
mkdir -p runs/era5_64x32_s5b_pde
RUN=runs/era5_64x32_s5b_pde BASE=runs/era5_64x32_physops_roll GUIDE_FROM=runs/era5_64x32_s5a_closure CLOSURE_TEND=1 \
  RESOLVED_TERMS=adv_u,adv_t850,adv_q700,coriolis,pgf RESOLVED_COEF=runs/era5_64x32_s5a_closure/resolved_coef.json \
  PDE_WEIGHT=0.1 ROLLOUT_STEPS=2 ROLLOUT_START=500 \
  bash scripts/run_guide_ablation.sh > runs/era5_64x32_s5b_pde/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] S5b exit $? CHAIN_DONE" >> runs/era5_64x32_s5a_closure/chain.log
