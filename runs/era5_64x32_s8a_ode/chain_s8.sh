#!/usr/bin/env bash
# Stage chain S8 (2026-11-03): graph neural ODE dynamics on top of the learned physics, after the S7 data stage.
#   S8a runs/era5_64x32_s8a_ode       integrator=ode, K=2 (12 h), Euler, no drift      (compare with S3)
#   S8b runs/era5_64x32_s8b_ode_phys  + physics drift = S3 guide's 6 h tendency head  (compare with S8a)
# Both reuse the S3 guide (GUIDE_FROM, Stage A skipped) and the S3 recipe (gated fusion, 2-step rollout from 500).
cd "$(dirname "$0")/../.."
S3=runs/era5_64x32_physops_roll
LOG=runs/era5_64x32_s8a_ode/chain_s8.log
until grep -q CHAIN_S7_DONE runs/era5_64x32_s7_data10y/chain_data.log 2>/dev/null; do sleep 60; done
echo "[$(date -u +%H:%M:%S)] S7 chain finished -> S8a" >> $LOG
RUN=runs/era5_64x32_s8a_ode BASE=$S3 GUIDE_FROM=$S3 COPY_NOGUIDE=0 INTEGRATOR=ode ODE_STEPS=2 ODE_SOLVER=euler \
  ODE_PHYSICS=0 ROLLOUT_STEPS=2 ROLLOUT_START=500 bash scripts/run_guide_ablation.sh > runs/era5_64x32_s8a_ode/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] S8a exit $? -> S8b" >> $LOG
mkdir -p runs/era5_64x32_s8b_ode_phys
RUN=runs/era5_64x32_s8b_ode_phys BASE=$S3 GUIDE_FROM=$S3 COPY_NOGUIDE=0 INTEGRATOR=ode ODE_STEPS=2 ODE_SOLVER=euler \
  ODE_PHYSICS=1 ROLLOUT_STEPS=2 ROLLOUT_START=500 bash scripts/run_guide_ablation.sh > runs/era5_64x32_s8b_ode_phys/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] S8b exit $? CHAIN_S8_DONE" >> $LOG
