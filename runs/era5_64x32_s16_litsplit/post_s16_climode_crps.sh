#!/usr/bin/env bash
# stage16 add-on (2026-11-07 19:20 KST): ClimODE probabilistic scores on 2017-2018 for the same 24 ensemble init times
# as stage16's eval (cases[:24]), 8 members from the emission Gaussian, authors' protocol (train mode, 2 ICs/batch).
# Evaluation only; runs after the stage16 chain so it does not compete for the 4 CPU cores.
cd "$(dirname "$0")/../.."
LOG=runs/era5_64x32_s16_litsplit/chain_s16.log
until grep -q "CHAIN_S16_DONE" $LOG 2>/dev/null; do sleep 120; done
python -W ignore scripts/eval_climode.py --data data/era5_wb2_64x32_2006_2018.npz \
  --out runs/climode_1718_trainmode_crps --train-mode --batch-ics 2 --threads 4 --max-cases 24 --members 8 \
  > runs/era5_64x32_s16_litsplit/climode_1718_crps.log 2>&1
echo "[$(date -u +%H:%M:%S)] ClimODE 2017-2018 CRPS exit $? POST_S16_DONE" >> $LOG
