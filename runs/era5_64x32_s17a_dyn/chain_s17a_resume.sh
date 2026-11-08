#!/usr/bin/env bash
# stage17a resume (2026-11-08 13:10 KST): the guided ODE Stage B of chain_s17a.sh was OOM-killed at its first step
# (14 GB at batch 4, rollout 2 x 4 sub-steps, 16 GB machine) and the chain was stopped by hand before it could write a
# decision or CHAIN_S17A_DONE.  Fix: per-sub-step gradient checkpointing in the ODE integrator (VNetConfig.ode_checkpoint,
# default on; same numbers, tests/test_ode_checkpoint.py).  The control (both arms) and stage17a's guide-free pair are
# finished and kept; this script runs the remaining stage17a steps with the original flags and writes the marker that
# chain_s17c.sh waits for into the same log.
cd "$(dirname "$0")/../.."
S9B=runs/era5_64x32_s9b_merged
CTL=runs/era5_64x32_s17_ctl
RUN=runs/era5_64x32_s17a_dyn
LOG=$RUN/chain_s17a.log
echo "[$(date -u +%H:%M:%S)] stage17a resume (guided Stage B with ODE checkpointing)" >> $LOG
FT="--threads 4 --batch 4 --lr 3e-4 --steps 1000 --val-every 250 --rollout-steps 2 --rollout-start 0 --rollout-weight 1.0"
EV="--threads 4 --stage eval --eval-stride 16 --max-lead-steps 5"
D=$RUN
{
python -W ignore scripts/train_eval.py --run $D $FT --stage b_noguide --init-vnet $S9B/vnet_noguide.pt &&
python -W ignore scripts/train_eval.py --run $D $FT --stage b --init-vnet $S9B/vnet.pt &&
python -W ignore scripts/train_eval.py --run $D $EV --n-ens-cases 0 &&
python -W ignore scripts/diagnose_guide.py --run $D --threads 4 && echo PIPELINE_DONE
} >> $D/pipeline.log 2>&1
RC=$?
echo "[$(date -u +%H:%M:%S)] $D pipeline exit $RC" >> $LOG
if [ $RC -ne 0 ] || [ ! -f $D/eval.json ]; then
  echo "[$(date -u +%H:%M:%S)] stage17a resume FAILED (no eval.json); marker not written" >> $LOG
  exit 1
fi
for Y in 2020 2022; do
  python -W ignore scripts/train_eval.py --run $D $EV --n-ens-cases 0 --eval-data data/era5_wb2_64x32_${Y}test.npz \
    --eval-column-data data/era5_wb2_64x32_column10_${Y}test.npz --eval-tag $Y --max-cases 88 > $D/eval_$Y.log 2>&1
  echo "[$(date -u +%H:%M:%S)] $D eval_$Y exit $?" >> $LOG
done
python scripts/decide_adoption.py $RUN/eval.json $CTL/eval.json --min-mean-gain 0.5 >> $LOG
echo "[$(date -u +%H:%M:%S)] stage17a done CHAIN_S17A_DONE" >> $LOG
