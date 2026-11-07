#!/usr/bin/env bash
# Stage stage17b (2026-11-07 19:55 KST): stage17a + the guide's own 6 h tendency as a drift term of the ODE, with a
# learnable per-variable gain initialised at 0 (so training starts exactly at stage17a's dynamics and the learned gain
# says how much of the guide's tendency the dynamics keep).  Same paired fine-tuning (1000 steps from stage9b);
# the guide-free pair has no guide and therefore no drift, so stage17a's pair is reused.  Compare with stage17a
# (+ stage17-ctl, ClimODE, 2020/2022) and report the learned gains and the physics-only forecast row.
cd "$(dirname "$0")/../.."
S9B=runs/era5_64x32_s9b_merged
S17A=runs/era5_64x32_s17a_dyn
RUN=runs/era5_64x32_s17b_drift
LOG=$RUN/chain_s17b.log
echo "[$(date -u +%H:%M:%S)] stage17b waits for CHAIN_S17A_DONE" >> $LOG
until grep -q "CHAIN_S17A_DONE" $S17A/chain_s17a.log 2>/dev/null; do sleep 120; done
echo "[$(date -u +%H:%M:%S)] stage17b start" >> $LOG
for f in normalizer.pt guide.pt a_done.json column_norm.json vnet_noguide.pt vnet_noguide_done.json log_vnet_noguide.jsonl; do
  [ -f $RUN/$f ] || cp $S17A/$f $RUN/
done
[ -f $RUN/config.json ] || python - "$S17A/config.json" "$RUN/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c.update(ode_physics=1, ode_drift_gain=1)
json.dump(c, open(sys.argv[2], "w"), indent=2)
PY
FT="--threads 4 --batch 4 --lr 3e-4 --steps 1000 --val-every 250 --rollout-steps 2 --rollout-start 0 --rollout-weight 1.0"
EV="--threads 4 --stage eval --eval-stride 16 --max-lead-steps 5"
{
python -W ignore scripts/train_eval.py --run $RUN $FT --stage b --init-vnet $S9B/vnet.pt &&
python -W ignore scripts/train_eval.py --run $RUN $EV --n-ens-cases 0 &&
python -W ignore scripts/diagnose_guide.py --run $RUN --threads 4 && echo PIPELINE_DONE
} >> $RUN/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] stage17b pipeline exit $?" >> $LOG
for Y in 2020 2022; do
  python -W ignore scripts/train_eval.py --run $RUN $EV --n-ens-cases 0 --eval-data data/era5_wb2_64x32_${Y}test.npz \
    --eval-column-data data/era5_wb2_64x32_column10_${Y}test.npz --eval-tag $Y --max-cases 88 > $RUN/eval_$Y.log 2>&1
  echo "[$(date -u +%H:%M:%S)] stage17b eval_$Y exit $?" >> $LOG
done
python scripts/decide_adoption.py $RUN/eval.json $S17A/eval.json --min-mean-gain 0.5 >> $LOG
echo "[$(date -u +%H:%M:%S)] stage17b done CHAIN_S17B_DONE" >> $LOG
