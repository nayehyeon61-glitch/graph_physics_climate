#!/usr/bin/env bash
# Stage stage18b (2026-11-07 21:45 KST): path memory — the third static part of the dynamic guide (감자 2026-11-07
# 20:31 KST).  stage18 + the previous-state input follows the ODE path: sub-step k of a lead gets the state one lead
# before its own time, X_{t + 6k h - 24 h} (observed past states in the first lead, the previous lead's path later),
# instead of X_{t - 24 h} for all four sub-steps.  The guide-free pair sees the same input, so it is fine-tuned again
# (1000 steps from stage9b's guide-free V-Net).  Same guide (stage17c's) and paired fine-tuning as stage18.
# Compare with stage18 (+ stage17a, ClimODE, 2020/2022).  Design: docs/design/Stage18_Graph_Tendency_Core_KO.md §5
cd "$(dirname "$0")/../.."
S9B=runs/era5_64x32_s9b_merged
S18=runs/era5_64x32_s18_core
RUN=runs/era5_64x32_s18b_prevpath
LOG=$RUN/chain_s18b.log
echo "[$(date -u +%H:%M:%S)] stage18b waits for CHAIN_S18_DONE" >> $LOG
until grep -q "CHAIN_S18_DONE" $S18/chain_s18.log 2>/dev/null; do sleep 120; done
echo "[$(date -u +%H:%M:%S)] stage18b start" >> $LOG
for f in normalizer.pt column_norm.json; do [ -f $RUN/$f ] || cp $S9B/$f $RUN/; done
for f in guide.pt a_done.json log_a.jsonl; do [ -f $RUN/$f ] || cp $S18/$f $RUN/; done
[ -f $RUN/config.json ] || python - "$S18/config.json" "$RUN/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c.update(prev_path=1)
json.dump(c, open(sys.argv[2], "w"), indent=2)
PY
FT="--threads 4 --batch 4 --lr 3e-4 --steps 1000 --val-every 250 --rollout-steps 2 --rollout-start 0 --rollout-weight 1.0"
EV="--threads 4 --stage eval --eval-stride 16 --max-lead-steps 5"
{
python -W ignore scripts/train_eval.py --run $RUN $FT --stage b_noguide --init-vnet $S9B/vnet_noguide.pt &&
python -W ignore scripts/train_eval.py --run $RUN $FT --stage b --init-vnet $S9B/vnet.pt &&
python -W ignore scripts/train_eval.py --run $RUN $EV --n-ens-cases 0 &&
python -W ignore scripts/diagnose_guide.py --run $RUN --threads 4 && echo PIPELINE_DONE
} >> $RUN/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] stage18b pipeline exit $?" >> $LOG
for Y in 2020 2022; do
  python -W ignore scripts/train_eval.py --run $RUN $EV --n-ens-cases 0 --eval-data data/era5_wb2_64x32_${Y}test.npz \
    --eval-column-data data/era5_wb2_64x32_column10_${Y}test.npz --eval-tag $Y --max-cases 88 > $RUN/eval_$Y.log 2>&1
  echo "[$(date -u +%H:%M:%S)] stage18b eval_$Y exit $?" >> $LOG
done
if python scripts/decide_adoption.py $RUN/eval.json $S18/eval.json --min-mean-gain 0.5 >> $LOG; then
  echo "stage18b ADOPTED (vs stage18)" >> $LOG
else
  echo "stage18b REJECTED (vs stage18)" >> $LOG
fi
echo "[$(date -u +%H:%M:%S)] stage18b done CHAIN_S18B_DONE" >> $LOG
