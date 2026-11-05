#!/usr/bin/env bash
# Stage stage9b (2026-11-05 23:55 KST): merge into stage9 whatever of stage7c / stage15a improves (감자 2026-11-05
# 23:51 KST "이것들도 각각 더 좋아지면 9단계에 합쳐주세요").
#   stage7c  = step budget (Stage B 4500 -> 9000, rollout start 1500 -> 3000) on the stage7b structure, vs stage7b
#   stage15a = stage9 + column inputs (fcn10, initial time only), vs stage9
# Rule (scripts/decide_adoption.py): 11-variable mean RMSE better at >= 3 of 5 leads and mean gain >= 0.5 %.
#   neither    -> stage9 stays the integrated model, nothing to run
#   15a only   -> stage15a already IS stage9 + columns, nothing to run
#   7c (+15a)  -> stage9b = stage9 (+ columns) with the 9000-step budget: guide/normaliser of stage9 (the step budget
#                 is Stage B's; stage7c also reused its guide), guide-free pair retrained with the same budget,
#                 Stage C, eval, diagnose, 2020/2022 benchmark years.  Compare with stage9 / stage15a + ClimODE.
cd "$(dirname "$0")/../.."
S9=runs/era5_64x32_s9_integrated
S7B=runs/era5_64x32_s7b_steps3x
S7C=runs/era5_64x32_s7c_steps6x
S15A=runs/era5_64x32_s15a_column
RUN=runs/era5_64x32_s9b_merged
LOG=$RUN/chain_s9b.log
echo "[$(date -u +%H:%M:%S)] stage9b waits for CHAIN_S15A_DONE" >> $LOG
until grep -q "CHAIN_S15A_DONE" $S15A/chain_s15a.log 2>/dev/null; do sleep 120; done
STEPS=0
COL=0
if [ -f $S7C/eval.json ] && python scripts/decide_adoption.py $S7C/eval.json $S7B/eval.json --min-mean-gain 0.5 >> $LOG; then STEPS=1; fi
if [ -f $S15A/eval.json ] && python scripts/decide_adoption.py $S15A/eval.json $S9/eval.json --min-mean-gain 0.5 >> $LOG; then COL=1; fi
echo "{\"steps_9000\": $STEPS, \"column_inputs\": $COL}" > $RUN/merge_decision.json
echo "[$(date -u +%H:%M:%S)] merge decision: steps_9000=$STEPS column_inputs=$COL" >> $LOG
if [ $STEPS -eq 0 ]; then
  echo "[$(date -u +%H:%M:%S)] no step-budget gain -> no stage9b run (integrated model = stage15a if column_inputs=1, else stage9) CHAIN_S9B_DONE" >> $LOG
  exit 0
fi
[ -f $RUN/normalizer.pt ] || cp $S9/normalizer.pt $RUN/
[ -f $RUN/guide.pt ] || cp $S9/guide.pt $RUN/
[ -f $RUN/a_done.json ] || cp $S9/a_done.json $RUN/
[ -f "$RUN/config.json" ] || python - "$S9/config.json" "$RUN/config.json" "$COL" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
if sys.argv[3] == "1":
    c.update(column_inputs="fcn10", column_data="data/era5_wb2_64x32_column10_2008_2019.npz")
json.dump(c, open(sys.argv[2], "w"), indent=2)
PY
COMMON="--run $RUN --threads 4 --batch 4 --lr 1e-3"
B="--steps 9000 --val-every 250 --rollout-steps 2 --rollout-start 3000 --rollout-weight 1.0"
{
python -W ignore scripts/train_eval.py $COMMON --stage b_noguide $B &&
python -W ignore scripts/train_eval.py $COMMON --stage b $B &&
python -W ignore scripts/train_eval.py $COMMON --stage c --steps 800 --val-every 200 --noise-constraints 1 \
  --noise-water 1 --noise-feas-weight 1.0 --noise-mean-weight 1.0 --noise-mean-every 5 --noise-mean-members 2 \
  --noise-mean-steps 4 &&
python -W ignore scripts/train_eval.py --run $RUN --threads 4 --stage eval --eval-stride 16 --max-lead-steps 5 \
  --n-ens-cases 24 --members 8 --sample-steps 20 &&
python -W ignore scripts/diagnose_guide.py --run $RUN --threads 4 && echo PIPELINE_DONE
} >> $RUN/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] stage9b pipeline exit $?" >> $LOG
for Y in 2020 2022; do
  EXTRA=""
  [ $COL -eq 1 ] && EXTRA="--eval-column-data data/era5_wb2_64x32_column10_${Y}test.npz"
  python -W ignore scripts/train_eval.py --run $RUN --threads 4 --stage eval --eval-stride 16 --max-lead-steps 5 \
    --n-ens-cases 0 --eval-data data/era5_wb2_64x32_${Y}test.npz $EXTRA --eval-tag $Y --max-cases 88 \
    > $RUN/eval_$Y.log 2>&1
  echo "[$(date -u +%H:%M:%S)] stage9b eval_$Y exit $?" >> $LOG
done
echo "[$(date -u +%H:%M:%S)] stage9b done CHAIN_S9B_DONE" >> $LOG
