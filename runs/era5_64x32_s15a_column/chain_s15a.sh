#!/usr/bin/env bash
# Stage stage15a (2026-11-05 21:50 KST): column inputs (감자 2026-11-05 21:25 KST "weather ode를 이길려면 어떤 단계를
# 추가하면 좋을것 같나요?" -> decision card 21:34 KST "stage15a 기둥 입력부터", after stage7c).
# One change vs stage9: the V-Net also sees the 10 FourCastNet column channels missing from base11 (sp, z1000, z850,
# t500, z50, r500, r850, tcwv, u/v1000) at the initial time only (later rollout steps: zeros + availability flag 0;
# future columns are never fed).  Outputs, loss, data years, steps, guide, Stage C are stage9's.  The guide does not
# see the columns, so stage9's guide (and its Stage A marker) is reused unchanged: the comparison isolates the V-Net
# input.  The guide-free pair is retrained with the same column inputs.  Compare with stage9 (+ its guide-free pair,
# ClimODE).  Design: docs/design/Stage15a_Column_Inputs_KO.md
cd "$(dirname "$0")/../.."
S9=runs/era5_64x32_s9_integrated
S7C=runs/era5_64x32_s7c_steps6x
RUN=runs/era5_64x32_s15a_column
LOG=$RUN/chain_s15a.log
echo "[$(date -u +%H:%M:%S)] stage15a waits for CHAIN_S7C_DONE" >> $LOG
until grep -q "CHAIN_S7C_DONE" $S7C/chain_s7c.log 2>/dev/null; do sleep 120; done
echo "[$(date -u +%H:%M:%S)] stage15a column inputs start" >> $LOG
[ -f $RUN/normalizer.pt ] || cp $S9/normalizer.pt $RUN/
[ -f $RUN/guide.pt ] || cp $S9/guide.pt $RUN/
[ -f $RUN/a_done.json ] || cp $S9/a_done.json $RUN/
[ -f "$RUN/config.json" ] || python - "$S9/config.json" "$RUN/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c.update(column_inputs="fcn10", column_data="data/era5_wb2_64x32_column10_2008_2019.npz")
json.dump(c, open(sys.argv[2], "w"), indent=2)
PY
COMMON="--run $RUN --threads 4 --batch 4 --lr 1e-3"
B="--steps 4500 --val-every 250 --rollout-steps 2 --rollout-start 1500 --rollout-weight 1.0"
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
echo "[$(date -u +%H:%M:%S)] stage15a pipeline exit $?" >> $LOG
# benchmark years (same protocol as runs/bench_wb2_2020 / 2022)
for Y in 2020 2022; do
  python -W ignore scripts/train_eval.py --run $RUN --threads 4 --stage eval --eval-stride 16 --max-lead-steps 5 \
    --n-ens-cases 0 --eval-data data/era5_wb2_64x32_${Y}test.npz \
    --eval-column-data data/era5_wb2_64x32_column10_${Y}test.npz --eval-tag $Y --max-cases 88 \
    > $RUN/eval_$Y.log 2>&1
  echo "[$(date -u +%H:%M:%S)] stage15a eval_$Y exit $?" >> $LOG
done
echo "[$(date -u +%H:%M:%S)] stage15a done CHAIN_S15A_DONE" >> $LOG
