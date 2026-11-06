#!/usr/bin/env bash
# Stage stage15b (2026-11-07 03:40 KST): intermediate-time loss on top of the integrated model stage9b (감자 2026-11-05
# 21:25 KST "weather ode를 이길려면 어떤 단계를 추가하면" -> proposed stage15a/15b/9b; stage9b result 11-07 03:15 KST left
# the gap at 24 h; decision card 03:25 KST, recommended option started while waiting).
# One change vs stage9b: an auxiliary head on the shared V-Net decoder embedding predicts X_{t+6h}, X_{t+12h},
# X_{t+18h} inside the first 24 h lead, trained with the mean MSE of the three (same delta units as L_f, weight 1.0).
# Forecasts, rollout, data years, steps (9000), guide, column inputs, Stage C are stage9b's; the head is not used at
# inference.  WeatherODE (arXiv 2410.06560) supervises every intermediate hour; this is the solver-free version.
# The guide-free pair is retrained with the same loss.  Compare with stage9b (+ its guide-free pair, ClimODE),
# adoption rule scripts/decide_adoption.py (>= 3/5 leads, mean >= 0.5 %).  Design: docs/design/Stage15b_Intermediate_Loss_KO.md
cd "$(dirname "$0")/../.."
S9B=runs/era5_64x32_s9b_merged
RUN=runs/era5_64x32_s15b_inter
LOG=$RUN/chain_s15b.log
echo "[$(date -u +%H:%M:%S)] stage15b intermediate-time loss start" >> $LOG
for f in normalizer.pt guide.pt a_done.json column_norm.json; do [ -f $RUN/$f ] || cp $S9B/$f $RUN/; done
[ -f "$RUN/config.json" ] || python - "$S9B/config.json" "$RUN/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c.update(inter_times=3, inter_weight=1.0)
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
echo "[$(date -u +%H:%M:%S)] stage15b pipeline exit $?" >> $LOG
for Y in 2020 2022; do
  python -W ignore scripts/train_eval.py --run $RUN --threads 4 --stage eval --eval-stride 16 --max-lead-steps 5 \
    --n-ens-cases 0 --eval-data data/era5_wb2_64x32_${Y}test.npz \
    --eval-column-data data/era5_wb2_64x32_column10_${Y}test.npz --eval-tag $Y --max-cases 88 > $RUN/eval_$Y.log 2>&1
  echo "[$(date -u +%H:%M:%S)] stage15b eval_$Y exit $?" >> $LOG
done
[ -f $RUN/eval.json ] && python scripts/decide_adoption.py $RUN/eval.json $S9B/eval.json --min-mean-gain 0.5 >> $LOG
echo "[$(date -u +%H:%M:%S)] stage15b done CHAIN_S15B_DONE" >> $LOG
