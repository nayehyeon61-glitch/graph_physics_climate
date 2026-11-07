#!/usr/bin/env bash
# Stage stage16 (2026-11-07 17:25 KST): literature-split retrain of the stage9b recipe (SOTA comparison step 3,
# reports/SOTA_COMPARISON_KO.md).  Same structure/recipe as stage9b (stage9 + column inputs fcn10 + Stage B 9000
# steps, rollout 2 from 3000), but trained on the split of the 5.625 deg papers: train 2006-2015, validation 2016,
# test 2017-2018.  Everything that saw 2016-2017 in stage9b is refitted from scratch (normaliser, column stats,
# guide-free pair = Stage A perturbation model, Stage A guide, Stage B, Stage C), so 2017-2018 stays unseen.
#   main comparison : stage16 vs ClimODE (published checkpoint, trained 2006-2015) on 2017-2018, same init times
#                     + direct RMSE against the paper tables (WeatherODE, ClimaX, FourCastNet, CoDiCast, ...)
#   previous stage  : stage16 vs stage9b on 2019 (unseen by both; only the training years differ)
# Design: docs/design/Stage16_Literature_Split_KO.md
cd "$(dirname "$0")/../.."
RUN=runs/era5_64x32_s16_litsplit
S9B=runs/era5_64x32_s9b_merged
D=data/era5_wb2_64x32_2006_2018.npz
DC=data/era5_wb2_64x32_column10_2006_2018.npz
LOG=$RUN/chain_s16.log
echo "[$(date -u +%H:%M:%S)] stage16 waits for FETCH_S16_DONE" >> $LOG
until grep -q "FETCH_S16_DONE" runs/s16_prep/fetch_litsplit.log 2>/dev/null; do sleep 60; done
echo "[$(date -u +%H:%M:%S)] stage16 start" >> $LOG
[ -f "$RUN/config.json" ] || python - "$S9B/config.json" "$RUN/config.json" "$D" "$DC" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c.update(data=sys.argv[3], column_data=sys.argv[4])
json.dump(c, open(sys.argv[2], "w"), indent=2)
PY
COMMON="--run $RUN --threads 4 --batch 4 --lr 1e-3"
B="--steps 9000 --val-every 250 --rollout-steps 2 --rollout-start 3000 --rollout-weight 1.0"
{
python -W ignore scripts/train_eval.py $COMMON --stage b_noguide $B &&
python -W ignore scripts/train_eval.py $COMMON --stage a --steps 1000 --val-every 200 \
  --perturb-model $RUN/vnet_noguide.pt --perturb-prob 0.3 &&
python -W ignore scripts/train_eval.py $COMMON --stage b $B &&
python -W ignore scripts/train_eval.py $COMMON --stage c --steps 800 --val-every 200 --noise-constraints 1 \
  --noise-water 1 --noise-feas-weight 1.0 --noise-mean-weight 1.0 --noise-mean-every 5 --noise-mean-members 2 \
  --noise-mean-steps 4 &&
python -W ignore scripts/train_eval.py --run $RUN --threads 4 --stage eval --eval-stride 16 --max-lead-steps 5 \
  --n-ens-cases 24 --members 8 --sample-steps 20 &&
python -W ignore scripts/diagnose_guide.py --run $RUN --threads 4 && echo PIPELINE_DONE
} >> $RUN/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] stage16 pipeline exit $?" >> $LOG
# ClimODE on the same 2017-2018 init times (authors' protocol: train mode, 2 ICs per batch)
[ -f runs/climode_1718_trainmode/eval.json ] || python -W ignore scripts/eval_climode.py --data $D \
  --out runs/climode_1718_trainmode --train-mode --batch-ics 2 --threads 4 --members 0 > $RUN/climode_1718.log 2>&1
echo "[$(date -u +%H:%M:%S)] ClimODE 2017-2018 exit $?" >> $LOG
# previous-stage comparison on 2019 (unseen by stage16 and stage9b)
python -W ignore scripts/train_eval.py --run $RUN --threads 4 --stage eval --eval-stride 16 --max-lead-steps 5 \
  --n-ens-cases 0 --eval-data data/era5_wb2_64x32_2008_2019.npz \
  --eval-column-data data/era5_wb2_64x32_column10_2008_2019.npz --eval-tag 2019 > $RUN/eval_2019.log 2>&1
echo "[$(date -u +%H:%M:%S)] stage16 eval_2019 exit $?" >> $LOG
echo "[$(date -u +%H:%M:%S)] stage16 done CHAIN_S16_DONE" >> $LOG
