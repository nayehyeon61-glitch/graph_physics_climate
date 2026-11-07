#!/usr/bin/env bash
# Stage stage18c (2026-11-07 21:45 KST): guide along the path — the first static part of the dynamic guide (감자
# 2026-11-07 20:31 KST: guide weights learnt on single snapshots and frozen while the forecast is trained).  The better
# of stage18 / stage18b (chain decision) + Stage B also fine-tunes the guide's physics heads (grid adapter, flow,
# source; --guide-heads-only) along the core's forecast path with a small lr (3e-5, V-Net 3e-4); the guide trunk and
# hence the features P stay frozen, because stage12c showed that fine-tuning the whole guide (lr 1e-4) turned it into a
# second forecaster (-4.3 % at 24 h).  Inference uses guide_finetuned.pt.  Same paired fine-tuning; the guide-free
# pair is the base's.  Compare with the base (+ stage17a, ClimODE, 2020/2022).
# Design: docs/design/Stage18_Graph_Tendency_Core_KO.md §5
cd "$(dirname "$0")/../.."
S9B=runs/era5_64x32_s9b_merged
S18=runs/era5_64x32_s18_core
S18B=runs/era5_64x32_s18b_prevpath
RUN=runs/era5_64x32_s18c_guidepath
LOG=$RUN/chain_s18c.log
echo "[$(date -u +%H:%M:%S)] stage18c waits for CHAIN_S18B_DONE" >> $LOG
until grep -q "CHAIN_S18B_DONE" $S18B/chain_s18b.log 2>/dev/null; do sleep 120; done
if grep -q "stage18b ADOPTED" $S18B/chain_s18b.log; then BASE=$S18B; else BASE=$S18; fi
echo "[$(date -u +%H:%M:%S)] stage18c start on $BASE" >> $LOG
for f in normalizer.pt column_norm.json guide.pt a_done.json log_a.jsonl config.json vnet_noguide.pt \
         vnet_noguide_done.json log_vnet_noguide.jsonl; do
  [ -f $RUN/$f ] || cp $BASE/$f $RUN/
done
FT="--threads 4 --batch 4 --lr 3e-4 --steps 1000 --val-every 250 --rollout-steps 2 --rollout-start 0 --rollout-weight 1.0"
EV="--threads 4 --stage eval --eval-stride 16 --max-lead-steps 5"
{
python -W ignore scripts/train_eval.py --run $RUN $FT --stage b --init-vnet $S9B/vnet.pt --guide-lr 3e-5 \
  --guide-heads-only 1 &&
python -W ignore scripts/train_eval.py --run $RUN $EV --n-ens-cases 0 &&
python -W ignore scripts/diagnose_guide.py --run $RUN --threads 4 && echo PIPELINE_DONE
} >> $RUN/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] stage18c pipeline exit $?" >> $LOG
for Y in 2020 2022; do
  python -W ignore scripts/train_eval.py --run $RUN $EV --n-ens-cases 0 --eval-data data/era5_wb2_64x32_${Y}test.npz \
    --eval-column-data data/era5_wb2_64x32_column10_${Y}test.npz --eval-tag $Y --max-cases 88 > $RUN/eval_$Y.log 2>&1
  echo "[$(date -u +%H:%M:%S)] stage18c eval_$Y exit $?" >> $LOG
done
echo "vs base $BASE:" >> $LOG
python scripts/decide_adoption.py $RUN/eval.json $BASE/eval.json --min-mean-gain 0.5 >> $LOG
echo "[$(date -u +%H:%M:%S)] stage18c done CHAIN_S18C_DONE" >> $LOG
