#!/usr/bin/env bash
# Stage stage18g (2026-11-09 14:45 KST): dual-space geometry theta_g (same request as stage18f; decision card 14:14 KST
# "18f 뒤 항상 실행").  A power diagram of the grid points with learned static cell weights a_i = MLP(terrain, land-sea
# mask, latitude), zero-initialised (the grid itself): faces move by 0.25 tanh(a_i - a_j) of the centre distance
# towards the lighter cell; cell areas, face lengths, covering weights and face masses are recomputed and every core
# operator (flux divergence, Laplace block) uses the new geometry, so the discrete conservation holds for the new
# areas.  Parent = stage18f if adopted else stage18f's parent; config = parent + core_geometry.  Same paired
# fine-tuning (V-Net from stage9b, 1000 steps); guide and pair from the chain base.  Runs whatever stage18f decided
# (theta_g is a different kind of parameter).  Design: docs/design/Stage18f_Dual_Space_Core_KO.md
cd "$(dirname "$0")/../.."
S9B=runs/era5_64x32_s9b_merged
S18F=runs/era5_64x32_s18f_edge
RUN=runs/era5_64x32_s18g_geom
LOG=$RUN/chain_s18g.log
echo "[$(date -u +%H:%M:%S)] stage18g waits for CHAIN_S18F_DONE" >> $LOG
until grep -q "CHAIN_S18F_DONE" $S18F/chain_s18f.log 2>/dev/null; do sleep 120; done
BASE=$(cat $S18F/base.txt); BFLAGS=$(cat $S18F/bflags.txt)
if grep -q "stage18f ADOPTED" $S18F/chain_s18f.log; then PARENT=$S18F; else PARENT=$(cat $S18F/parent.txt); fi
echo "$PARENT" > $RUN/parent.txt
echo "[$(date -u +%H:%M:%S)] stage18g start: $PARENT config + core_geometry, base $BASE ($BFLAGS)" >> $LOG
for f in normalizer.pt column_norm.json guide.pt a_done.json log_a.jsonl vnet_noguide.pt vnet_noguide_done.json \
         log_vnet_noguide.jsonl tv_eps.json; do
  if [ ! -f $RUN/$f ] && [ -f $PARENT/$f ]; then cp $PARENT/$f $RUN/; fi
done
[ -f $RUN/config.json ] || python - "$PARENT/config.json" "$RUN/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c.update(core_geometry=1)
json.dump(c, open(sys.argv[2], "w"), indent=2)
PY
FT="--threads 4 --batch 4 --lr 3e-4 --steps 1000 --val-every 250 --rollout-steps 2 --rollout-start 0 --rollout-weight 1.0"
EV="--threads 4 --stage eval --eval-stride 16 --max-lead-steps 5"
{
python -W ignore scripts/train_eval.py --run $RUN $FT --stage b --init-vnet $S9B/vnet.pt $BFLAGS &&
python -W ignore scripts/train_eval.py --run $RUN $EV --n-ens-cases 0 &&
python -W ignore scripts/diagnose_guide.py --run $RUN --threads 4 && echo PIPELINE_DONE
} >> $RUN/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] stage18g pipeline exit $?" >> $LOG
for Y in 2020 2022; do
  python -W ignore scripts/train_eval.py --run $RUN $EV --n-ens-cases 0 --eval-data data/era5_wb2_64x32_${Y}test.npz \
    --eval-column-data data/era5_wb2_64x32_column10_${Y}test.npz --eval-tag $Y --max-cases 88 > $RUN/eval_$Y.log 2>&1
  echo "[$(date -u +%H:%M:%S)] stage18g eval_$Y exit $?" >> $LOG
done
# one change per stage: decided against the parent; when the parent is not the chain base, also against the base
echo "vs parent $PARENT:" >> $LOG
OKP=0
if python scripts/decide_adoption.py $RUN/eval.json $PARENT/eval.json --min-mean-gain 0.5 >> $LOG; then OKP=1; fi
OKB=$OKP
if [ "$BASE" != "$PARENT" ]; then
  echo "vs base $BASE:" >> $LOG
  OKB=0
  if python scripts/decide_adoption.py $RUN/eval.json $BASE/eval.json --min-mean-gain 0.5 >> $LOG; then OKB=1; fi
fi
if [ $OKP = 1 ] && [ $OKB = 1 ]; then
  echo "stage18g ADOPTED" >> $LOG
  echo "$RUN" > $RUN/base.txt
else
  echo "stage18g REJECTED" >> $LOG
  echo "$BASE" > $RUN/base.txt
fi
echo "$BFLAGS" > $RUN/bflags.txt
echo "[$(date -u +%H:%M:%S)] stage18g done CHAIN_S18G_DONE" >> $LOG
