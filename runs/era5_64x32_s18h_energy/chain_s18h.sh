#!/usr/bin/env bash
# Stage stage18h (2026-11-10 15:30 KST): energy-based Delaunay/Voronoi partition of the graph tendency core (감자
# 2026-11-10 14:40 KST: "들로네, 보르노이가 단순히 격자가 아닌 에너지 base로 나눌 수 있을까요?").  The power diagram of
# stage18g is re-drawn every 6 h sub-step from the energy of the current state, a_i = -sum_c alpha_c log(e_c,i/<e_c>
# + 0.1) (+ the stage18g static weights if the parent has them), components: kinetic energy at 850 / 500 hPa and 10 m,
# APE proxy (t850 - zonal mean)^2, latent proxy q700, kinetic-energy rate of the guide's core terms; alpha_c learned
# (gain 4, zero-initialised = the parent).  The partition changes in time, so the operators keep the grid areas (exact
# conservation of the grid integral): only face lengths and l / d move.  Parent = stage18g if adopted else stage18g's
# parent; config = parent + core_energy_geom.  Same paired fine-tuning (V-Net from stage9b, 1000 steps); guide and pair
# from the parent.  Design: docs/design/Stage18h_Energy_Partition_KO.md
cd "$(dirname "$0")/../.."
S9B=runs/era5_64x32_s9b_merged
S18G=runs/era5_64x32_s18g_geom
RUN=runs/era5_64x32_s18h_energy
LOG=$RUN/chain_s18h.log
echo "[$(date -u +%H:%M:%S)] stage18h waits for CHAIN_S18G_DONE" >> $LOG
until grep -q "CHAIN_S18G_DONE" $S18G/chain_s18g.log 2>/dev/null; do sleep 120; done
BASE=$(cat $S18G/base.txt); BFLAGS=$(cat $S18G/bflags.txt)
if grep -q "stage18g ADOPTED" $S18G/chain_s18g.log; then PARENT=$S18G; else PARENT=$(cat $S18G/parent.txt); fi
echo "$PARENT" > $RUN/parent.txt
echo "[$(date -u +%H:%M:%S)] stage18h start: $PARENT config + core_energy_geom, base $BASE ($BFLAGS)" >> $LOG
for f in normalizer.pt column_norm.json guide.pt a_done.json log_a.jsonl vnet_noguide.pt vnet_noguide_done.json \
         log_vnet_noguide.jsonl tv_eps.json; do
  if [ ! -f $RUN/$f ] && [ -f $PARENT/$f ]; then cp $PARENT/$f $RUN/; fi
done
[ -f $RUN/config.json ] || python - "$PARENT/config.json" "$RUN/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c.update(core_energy_geom=1)
json.dump(c, open(sys.argv[2], "w"), indent=2)
PY
FT="--threads 4 --batch 4 --lr 3e-4 --steps 1000 --val-every 250 --rollout-steps 2 --rollout-start 0 --rollout-weight 1.0"
EV="--threads 4 --stage eval --eval-stride 16 --max-lead-steps 5"
{
python -W ignore scripts/train_eval.py --run $RUN $FT --stage b --init-vnet $S9B/vnet.pt $BFLAGS &&
python -W ignore scripts/train_eval.py --run $RUN $EV --n-ens-cases 0 &&
python -W ignore scripts/diagnose_guide.py --run $RUN --threads 4 && echo PIPELINE_DONE
} >> $RUN/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] stage18h pipeline exit $?" >> $LOG
for Y in 2020 2022; do
  python -W ignore scripts/train_eval.py --run $RUN $EV --n-ens-cases 0 --eval-data data/era5_wb2_64x32_${Y}test.npz \
    --eval-column-data data/era5_wb2_64x32_column10_${Y}test.npz --eval-tag $Y --max-cases 88 > $RUN/eval_$Y.log 2>&1
  echo "[$(date -u +%H:%M:%S)] stage18h eval_$Y exit $?" >> $LOG
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
  echo "stage18h ADOPTED" >> $LOG
  echo "$RUN" > $RUN/base.txt
else
  echo "stage18h REJECTED" >> $LOG
  echo "$BASE" > $RUN/base.txt
fi
echo "$BFLAGS" > $RUN/bflags.txt
echo "[$(date -u +%H:%M:%S)] stage18h done CHAIN_S18H_DONE" >> $LOG
