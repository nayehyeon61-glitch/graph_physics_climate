#!/usr/bin/env bash
# Stage stage21a / stage21b (2026-11-09 15:07 KST, 감자: "우리의 목적은 학습데이터보다는 모델의 즉 물리적 graph에 있는데 이를
# 최대로 할려면?" -> "오케이 그럼 1+2로 진행해보면?"): physics-first core.  The stage18 logs show the free V-Net source S
# at 7x the structured terms (rms T 0.022 F 0.009 S 0.162), so the physics graph contributes little however its terms are
# shaped.  Two changes, one stage each, both decided on RMSE (adoption rule) AND reported on the physics metrics
# (physics share of the core, guide-only skill, guide gain; scripts/core_physics_summary.py):
#   stage21a: beta starts at 1 (the guide's whole flow carries the state from the first step; core_beta_init) and
#             Stage B adds the source budget --source-penalty 0.1 = squared share of the V-Net source in the core's
#             sub-step change, so the learned part is pushed into beta, the conservative face flux and the diffusion.
#   stage21b: the guide is trained as a dynamical core of its own: Stage A (heads only, from the base's guide) rolls the
#             guide's own dynamics x_{j+1} = SL_V(x_j) + S(x_j) tend_scale 4 x 6 h and matches the observed states
#             (--guide-path-steps 4), then the same Stage B as stage21a with the new guide.
# Base = the stage18 family member the chains adopted (18h's base.txt; 18h added 2026-11-10 after 18g, card 14:59 KST).  stage21 does not change the chain base for
# stage19/20 (base.txt / bflags.txt are passed through); stage20_compose considers both runs like the others.
# Design: docs/design/Stage21_Physics_First_Core_KO.md
cd "$(dirname "$0")/../.."
S9B=runs/era5_64x32_s9b_merged
S17A=runs/era5_64x32_s17a_dyn
S18H=runs/era5_64x32_s18h_energy
DIR=runs/era5_64x32_s21_physfirst
LOG=$DIR/chain_s21.log
echo "[$(date -u +%H:%M:%S)] stage21 waits for CHAIN_S18H_DONE" >> $LOG
until grep -q "CHAIN_S18H_DONE" $S18H/chain_s18h.log 2>/dev/null; do sleep 120; done
BASE=$(cat $S18H/base.txt); BFLAGS=$(cat $S18H/bflags.txt)
echo "$BASE" > $DIR/base.txt; echo "$BFLAGS" > $DIR/bflags.txt
echo "[$(date -u +%H:%M:%S)] stage21 start on $BASE ($BFLAGS)" >> $LOG
FT="--threads 4 --batch 4 --lr 3e-4 --steps 1000 --val-every 250 --rollout-steps 2 --rollout-start 0 --rollout-weight 1.0"
EV="--threads 4 --stage eval --eval-stride 16 --max-lead-steps 5"
SRC="--source-penalty 0.1"

run_b_and_eval() {  # $1 = run dir, $2 = stage tag
  local RUN=$1 TAG=$2
  {
  python -W ignore scripts/train_eval.py --run $RUN $FT --stage b --init-vnet $S9B/vnet.pt $BFLAGS $SRC &&
  python -W ignore scripts/train_eval.py --run $RUN $EV --n-ens-cases 0 &&
  python -W ignore scripts/diagnose_guide.py --run $RUN --threads 4 && echo PIPELINE_DONE
  } >> $RUN/pipeline.log 2>&1
  echo "[$(date -u +%H:%M:%S)] $TAG pipeline exit $?" >> $LOG
  for Y in 2020 2022; do
    python -W ignore scripts/train_eval.py --run $RUN $EV --n-ens-cases 0 --eval-data data/era5_wb2_64x32_${Y}test.npz \
      --eval-column-data data/era5_wb2_64x32_column10_${Y}test.npz --eval-tag $Y --max-cases 88 > $RUN/eval_$Y.log 2>&1
    echo "[$(date -u +%H:%M:%S)] $TAG eval_$Y exit $?" >> $LOG
  done
  echo "$TAG physics metrics (run, then base):" >> $LOG
  python scripts/core_physics_summary.py $RUN/eval.json $BASE/eval.json >> $LOG 2>&1
}

# ---------------------------------------------------------------- stage21a: physics-first core
RUN=runs/era5_64x32_s21a_physfirst
mkdir -p $RUN
for f in normalizer.pt column_norm.json; do [ -f $RUN/$f ] || cp $S9B/$f $RUN/; done
for f in guide.pt a_done.json log_a.jsonl vnet_noguide.pt vnet_noguide_done.json log_vnet_noguide.jsonl tv_eps.json; do
  if [ ! -f $RUN/$f ] && [ -f $BASE/$f ]; then cp $BASE/$f $RUN/; fi
done
[ -f $RUN/config.json ] || python - "$BASE/config.json" "$RUN/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c.update(core_beta_init=1.0)
json.dump(c, open(sys.argv[2], "w"), indent=2)
PY
echo "[$(date -u +%H:%M:%S)] stage21a start: $BASE config + core_beta_init 1, Stage B $SRC" >> $LOG
run_b_and_eval $RUN stage21a
echo "stage21a vs base $BASE:" >> $LOG
if python scripts/decide_adoption.py $RUN/eval.json $BASE/eval.json --min-mean-gain 0.5 >> $LOG; then
  echo "stage21a ADOPTED" >> $LOG
else
  echo "stage21a REJECTED" >> $LOG
fi
S21A=$RUN

# ---------------------------------------------------------------- stage21b: the guide as a dynamical core
RUN=runs/era5_64x32_s21b_guidecore
mkdir -p $RUN
for f in normalizer.pt column_norm.json; do [ -f $RUN/$f ] || cp $S9B/$f $RUN/; done
for f in vnet_noguide.pt vnet_noguide_done.json log_vnet_noguide.jsonl tv_eps.json; do
  if [ ! -f $RUN/$f ] && [ -f $BASE/$f ]; then cp $BASE/$f $RUN/; fi
done
[ -f $RUN/config.json ] || cp $S21A/config.json $RUN/config.json
A="--threads 4 --batch 4 --lr 5e-4 --stage a --steps 1000 --val-every 200 --init-guide $BASE/guide.pt --guide-heads-only 1 \
   --perturb-model $S17A/vnet_noguide.pt --perturb-prob 0.3 --guide-path-steps 4 --guide-path-weight 1.0"
echo "[$(date -u +%H:%M:%S)] stage21b start: Stage A guide dynamical core (4 x 6 h path) from $BASE/guide.pt" >> $LOG
python -W ignore scripts/train_eval.py --run $RUN $A >> $RUN/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] stage21b stage A exit $?" >> $LOG
grep "^\[.*\] A " $RUN/pipeline.log | tail -n 1 >> $LOG
run_b_and_eval $RUN stage21b
echo "stage21b vs stage21a:" >> $LOG
OKA=0
if python scripts/decide_adoption.py $RUN/eval.json $S21A/eval.json --min-mean-gain 0.5 >> $LOG; then OKA=1; fi
echo "stage21b vs base $BASE:" >> $LOG
OKB=0
if python scripts/decide_adoption.py $RUN/eval.json $BASE/eval.json --min-mean-gain 0.5 >> $LOG; then OKB=1; fi
if [ $OKA = 1 ] && [ $OKB = 1 ]; then echo "stage21b ADOPTED" >> $LOG; else echo "stage21b REJECTED" >> $LOG; fi
echo "[$(date -u +%H:%M:%S)] stage21 done CHAIN_S21_DONE" >> $LOG
