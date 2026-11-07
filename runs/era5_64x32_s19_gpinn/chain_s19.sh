#!/usr/bin/env bash
# Stage stage19a/b/c (2026-11-07 21:25 KST): GPINN guide (감자 2026-11-07 20:20 KST, arXiv 2306.09792: "gpinn은 원본
# 가이드, dynamic guide는 예측구조의 보조로"; 21:18 KST MTMLab/GNN-PINN as a reference: staged training, the physics
# penalty only where the relation holds).  Base = the stage18 family member the chains adopted (18d > 18c > 18b > 18).
#   stage19a: the guide's Stage A of stage17c (heads from stage9b's guide, trunk frozen, same perturbations, 1000 steps)
#             + soft PINN residuals of the relations that hold for ERA5 (geostrophic 500/850 hPa, thermal wind,
#             balanced 6 h change, hypsometric change, 500 hPa vorticity; scripts/pinn_equation_check.py) on the
#             guide's outputs (denoised state and its 6 h core step), weight 0.05
#   stage19b: + GPINN graph embedding: 8 eigenvectors of the terrain / coastline weighted grid Laplacian as extra
#             guide inputs (zero-initialised adapter trained with the heads)
#   stage19c: + the same residuals on the forecast path in Stage B (every 6 h path state, weight 0.02) -- the path
#             states are the PINN collocation points
# Each: paired fine-tuning (V-Net from stage9b, 1000 steps) with the base's pair; cumulative (19b keeps 19a's losses),
# each decided against the sub-stage right before it and also reported against the base.
# Design: docs/design/Stage18_Graph_Tendency_Core_KO.md §5 / docs/design/Stage19_GPINN_Guide_KO.md
cd "$(dirname "$0")/../.."
S9B=runs/era5_64x32_s9b_merged
S17A=runs/era5_64x32_s17a_dyn
S18=runs/era5_64x32_s18_core
S18B=runs/era5_64x32_s18b_prevpath
S18C=runs/era5_64x32_s18c_guidepath
S18D=runs/era5_64x32_s18d_laplace
LOG=runs/era5_64x32_s19_gpinn/chain_s19.log
echo "[$(date -u +%H:%M:%S)] stage19 waits for CHAIN_S18D_DONE" >> $LOG
until grep -q "CHAIN_S18D_DONE" $S18D/chain_s18d.log 2>/dev/null; do sleep 120; done
# stage18d recorded the adopted member of 18 / 18b / 18c (and its Stage B guide flags); 18d itself if adopted
BASE=$(cat $S18D/base.txt); BFLAGS=$(cat $S18D/bflags.txt)
if grep -q "stage18d ADOPTED" $S18D/chain_s18d.log; then BASE=$S18D; fi
echo "[$(date -u +%H:%M:%S)] stage19 start on $BASE ($BFLAGS)" >> $LOG
FT="--threads 4 --batch 4 --lr 3e-4 --steps 1000 --val-every 250 --rollout-steps 2 --rollout-start 0 --rollout-weight 1.0"
EV="--threads 4 --stage eval --eval-stride 16 --max-lead-steps 5"
A="--threads 4 --batch 4 --lr 5e-4 --stage a --steps 1000 --val-every 200 --init-guide $S9B/guide.pt --guide-heads-only 1 \
   --perturb-model $S17A/vnet_noguide.pt --perturb-prob 0.3"
PREV=$BASE
for S in a b c; do
  RUN=runs/era5_64x32_s19${S}_gpinn
  mkdir -p $RUN
  for f in normalizer.pt column_norm.json; do [ -f $RUN/$f ] || cp $S9B/$f $RUN/; done
  for f in vnet_noguide.pt vnet_noguide_done.json log_vnet_noguide.jsonl; do [ -f $RUN/$f ] || cp $BASE/$f $RUN/; done
  EMB=0; [ $S != a ] && EMB=8
  [ -f $RUN/config.json ] || python - "$BASE/config.json" "$RUN/config.json" "$EMB" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c.update(graph_embed=int(sys.argv[3]))
json.dump(c, open(sys.argv[2], "w"), indent=2)
PY
  PB=""; [ $S = c ] && PB="--pinn-weight-b 0.02"
  {
  python -W ignore scripts/train_eval.py --run $RUN $A --pinn-weight 0.05 &&
  python -W ignore scripts/train_eval.py --run $RUN $FT --stage b --init-vnet $S9B/vnet.pt $BFLAGS $PB &&
  python -W ignore scripts/train_eval.py --run $RUN $EV --n-ens-cases 0 &&
  python -W ignore scripts/diagnose_guide.py --run $RUN --threads 4 && echo PIPELINE_DONE
  } >> $RUN/pipeline.log 2>&1
  echo "[$(date -u +%H:%M:%S)] stage19$S pipeline exit $?" >> $LOG
  for Y in 2020 2022; do
    python -W ignore scripts/train_eval.py --run $RUN $EV --n-ens-cases 0 --eval-data data/era5_wb2_64x32_${Y}test.npz \
      --eval-column-data data/era5_wb2_64x32_column10_${Y}test.npz --eval-tag $Y --max-cases 88 > $RUN/eval_$Y.log 2>&1
    echo "[$(date -u +%H:%M:%S)] stage19$S eval_$Y exit $?" >> $LOG
  done
  # one change per sub-stage: each is decided against the sub-stage right before it (19a vs base, 19b vs 19a, ...)
  echo "stage19$S vs $PREV:" >> $LOG
  if python scripts/decide_adoption.py $RUN/eval.json $PREV/eval.json --min-mean-gain 0.5 >> $LOG; then
    echo "stage19$S ADOPTED" >> $LOG
  else
    echo "stage19$S REJECTED" >> $LOG
  fi
  if [ "$PREV" != "$BASE" ]; then
    echo "stage19$S vs base $BASE (cumulative):" >> $LOG
    python scripts/decide_adoption.py $RUN/eval.json $BASE/eval.json --min-mean-gain 0.5 >> $LOG
  fi
  PREV=$RUN
done
echo "[$(date -u +%H:%M:%S)] stage19 done CHAIN_S19_DONE" >> $LOG
