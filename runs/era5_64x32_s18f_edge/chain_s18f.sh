#!/usr/bin/env bash
# Stage stage18f (2026-11-09 14:45 KST): dual-space edge physics theta_p (감자 2026-11-09 14:09 KST, Delaunay-Voronoi
# discussion + arXiv 2406.05225 "A manifold perspective on the statistical generalization of GNNs"; decision card
# 14:14 KST "18f 뒤 항상 실행").  Fix the Voronoi areas and the Delaunay connectivity, learn the edge transfer
# coefficient  du_i/dt = (1/|V_i|) sum_j (|F_ij|/d_ij) kappa_theta(u_i, u_j, p_ij) (u_j - u_i),  kappa_ij = kappa_ji
# >= 0, from the two cells' embeddings and physical edge features (terrain / land-sea difference, guide flow across and
# along the face, |q_j - q_i|, latitude, orientation), as a zero-initialised multiplier of the Laplace block's face
# diffusivity.  Parent = stage18e if adopted else stage18d (both carry the Laplace block); config = parent + core_edge.
# Same paired fine-tuning (V-Net from stage9b, 1000 steps); guide and pair from the chain base.
# Design: docs/design/Stage18f_Dual_Space_Core_KO.md
cd "$(dirname "$0")/../.."
S9B=runs/era5_64x32_s9b_merged
S18D=runs/era5_64x32_s18d_laplace
S18E=runs/era5_64x32_s18e_tv
RUN=runs/era5_64x32_s18f_edge
LOG=$RUN/chain_s18f.log
echo "[$(date -u +%H:%M:%S)] stage18f waits for CHAIN_S18E_DONE" >> $LOG
until grep -q "CHAIN_S18E_DONE" $S18E/chain_s18e.log 2>/dev/null; do sleep 120; done
BASE=$(cat $S18E/base.txt); BFLAGS=$(cat $S18E/bflags.txt)
if grep -q "stage18e ADOPTED" $S18E/chain_s18e.log; then PARENT=$S18E; else PARENT=$S18D; fi
echo "$PARENT" > $RUN/parent.txt
echo "[$(date -u +%H:%M:%S)] stage18f start: $PARENT config + core_edge, base $BASE ($BFLAGS)" >> $LOG
for f in normalizer.pt column_norm.json guide.pt a_done.json log_a.jsonl vnet_noguide.pt vnet_noguide_done.json \
         log_vnet_noguide.jsonl tv_eps.json; do
  if [ ! -f $RUN/$f ] && [ -f $PARENT/$f ]; then cp $PARENT/$f $RUN/; fi
done
[ -f $RUN/config.json ] || python - "$PARENT/config.json" "$RUN/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c.update(core_laplace=1, core_edge=1)
json.dump(c, open(sys.argv[2], "w"), indent=2)
PY
FT="--threads 4 --batch 4 --lr 3e-4 --steps 1000 --val-every 250 --rollout-steps 2 --rollout-start 0 --rollout-weight 1.0"
EV="--threads 4 --stage eval --eval-stride 16 --max-lead-steps 5"
{
python -W ignore scripts/train_eval.py --run $RUN $FT --stage b --init-vnet $S9B/vnet.pt $BFLAGS &&
python -W ignore scripts/train_eval.py --run $RUN $EV --n-ens-cases 0 &&
python -W ignore scripts/diagnose_guide.py --run $RUN --threads 4 && echo PIPELINE_DONE
} >> $RUN/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] stage18f pipeline exit $?" >> $LOG
for Y in 2020 2022; do
  python -W ignore scripts/train_eval.py --run $RUN $EV --n-ens-cases 0 --eval-data data/era5_wb2_64x32_${Y}test.npz \
    --eval-column-data data/era5_wb2_64x32_column10_${Y}test.npz --eval-tag $Y --max-cases 88 > $RUN/eval_$Y.log 2>&1
  echo "[$(date -u +%H:%M:%S)] stage18f eval_$Y exit $?" >> $LOG
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
  echo "stage18f ADOPTED" >> $LOG
  echo "$RUN" > $RUN/base.txt
else
  echo "stage18f REJECTED" >> $LOG
  echo "$BASE" > $RUN/base.txt
fi
echo "$BFLAGS" > $RUN/bflags.txt
echo "[$(date -u +%H:%M:%S)] stage18f done CHAIN_S18F_DONE" >> $LOG
