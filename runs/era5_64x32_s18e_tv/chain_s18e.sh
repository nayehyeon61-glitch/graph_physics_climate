#!/usr/bin/env bash
# Stage stage18e (2026-11-09 11:40 KST): edge-preserving diffusion in the Laplace block (감자 2026-11-09, Kovac & Smith
# 2009 arXiv 0911.1928 "Regression on a graph": "physical tendency core 구현시 이 graph regression 도 반영하면 어떨가요?";
# decision card: "stage18e 경계 보존 확산").  stage18d's core with the face diffusivity multiplied by
# eps / sqrt((q_j - q_i)^2 + eps^2), the eps-regularised gradient flow of the paper's L1 edge penalty: linear diffusion
# across small differences, a bounded flux across fronts, so the block removes noise but keeps jets and fronts.  eps per
# variable = the median neighbour difference of 64 training states (tv_eps.json) x a learned factor (init 1).  Same
# paired fine-tuning (V-Net from stage9b, 1000 steps); guide and pair are stage18d's (= the base's).  Decided against
# stage18d (one change) and, when 18d was rejected, also against the base the chain adopted: 18e becomes the base of
# stage19 only when it beats both.  Design: docs/design/Stage18_Graph_Tendency_Core_KO.md §5
cd "$(dirname "$0")/../.."
S9B=runs/era5_64x32_s9b_merged
S18D=runs/era5_64x32_s18d_laplace
RUN=runs/era5_64x32_s18e_tv
LOG=$RUN/chain_s18e.log
echo "[$(date -u +%H:%M:%S)] stage18e waits for CHAIN_S18D_DONE" >> $LOG
until grep -q "CHAIN_S18D_DONE" $S18D/chain_s18d.log 2>/dev/null; do sleep 120; done
# stage18d recorded the adopted member of 18 / 18b / 18c (and its Stage B guide flags); 18d itself when adopted
BASE=$(cat $S18D/base.txt); BFLAGS=$(cat $S18D/bflags.txt)
if grep -q "stage18d ADOPTED" $S18D/chain_s18d.log; then BASE=$S18D; fi
echo "[$(date -u +%H:%M:%S)] stage18e start: stage18d config + core_tv, base $BASE ($BFLAGS)" >> $LOG
for f in normalizer.pt column_norm.json guide.pt a_done.json log_a.jsonl vnet_noguide.pt vnet_noguide_done.json \
         log_vnet_noguide.jsonl; do
  [ -f $RUN/$f ] || cp $S18D/$f $RUN/
done
[ -f $RUN/config.json ] || python - "$S18D/config.json" "$RUN/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c.update(core_laplace=1, core_tv=1)
json.dump(c, open(sys.argv[2], "w"), indent=2)
PY
FT="--threads 4 --batch 4 --lr 3e-4 --steps 1000 --val-every 250 --rollout-steps 2 --rollout-start 0 --rollout-weight 1.0"
EV="--threads 4 --stage eval --eval-stride 16 --max-lead-steps 5"
{
python -W ignore scripts/train_eval.py --run $RUN $FT --stage b --init-vnet $S9B/vnet.pt $BFLAGS &&
python -W ignore scripts/train_eval.py --run $RUN $EV --n-ens-cases 0 &&
python -W ignore scripts/diagnose_guide.py --run $RUN --threads 4 && echo PIPELINE_DONE
} >> $RUN/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] stage18e pipeline exit $?" >> $LOG
for Y in 2020 2022; do
  python -W ignore scripts/train_eval.py --run $RUN $EV --n-ens-cases 0 --eval-data data/era5_wb2_64x32_${Y}test.npz \
    --eval-column-data data/era5_wb2_64x32_column10_${Y}test.npz --eval-tag $Y --max-cases 88 > $RUN/eval_$Y.log 2>&1
  echo "[$(date -u +%H:%M:%S)] stage18e eval_$Y exit $?" >> $LOG
done
echo "vs stage18d $S18D:" >> $LOG
OK18D=0
if python scripts/decide_adoption.py $RUN/eval.json $S18D/eval.json --min-mean-gain 0.5 >> $LOG; then OK18D=1; fi
OKBASE=$OK18D
if [ "$BASE" != "$S18D" ]; then
  echo "vs base $BASE:" >> $LOG
  OKBASE=0
  if python scripts/decide_adoption.py $RUN/eval.json $BASE/eval.json --min-mean-gain 0.5 >> $LOG; then OKBASE=1; fi
fi
if [ $OK18D = 1 ] && [ $OKBASE = 1 ]; then
  echo "stage18e ADOPTED" >> $LOG
  echo "$RUN" > $RUN/base.txt
else
  echo "stage18e REJECTED" >> $LOG
  echo "$BASE" > $RUN/base.txt
fi
echo "$BFLAGS" > $RUN/bflags.txt
echo "[$(date -u +%H:%M:%S)] stage18e done CHAIN_S18E_DONE" >> $LOG
