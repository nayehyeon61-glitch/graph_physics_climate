#!/usr/bin/env bash
# Stage stage18d (2026-11-07 21:40 KST): Laplace block in the graph tendency core (감자 2026-11-07 21:32 KST, PhyMPGN
# ICLR 2025, github.com/intell-sci-comput/PhyMPGN: "주위를 covering 하는 게 core 만들때 괜찮을것 같은데요").  The core
# member the stage18 chains adopted (18c > 18b > 18) + one explicit diffusion step of the scalar channels per 6 h
# sub-step on the face graph: finite-volume Laplace-Beltrami on the covering cells (area A_i, cotangent weights l/d,
# PhyMPGN eq. 4-5), diffusivity K >= 0 per cell and channel from the V-Net embedding (starts ~ off), face numbers capped
# for stability.  Restores the K lap q term of the PG-Flow core.  Same paired fine-tuning (V-Net from stage9b, 1000
# steps); the guide and the guide-free pair are the base's; with the 18c base the guide heads are fine-tuned again.
# Compare with the base (+ stage17a, ClimODE, 2020/2022).  Design: docs/design/Stage18_Graph_Tendency_Core_KO.md §5
cd "$(dirname "$0")/../.."
S9B=runs/era5_64x32_s9b_merged
S18=runs/era5_64x32_s18_core
S18B=runs/era5_64x32_s18b_prevpath
S18C=runs/era5_64x32_s18c_guidepath
RUN=runs/era5_64x32_s18d_laplace
LOG=$RUN/chain_s18d.log
echo "[$(date -u +%H:%M:%S)] stage18d waits for CHAIN_S18C_DONE" >> $LOG
until grep -q "CHAIN_S18C_DONE" $S18C/chain_s18c.log 2>/dev/null; do sleep 120; done
if grep -q "stage18b ADOPTED" $S18B/chain_s18b.log; then BASE=$S18B; else BASE=$S18; fi
BFLAGS=""
if grep -A1 "vs base" $S18C/chain_s18c.log | grep -q '"adopt": true'; then
  BASE=$S18C; BFLAGS="--guide-lr 3e-5 --guide-heads-only 1"
fi
echo "$BASE" > $RUN/base.txt
echo "$BFLAGS" > $RUN/bflags.txt
echo "[$(date -u +%H:%M:%S)] stage18d start on $BASE ($BFLAGS)" >> $LOG
for f in normalizer.pt column_norm.json guide.pt a_done.json log_a.jsonl vnet_noguide.pt vnet_noguide_done.json \
         log_vnet_noguide.jsonl; do
  [ -f $RUN/$f ] || cp $BASE/$f $RUN/
done
[ -f $RUN/config.json ] || python - "$BASE/config.json" "$RUN/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c.update(core_laplace=1)
json.dump(c, open(sys.argv[2], "w"), indent=2)
PY
FT="--threads 4 --batch 4 --lr 3e-4 --steps 1000 --val-every 250 --rollout-steps 2 --rollout-start 0 --rollout-weight 1.0"
EV="--threads 4 --stage eval --eval-stride 16 --max-lead-steps 5"
{
python -W ignore scripts/train_eval.py --run $RUN $FT --stage b --init-vnet $S9B/vnet.pt $BFLAGS &&
python -W ignore scripts/train_eval.py --run $RUN $EV --n-ens-cases 0 &&
python -W ignore scripts/diagnose_guide.py --run $RUN --threads 4 && echo PIPELINE_DONE
} >> $RUN/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] stage18d pipeline exit $?" >> $LOG
for Y in 2020 2022; do
  python -W ignore scripts/train_eval.py --run $RUN $EV --n-ens-cases 0 --eval-data data/era5_wb2_64x32_${Y}test.npz \
    --eval-column-data data/era5_wb2_64x32_column10_${Y}test.npz --eval-tag $Y --max-cases 88 > $RUN/eval_$Y.log 2>&1
  echo "[$(date -u +%H:%M:%S)] stage18d eval_$Y exit $?" >> $LOG
done
echo "vs base $BASE:" >> $LOG
if python scripts/decide_adoption.py $RUN/eval.json $BASE/eval.json --min-mean-gain 0.5 >> $LOG; then
  echo "stage18d ADOPTED" >> $LOG
else
  echo "stage18d REJECTED" >> $LOG
fi
echo "[$(date -u +%H:%M:%S)] stage18d done CHAIN_S18D_DONE" >> $LOG
