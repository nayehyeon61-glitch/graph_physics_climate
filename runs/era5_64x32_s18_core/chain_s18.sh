#!/usr/bin/env bash
# Stage stage18 (2026-11-07 21:20 KST): graph tendency core (감자 2026-11-07 20:31 KST: "dynamic guide에 정적인 부분이
# 있다 ... PG-Flow의 tendency core를 graph version으로, 물리가이드에 대한 graph transformer가 이해하도록").
# Every 6 h ODE sub-step becomes the PG-Flow core on the grid's finite-volume face graph,
#   X_{k+1} = SL_{beta V}(X_k) - div_G J_theta + S_theta,
# V = the guide's flow re-evaluated on X_k (stage17c flow head), beta = per-cell/level share of that flow (V-Net head,
# 0 at start), J_theta = antisymmetric face flux from the V-Net embedding (exactly conservative), S_theta = V-Net output.
# The guide's own core terms (its transport change and source) enter the V-Net every sub-step, in the grid embedding and
# as a fourth PIGT token source C.  The static per-variable gain of stage17c is gone (ode_physics 0).  Guide = stage17c's
# (trunk = stage9b, heads retrained with the flow head), same paired fine-tuning (V-Net from stage9b, 1000 steps), the
# guide-free pair is stage17a's.  Compare with stage17c (immediately preceding) and stage17a (+ stage17-ctl, ClimODE,
# 2020/2022); report beta, the term shares, the C attention mass and the guide-only (physics_only) forecast.
# Design: docs/design/Stage18_Graph_Tendency_Core_KO.md
cd "$(dirname "$0")/../.."
S9B=runs/era5_64x32_s9b_merged
S17A=runs/era5_64x32_s17a_dyn
S17C=runs/era5_64x32_s17c_transport
RUN=runs/era5_64x32_s18_core
LOG=$RUN/chain_s18.log
echo "[$(date -u +%H:%M:%S)] stage18 waits for CHAIN_S17C_DONE" >> $LOG
until grep -q "CHAIN_S17C_DONE" $S17C/chain_s17c.log 2>/dev/null; do sleep 120; done
echo "[$(date -u +%H:%M:%S)] stage18 start" >> $LOG
for f in normalizer.pt column_norm.json; do [ -f $RUN/$f ] || cp $S9B/$f $RUN/; done
for f in guide.pt a_done.json log_a.jsonl; do [ -f $RUN/$f ] || cp $S17C/$f $RUN/; done
for f in vnet_noguide.pt vnet_noguide_done.json log_vnet_noguide.jsonl; do [ -f $RUN/$f ] || cp $S17A/$f $RUN/; done
[ -f $RUN/config.json ] || python - "$S17C/config.json" "$RUN/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c.update(ode_physics=0, ode_drift_gain=0, ode_drift="transport", flow_head=1, tendency_core=1, core_tokens=1)
json.dump(c, open(sys.argv[2], "w"), indent=2)
PY
FT="--threads 4 --batch 4 --lr 3e-4 --steps 1000 --val-every 250 --rollout-steps 2 --rollout-start 0 --rollout-weight 1.0"
EV="--threads 4 --stage eval --eval-stride 16 --max-lead-steps 5"
{
python -W ignore scripts/train_eval.py --run $RUN $FT --stage b --init-vnet $S9B/vnet.pt &&
python -W ignore scripts/train_eval.py --run $RUN $EV --n-ens-cases 0 &&
python -W ignore scripts/diagnose_guide.py --run $RUN --threads 4 && echo PIPELINE_DONE
} >> $RUN/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] stage18 pipeline exit $?" >> $LOG
for Y in 2020 2022; do
  python -W ignore scripts/train_eval.py --run $RUN $EV --n-ens-cases 0 --eval-data data/era5_wb2_64x32_${Y}test.npz \
    --eval-column-data data/era5_wb2_64x32_column10_${Y}test.npz --eval-tag $Y --max-cases 88 > $RUN/eval_$Y.log 2>&1
  echo "[$(date -u +%H:%M:%S)] stage18 eval_$Y exit $?" >> $LOG
done
echo "vs stage17c (immediately preceding):" >> $LOG
python scripts/decide_adoption.py $RUN/eval.json $S17C/eval.json --min-mean-gain 0.5 >> $LOG
echo "vs stage17a:" >> $LOG
python scripts/decide_adoption.py $RUN/eval.json $S17A/eval.json --min-mean-gain 0.5 >> $LOG
echo "[$(date -u +%H:%M:%S)] stage18 done CHAIN_S18_DONE" >> $LOG
