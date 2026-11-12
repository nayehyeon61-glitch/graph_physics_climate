#!/usr/bin/env bash
# Stage stage17c (2026-11-07 20:50 KST; re-based 21:00 KST): Lie transport guide (감자 2026-11-07 20:07 KST "유의미한
# guide 구조를 부탁드립니다").  stage17a + a guide drift in the ODE given by the structured guide step
#   G(X) - X = [semi-Lagrangian transport of every channel by the guide's flow (level wind + learned correction),
#               vectors parallel-transported] - X + source head * dt
# with a zero-initialised learnable per-variable gain (training starts at stage17a's dynamics).  stage17b (the same
# gain on the black-box tendency head) was cancelled by 감자 (2026-11-07 20:38 KST, "17b 취소, 17c 뒤 core"), so
# stage17c is compared with stage17a and the change is "structured transport-guide drift" as a whole.  The guide
# trunk (features P seen by the V-Net) is frozen at stage9b's weights; Stage A trains only the heads (grid adapter,
# flow, source) on the 6 h step.  Same paired fine-tuning (V-Net from stage9b, 1000 steps); the guide-free pair has
# no guide and no drift, so stage17a's pair is reused.  Compare with stage17a (+ stage17-ctl, ClimODE, 2020/2022),
# report the learned gains, the guide-only (physics_only) forecast and the transport share.  The flow head trained
# here is the transport velocity of stage18 (graph tendency core).
# Design: docs/design/Stage17c_Lie_Transport_Guide_KO.md
cd "$(dirname "$0")/../.."
S9B=runs/era5_64x32_s9b_merged
S17A=runs/era5_64x32_s17a_dyn
RUN=runs/era5_64x32_s17c_transport
LOG=$RUN/chain_s17c.log
echo "[$(date -u +%H:%M:%S)] stage17c waits for CHAIN_S17A_DONE" >> $LOG
until grep -q "CHAIN_S17A_DONE" $S17A/chain_s17a.log 2>/dev/null; do sleep 120; done
echo "[$(date -u +%H:%M:%S)] stage17c start" >> $LOG
for f in normalizer.pt column_norm.json; do [ -f $RUN/$f ] || cp $S9B/$f $RUN/; done
for f in vnet_noguide.pt vnet_noguide_done.json log_vnet_noguide.jsonl; do [ -f $RUN/$f ] || cp $S17A/$f $RUN/; done
[ -f $RUN/config.json ] || python - "$S17A/config.json" "$RUN/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c.update(ode_physics=1, ode_drift_gain=1, flow_head=1, ode_drift="transport")
json.dump(c, open(sys.argv[2], "w"), indent=2)
PY
FT="--threads 4 --batch 4 --lr 3e-4 --steps 1000 --val-every 250 --rollout-steps 2 --rollout-start 0 --rollout-weight 1.0"
EV="--threads 4 --stage eval --eval-stride 16 --max-lead-steps 5"
{
python -W ignore scripts/train_eval.py --run $RUN --threads 4 --batch 4 --lr 5e-4 --stage a --steps 1000 --val-every 200 \
  --init-guide $S9B/guide.pt --guide-heads-only 1 --perturb-model $S17A/vnet_noguide.pt --perturb-prob 0.3 &&
python -W ignore scripts/train_eval.py --run $RUN $FT --stage b --init-vnet $S9B/vnet.pt &&
python -W ignore scripts/train_eval.py --run $RUN $EV --n-ens-cases 0 &&
python -W ignore scripts/diagnose_guide.py --run $RUN --threads 4 && echo PIPELINE_DONE
} >> $RUN/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] stage17c pipeline exit $?" >> $LOG
for Y in 2020 2022; do
  python -W ignore scripts/train_eval.py --run $RUN $EV --n-ens-cases 0 --eval-data data/era5_wb2_64x32_${Y}test.npz \
    --eval-column-data data/era5_wb2_64x32_column10_${Y}test.npz --eval-tag $Y --max-cases 88 > $RUN/eval_$Y.log 2>&1
  echo "[$(date -u +%H:%M:%S)] stage17c eval_$Y exit $?" >> $LOG
done
python scripts/decide_adoption.py $RUN/eval.json $S17A/eval.json --min-mean-gain 0.5 >> $LOG
echo "[$(date -u +%H:%M:%S)] stage17c done CHAIN_S17C_DONE" >> $LOG
