#!/usr/bin/env bash
# Stage stage17a (2026-11-07 19:55 KST): dynamic guide (감자 2026-11-07 19:31 KST "우선 추천방향 진행", 19:31 "아직 논문으로
# 하기엔 역시 guide 가 약해요").  The static guide is evaluated once on X_t, knows the 6 h tendency, and its gain fades
# with lead (+4.26 -> +1.75 % in stage9b).  stage17a integrates the 24 h lead in four 6 h sub-steps (graph neural ODE,
# Euler) and re-evaluates the guide and the fixed operators on every path state, with the solar forcing of the
# sub-step time; the path states X_{t+6/12/18 h} are supervised with ERA5 (WeatherODE-style).  No guide drift yet
# (that is stage17b).  CPU budget: paired fine-tuning, 1000 steps from stage9b's weights, for both arms:
#   stage17-ctl : stage9b V-Net (delta) fine-tuned 1000 steps, same lr / rollout / data  (the comparison baseline)
#   stage17a    : the same weights fine-tuned 1000 steps as the dynamic-guide ODE
# each with its guide-free pair fine-tuned the same way (guide gain under each integrator).  Compare stage17a vs
# stage17-ctl (+ stage9b, pairs, ClimODE, 2020/2022).  Design: docs/design/Stage17_Dynamic_Guide_KO.md
cd "$(dirname "$0")/../.."
S9B=runs/era5_64x32_s9b_merged
CTL=runs/era5_64x32_s17_ctl
RUN=runs/era5_64x32_s17a_dyn
LOG=$RUN/chain_s17a.log
echo "[$(date -u +%H:%M:%S)] stage17a waits for POST_S16_DONE" >> $LOG
until grep -q "POST_S16_DONE" runs/era5_64x32_s16_litsplit/chain_s16.log 2>/dev/null; do sleep 120; done
echo "[$(date -u +%H:%M:%S)] stage17a start" >> $LOG
for D in $CTL $RUN; do
  for f in normalizer.pt guide.pt a_done.json column_norm.json; do [ -f $D/$f ] || cp $S9B/$f $D/; done
done
[ -f $CTL/config.json ] || cp $S9B/config.json $CTL/config.json
[ -f $RUN/config.json ] || python - "$S9B/config.json" "$RUN/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c.update(integrator="ode", ode_steps=4, ode_solver="euler", ode_physics=0, ode_path_forcing=1, ode_drift_gain=0,
         inter_times=3, inter_weight=1.0)
json.dump(c, open(sys.argv[2], "w"), indent=2)
PY
FT="--threads 4 --batch 4 --lr 3e-4 --steps 1000 --val-every 250 --rollout-steps 2 --rollout-start 0 --rollout-weight 1.0"
EV="--threads 4 --stage eval --eval-stride 16 --max-lead-steps 5"
for D in $CTL $RUN; do
  {
  python -W ignore scripts/train_eval.py --run $D $FT --stage b_noguide --init-vnet $S9B/vnet_noguide.pt &&
  python -W ignore scripts/train_eval.py --run $D $FT --stage b --init-vnet $S9B/vnet.pt &&
  python -W ignore scripts/train_eval.py --run $D $EV --n-ens-cases 0 &&
  python -W ignore scripts/diagnose_guide.py --run $D --threads 4 && echo PIPELINE_DONE
  } >> $D/pipeline.log 2>&1
  echo "[$(date -u +%H:%M:%S)] $D pipeline exit $?" >> $LOG
  for Y in 2020 2022; do
    python -W ignore scripts/train_eval.py --run $D $EV --n-ens-cases 0 --eval-data data/era5_wb2_64x32_${Y}test.npz \
      --eval-column-data data/era5_wb2_64x32_column10_${Y}test.npz --eval-tag $Y --max-cases 88 > $D/eval_$Y.log 2>&1
    echo "[$(date -u +%H:%M:%S)] $D eval_$Y exit $?" >> $LOG
  done
done
python scripts/decide_adoption.py $RUN/eval.json $CTL/eval.json --min-mean-gain 0.5 >> $LOG
echo "[$(date -u +%H:%M:%S)] stage17a done CHAIN_S17A_DONE" >> $LOG
