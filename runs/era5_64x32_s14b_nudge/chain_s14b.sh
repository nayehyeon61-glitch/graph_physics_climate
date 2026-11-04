#!/usr/bin/env bash
# Stage stage14b (2026-11-04 12:20 KST, 감자 "초기에 nudging 구조로 초기 보정 강화" 11:56 KST): initial-increment
# nudging on top of stage14a.  Before the first forecast step the model forecasts the *initial* time from the previous
# state, M(X_{t-24h}); the increment e_t = X_t - M(X_{t-24h}) (the development the model failed to predict) is relaxed
# into the first step with a learned state-dependent coefficient alpha in (0,1) per variable and grid node
# (gpclimate/nn/nudging.py).  The added term is carried into later rollout steps (Evensen 2009 Ch. 12 red-noise
# recursion q_k = rho q_{k-1}; alpha plays rho; init = measured rho 0.07 of stage3's 24 h error).
# Everything else = stage14a (stage3 guide.pt / normaliser reused, Stage A skipped, Stage B 1500 steps, rollout 2
# from 500, prev_state=1); the guide-free baseline is retrained with the same nudging.  Queued after stage14a; ~3.3 h
# (+1 no-grad forward per training step).  Compare with stage14a (+ stage3, ClimODE).
cd "$(dirname "$0")/../.."
S3=runs/era5_64x32_physops_roll
RUN=runs/era5_64x32_s14b_nudge
LOG=$RUN/chain_s14b.log
until grep -q CHAIN_S14A_DONE runs/era5_64x32_s14a_prev/chain_s14a.log 2>/dev/null; do sleep 60; done
echo "[$(date -u +%H:%M:%S)] stage14a chain finished -> stage14b (initial-increment nudging)" >> $LOG
for f in normalizer.pt guide.pt log_a.jsonl; do [ -f "$RUN/$f" ] || cp "$S3/$f" "$RUN/"; done
[ -f "$RUN/config.json" ] || python - "$S3/config.json" "$RUN/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c.update(prev_state=1, nudge=1, nudge_carry=1, nudge_alpha_init=0.07)  # alpha init = measured rho (scripts/model_error_autocorr.py)
json.dump(c, open(sys.argv[2], "w"), indent=2)
PY
COMMON="--run $RUN --threads 4 --batch 4 --lr 1e-3"
B="--steps 1500 --val-every 250 --rollout-steps 2 --rollout-start 500 --rollout-weight 1.0"
{
python -W ignore scripts/train_eval.py $COMMON --stage b $B &&
python -W ignore scripts/train_eval.py $COMMON --stage b_noguide $B &&
python -W ignore scripts/train_eval.py --run $RUN --threads 4 --stage eval --eval-stride 16 --max-lead-steps 5 \
  --n-ens-cases 24 --members 8 --sample-steps 20 &&
python -W ignore scripts/diagnose_guide.py --run $RUN --threads 4 && echo PIPELINE_DONE
} > $RUN/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] stage14b pipeline exit $? CHAIN_S14B_DONE" >> $LOG
