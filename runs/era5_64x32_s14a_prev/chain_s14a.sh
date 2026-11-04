#!/usr/bin/env bash
# Stage stage14a (2026-11-04 10:30 KST, 감자 "초기만 잡으면" 10:03 KST): the V-Net also sees the observed change over
# the previous lead, X_t - X_{t-24h} (GraphCast-style two-time-step input; ClimODE initialises its velocity from
# past frames).  Hypothesis: the 24-48 h mid-latitude/polar gap is an *initial development* error that a single
# snapshot cannot resolve.  In rollouts the previous forecast state plays X_{t-24h}, so training and inference agree.
# Everything else = stage3 (guide.pt and normaliser reused, Stage A skipped, Stage B 1500 steps, rollout 2 from 500);
# the guide-free baseline is retrained with the same input.  Queued after stage13b; ~2.6 h.  Compare with stage3.
cd "$(dirname "$0")/../.."
S3=runs/era5_64x32_physops_roll
RUN=runs/era5_64x32_s14a_prev
LOG=$RUN/chain_s14a.log
until grep -q CHAIN_S13B_DONE runs/era5_64x32_s13b_sfno_spec/chain_s13b.log 2>/dev/null; do sleep 60; done
echo "[$(date -u +%H:%M:%S)] stage13b chain finished -> stage14a (previous-state input)" >> $LOG
for f in normalizer.pt guide.pt log_a.jsonl; do [ -f "$RUN/$f" ] || cp "$S3/$f" "$RUN/"; done
[ -f "$RUN/config.json" ] || python - "$S3/config.json" "$RUN/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c.update(prev_state=1)
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
echo "[$(date -u +%H:%M:%S)] stage14a pipeline exit $? CHAIN_S14A_DONE" >> $LOG
