#!/usr/bin/env bash
# Stage stage12d (2026-11-04 09:00 KST): improvement stage 4 — rollout curriculum.  stage3 uses 2 autoregressive
# steps from step 500; here "500:2,1000:3": a third step (72 h) for the last 500 steps.  Evidence: forecast
# activity decays with lead (msl 0.94 -> 0.74, winds 0.88 -> 0.65 of the observed anomaly std by 120 h) and the
# error growth beyond 48 h is where our lead over ClimODE comes from.  Everything else = stage3; the guide-free
# baseline is retrained with the same curriculum.  Queued after stage12c; ≈3 h.  Compare with stage3 (+ ClimODE).
cd "$(dirname "$0")/../.."
S3=runs/era5_64x32_physops_roll
RUN=runs/era5_64x32_s12d_roll3
LOG=$RUN/chain_s12d.log
until grep -q CHAIN_S12C_DONE runs/era5_64x32_s12c_guideft/chain_s12c.log 2>/dev/null; do sleep 60; done
echo "[$(date -u +%H:%M:%S)] stage12c chain finished -> stage12d (rollout schedule 500:2,1000:3)" >> $LOG
for f in config.json normalizer.pt guide.pt log_a.jsonl; do [ -f "$RUN/$f" ] || cp "$S3/$f" "$RUN/"; done
COMMON="--run $RUN --threads 4 --batch 4 --lr 1e-3"
B="--steps 1500 --val-every 250 --rollout-schedule 500:2,1000:3 --rollout-weight 1.0"
{
python -W ignore scripts/train_eval.py $COMMON --stage b $B &&
python -W ignore scripts/train_eval.py $COMMON --stage b_noguide $B &&
python -W ignore scripts/train_eval.py --run $RUN --threads 4 --stage eval --eval-stride 16 --max-lead-steps 5 \
  --n-ens-cases 24 --members 8 --sample-steps 20 &&
python -W ignore scripts/diagnose_guide.py --run $RUN --threads 4 && echo PIPELINE_DONE
} > $RUN/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] stage12d pipeline exit $? CHAIN_S12D_DONE" >> $LOG
