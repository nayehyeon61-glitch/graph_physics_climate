#!/usr/bin/env bash
# Stage stage12a (2026-11-04 09:00 KST): improvement stage 1 of the error-structure investigation — EMA of the trained
# weights in Stage B (decay 0.995, warm start; validation and the saved checkpoint use the EMA weights).  Everything
# else = stage3: same data 2016–2017, normaliser, Stage A guide reused from stage3 (Stage A skipped), Stage B 1500
# steps, batch 4, rollout 2 from step 500; the guide-free baseline is retrained with the same EMA.  Evidence: the
# validation MSE still falls 5.5 % over the last 500 steps and batch 4 makes the final weights noisy.
# Queued right after stage11a (CPU is shared); ≈2.6 h.  Compare with stage3 (+ ClimODE).
cd "$(dirname "$0")/../.."
S3=runs/era5_64x32_physops_roll
RUN=runs/era5_64x32_s12a_ema
LOG=$RUN/chain_s12a.log
until grep -q CHAIN_S11_DONE runs/era5_64x32_s11a_curv/chain_s11.log 2>/dev/null; do sleep 60; done
echo "[$(date -u +%H:%M:%S)] stage11a chain finished -> stage12a (EMA 0.995)" >> $LOG
for f in config.json normalizer.pt guide.pt log_a.jsonl; do [ -f "$RUN/$f" ] || cp "$S3/$f" "$RUN/"; done
COMMON="--run $RUN --threads 4 --batch 4 --lr 1e-3"
B="--steps 1500 --val-every 250 --rollout-steps 2 --rollout-start 500 --rollout-weight 1.0 --ema 0.995"
{
python -W ignore scripts/train_eval.py $COMMON --stage b $B &&
python -W ignore scripts/train_eval.py $COMMON --stage b_noguide $B &&
python -W ignore scripts/train_eval.py --run $RUN --threads 4 --stage eval --eval-stride 16 --max-lead-steps 5 \
  --n-ens-cases 24 --members 8 --sample-steps 20 &&
python -W ignore scripts/diagnose_guide.py --run $RUN --threads 4 && echo PIPELINE_DONE
} > $RUN/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] stage12a pipeline exit $? CHAIN_S12A_DONE" >> $LOG
