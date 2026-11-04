#!/usr/bin/env bash
# Stage stage13b (2026-11-04 10:00 KST): stage13a + **energy continuity** — Stage B adds the zonal spectral energy
# loss (mean squared log ratio of forecast / truth power per channel and wavenumber band 1–3, 4–7, 8–15, 16–32,
# cos-weighted; weight 0.1) on the one-step and rollout forecasts, so the forecast carries the observed energy in
# every band (neither damped synoptic scales nor spurious grid-scale growth).  Everything else = stage13a (same
# operator layer, data, guide, recipe); the guide-free baseline is retrained with the same loss.
# Queued after stage13a; ≈3 h.  Compare with stage13a (+ stage3, ClimODE).
cd "$(dirname "$0")/../.."
S13A=runs/era5_64x32_s13a_sfno
RUN=runs/era5_64x32_s13b_sfno_spec
LOG=$RUN/chain_s13b.log
until grep -q CHAIN_S13A_DONE runs/era5_64x32_s13a_sfno/chain_s13a.log 2>/dev/null; do sleep 60; done
echo "[$(date -u +%H:%M:%S)] stage13a chain finished -> stage13b (+ spectral energy loss 0.1)" >> $LOG
for f in config.json normalizer.pt guide.pt log_a.jsonl; do [ -f "$RUN/$f" ] || cp "$S13A/$f" "$RUN/"; done
COMMON="--run $RUN --threads 4 --batch 4 --lr 1e-3"
B="--steps 1500 --val-every 250 --rollout-steps 2 --rollout-start 500 --rollout-weight 1.0 --spec-energy-weight 0.1"
{
python -W ignore scripts/train_eval.py $COMMON --stage b $B &&
python -W ignore scripts/train_eval.py $COMMON --stage b_noguide $B &&
python -W ignore scripts/train_eval.py --run $RUN --threads 4 --stage eval --eval-stride 16 --max-lead-steps 5 \
  --n-ens-cases 24 --members 8 --sample-steps 20 &&
python -W ignore scripts/diagnose_guide.py --run $RUN --threads 4 && echo PIPELINE_DONE
} > $RUN/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] stage13b pipeline exit $? CHAIN_S13B_DONE" >> $LOG
