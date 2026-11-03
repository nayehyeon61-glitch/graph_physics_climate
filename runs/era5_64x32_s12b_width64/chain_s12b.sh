#!/usr/bin/env bash
# Stage stage12b (2026-11-04 09:00 KST): improvement stage 2 — V-Net capacity.  Width cs 48 / cv 12 -> 64 / 16
# (554,665 -> 825,505 params, +33 % step time); guide, data, normaliser and recipe = stage3 (Stage A skipped, guide.pt
# reused; Stage B 1500 steps, rollout 2 from 500; guide-free baseline retrained at the same width).  Evidence: the
# 24 h gap to ClimODE (2.75 M params) sits in the mid-latitude / polar synoptic scales (zonal wavenumbers 8–15 keep
# only 73 % of the observed power at 24 h) while the forecasts are otherwise unbiased.
# Queued after stage12a; ≈3.3 h.  Compare with stage3 (+ ClimODE).
cd "$(dirname "$0")/../.."
S3=runs/era5_64x32_physops_roll
RUN=runs/era5_64x32_s12b_width64
LOG=$RUN/chain_s12b.log
until grep -q CHAIN_S12A_DONE runs/era5_64x32_s12a_ema/chain_s12a.log 2>/dev/null; do sleep 60; done
echo "[$(date -u +%H:%M:%S)] stage12a chain finished -> stage12b (V-Net width 64/16)" >> $LOG
for f in normalizer.pt guide.pt log_a.jsonl; do [ -f "$RUN/$f" ] || cp "$S3/$f" "$RUN/"; done
[ -f "$RUN/config.json" ] || python - "$S3/config.json" "$RUN/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c.update(cs=64, cv=16)
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
echo "[$(date -u +%H:%M:%S)] stage12b pipeline exit $? CHAIN_S12B_DONE" >> $LOG
