#!/usr/bin/env bash
# Stage stage12c (2026-11-04 09:00 KST): improvement stage 3 — guide fine-tuning in Stage B.  The stage3 guide is
# trained on the 6 h tendency objective and then frozen; here its weights get their own cosine lr 1e-4 (V-Net 1e-3)
# during Stage B so the physics features adapt to the forecast objective (guide_finetuned.pt, used at inference).
# Everything else = stage3 (data, normaliser, guide.pt start, 1500 steps, rollout 2 from 500); the guide-free
# baseline is unchanged by this option, so stage3's is copied.  Queued after stage11b; ≈1.6 h.  Compare with stage3.
cd "$(dirname "$0")/../.."
S3=runs/era5_64x32_physops_roll
RUN=runs/era5_64x32_s12c_guideft
LOG=$RUN/chain_s12c.log
until grep -q CHAIN_S11B_DONE runs/era5_64x32_s11b_simp/chain_s11b.log 2>/dev/null; do sleep 60; done
echo "[$(date -u +%H:%M:%S)] stage11b chain finished -> stage12c (guide fine-tuning lr 1e-4)" >> $LOG
for f in config.json normalizer.pt guide.pt log_a.jsonl vnet_noguide.pt log_vnet_noguide.jsonl; do
  [ -f "$RUN/$f" ] || cp "$S3/$f" "$RUN/"; done
COMMON="--run $RUN --threads 4 --batch 4 --lr 1e-3"
B="--steps 1500 --val-every 250 --rollout-steps 2 --rollout-start 500 --rollout-weight 1.0 --guide-lr 1e-4"
{
python -W ignore scripts/train_eval.py $COMMON --stage b $B &&
python -W ignore scripts/train_eval.py --run $RUN --threads 4 --stage eval --eval-stride 16 --max-lead-steps 5 \
  --n-ens-cases 24 --members 8 --sample-steps 20 &&
python -W ignore scripts/diagnose_guide.py --run $RUN --threads 4 && echo PIPELINE_DONE
} > $RUN/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] stage12c pipeline exit $? CHAIN_S12C_DONE" >> $LOG
