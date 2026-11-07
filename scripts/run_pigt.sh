#!/usr/bin/env bash
# PIGT Stage B on top of an existing Stage-A guide (copied from BASE), same rollout training as BASE's V-Nets;
# the guide-free rollout-trained V-Net of BASE is reused as the baseline.  Then eval + diagnose.
set -uo pipefail
cd "$(dirname "$0")/.."
RUN=${RUN:-runs/era5_64x32_pigt}
BASE=${BASE:-runs/era5_64x32_physops_roll}
THREADS=${THREADS:-4}
mkdir -p "$RUN"
for f in guide.pt normalizer.pt log_a.jsonl vnet_noguide.pt log_vnet_noguide.jsonl; do [ -f "$RUN/$f" ] || cp "$BASE/$f" "$RUN/"; done
COMMON="--run $RUN --threads $THREADS --batch ${BATCH:-4} --lr ${LR:-1e-3} --physics-ops 1 --tend-head 1 --fusion pigt --pigt-layers ${PIGT_LAYERS:-1} --pigt-ops ${PIGT_OPS:-1}"
B_EXTRA="--rollout-steps ${ROLLOUT_STEPS:-2} --rollout-start ${ROLLOUT_START:-500} --rollout-weight ${ROLLOUT_WEIGHT:-1.0}"
python -W ignore scripts/train_eval.py $COMMON --stage b --steps "${STEPS_B:-1500}" --val-every 250 $B_EXTRA || exit 1
python -W ignore scripts/train_eval.py --run "$RUN" --threads "$THREADS" --stage eval --eval-stride "${EVAL_STRIDE:-16}" \
  --max-lead-steps 5 --n-ens-cases "${N_ENS_CASES:-24}" --members 8 --sample-steps 20 || exit 1
python -W ignore scripts/diagnose_guide.py --run "$RUN" --threads "$THREADS" || exit 1
echo PIPELINE_DONE
