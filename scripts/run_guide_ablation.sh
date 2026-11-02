#!/usr/bin/env bash
# Strengthened-guide run: Stage A (physics operators + tendency head) -> Stage B -> eval.
# The guide-free V-Net is independent of the guide, so it is reused from BASE instead of being retrained
# (identical data, normaliser and V-Net config).  Override RUN / BASE / STEPS_* via environment.
set -uo pipefail
cd "$(dirname "$0")/.."
RUN=${RUN:-runs/era5_64x32_physops}
BASE=${BASE:-runs/era5_64x32}
THREADS=${THREADS:-4}
mkdir -p "$RUN"
[ -f "$RUN/vnet_noguide.pt" ] || cp "$BASE/vnet_noguide.pt" "$BASE/log_vnet_noguide.jsonl" "$RUN/"
COMMON="--run $RUN --threads $THREADS --batch ${BATCH:-4} --lr ${LR:-1e-3} --physics-ops ${PHYSICS_OPS:-1} --tend-head ${TEND_HEAD:-1}"
python -W ignore scripts/train_eval.py $COMMON --stage a --steps "${STEPS_A:-1000}" --val-every 200 || exit 1
python -W ignore scripts/train_eval.py $COMMON --stage b --steps "${STEPS_B:-1500}" --val-every 250 || exit 1
python -W ignore scripts/train_eval.py --run "$RUN" --threads "$THREADS" --stage eval --eval-stride "${EVAL_STRIDE:-16}" \
  --max-lead-steps 5 --n-ens-cases "${N_ENS_CASES:-24}" --members 8 --sample-steps 20 || exit 1
python -W ignore scripts/diagnose_guide.py --run "$RUN" --threads "$THREADS" || exit 1
echo PIPELINE_DONE
