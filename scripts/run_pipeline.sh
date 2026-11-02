#!/usr/bin/env bash
# Full staged run: A -> B -> B (no guide) -> C -> eval.  Override RUN / STEPS_* via environment.
set -uo pipefail
cd "$(dirname "$0")/.."
RUN=${RUN:-runs/era5_64x32}
THREADS=${THREADS:-4}
mkdir -p "$RUN"
COMMON="--run $RUN --threads $THREADS --batch ${BATCH:-4} --lr ${LR:-1e-3}"
python -W ignore scripts/train_eval.py $COMMON --stage a         --steps "${STEPS_A:-1000}" --val-every 200 || exit 1
python -W ignore scripts/train_eval.py $COMMON --stage b         --steps "${STEPS_B:-1500}" --val-every 250 || exit 1
python -W ignore scripts/train_eval.py $COMMON --stage b_noguide --steps "${STEPS_B:-1500}" --val-every 250 || exit 1
python -W ignore scripts/train_eval.py $COMMON --stage c         --steps "${STEPS_C:-800}"  --val-every 200 || exit 1
python -W ignore scripts/train_eval.py --run "$RUN" --threads "$THREADS" --stage eval --eval-stride "${EVAL_STRIDE:-16}" \
  --max-lead-steps 5 --n-ens-cases "${N_ENS_CASES:-24}" --members 8 --sample-steps 20 || exit 1
echo PIPELINE_DONE
