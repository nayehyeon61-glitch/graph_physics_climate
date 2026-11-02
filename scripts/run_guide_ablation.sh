#!/usr/bin/env bash
# Guide-ablation run: Stage A (physics operators + tendency head) -> Stage B -> [Stage B no guide] -> eval -> diagnose.
# Unless TRAIN_NOGUIDE=1, the guide-free V-Net is reused from BASE (it does not depend on the guide; same data,
# normaliser and V-Net config).  Rollout training (ROLLOUT_STEPS>1) and forecast-state perturbation of Stage A
# (PERTURB_MODEL=path/to/vnet_noguide.pt) are optional.  Override RUN / BASE / STEPS_* via environment.
set -uo pipefail
cd "$(dirname "$0")/.."
RUN=${RUN:-runs/era5_64x32_physops}
BASE=${BASE:-runs/era5_64x32}
THREADS=${THREADS:-4}
mkdir -p "$RUN"
COMMON="--run $RUN --threads $THREADS --batch ${BATCH:-4} --lr ${LR:-1e-3} --physics-ops ${PHYSICS_OPS:-1} --tend-head ${TEND_HEAD:-1}"
A_EXTRA=""
[ -n "${PERTURB_MODEL:-}" ] && A_EXTRA="--perturb-model $PERTURB_MODEL --perturb-prob ${PERTURB_PROB:-0.3}"
B_EXTRA="--rollout-steps ${ROLLOUT_STEPS:-1} --rollout-start ${ROLLOUT_START:-0} --rollout-weight ${ROLLOUT_WEIGHT:-1.0}"
python -W ignore scripts/train_eval.py $COMMON --stage a --steps "${STEPS_A:-1000}" --val-every 200 $A_EXTRA || exit 1
python -W ignore scripts/train_eval.py $COMMON --stage b --steps "${STEPS_B:-1500}" --val-every 250 $B_EXTRA || exit 1
if [ "${TRAIN_NOGUIDE:-0}" = "1" ]; then
  python -W ignore scripts/train_eval.py $COMMON --stage b_noguide --steps "${STEPS_B:-1500}" --val-every 250 $B_EXTRA || exit 1
else
  [ -f "$RUN/vnet_noguide.pt" ] || cp "$BASE/vnet_noguide.pt" "$BASE/log_vnet_noguide.jsonl" "$RUN/"
fi
python -W ignore scripts/train_eval.py --run "$RUN" --threads "$THREADS" --stage eval --eval-stride "${EVAL_STRIDE:-16}" \
  --max-lead-steps 5 --n-ens-cases "${N_ENS_CASES:-24}" --members 8 --sample-steps 20 || exit 1
python -W ignore scripts/diagnose_guide.py --run "$RUN" --threads "$THREADS" || exit 1
echo PIPELINE_DONE
