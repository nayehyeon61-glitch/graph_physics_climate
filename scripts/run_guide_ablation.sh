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
# stage S7: DATA=path/to/other.npz trains on another data file (stored in config.json at run creation)
[ -n "${DATA:-}" ] && COMMON="$COMMON --data $DATA"
# stage S5a: closure-structured guide (fixed resolved dynamics + learned closure); RESOLVED_TERMS=adv_u,coriolis,...
COMMON="$COMMON --closure-tend ${CLOSURE_TEND:-0}"
[ -n "${RESOLVED_TERMS:-}" ] && COMMON="$COMMON --resolved-terms $RESOLVED_TERMS"
[ -n "${RESOLVED_COEF:-}" ] && COMMON="$COMMON --resolved-coef $RESOLVED_COEF"
A_EXTRA=""
[ -n "${PERTURB_MODEL:-}" ] && A_EXTRA="--perturb-model $PERTURB_MODEL --perturb-prob ${PERTURB_PROB:-0.3}"
B_EXTRA="--rollout-steps ${ROLLOUT_STEPS:-1} --rollout-start ${ROLLOUT_START:-0} --rollout-weight ${ROLLOUT_WEIGHT:-1.0}"
# stage S5b: PDE-residual loss in Stage B (PDE_WEIGHT>0); GUIDE_FROM=run_dir reuses that run's guide (skips Stage A)
B_EXTRA="$B_EXTRA --pde-weight ${PDE_WEIGHT:-0}"
if [ -n "${GUIDE_FROM:-}" ]; then
  for f in config.json guide.pt normalizer.pt log_a.jsonl; do [ -f "$RUN/$f" ] || cp "$GUIDE_FROM/$f" "$RUN/"; done
else
  python -W ignore scripts/train_eval.py $COMMON --stage a --steps "${STEPS_A:-1000}" --val-every 200 $A_EXTRA || exit 1
fi
# stage S5c: VNET_FROM=run_dir reuses that run's deterministic V-Nets (skips Stage B; the Noise layer is the only change)
if [ -n "${VNET_FROM:-}" ]; then
  for f in vnet.pt log_vnet.jsonl vnet_noguide.pt log_vnet_noguide.jsonl; do [ -f "$RUN/$f" ] || cp "$VNET_FROM/$f" "$RUN/"; done
else
  python -W ignore scripts/train_eval.py $COMMON --stage b --steps "${STEPS_B:-1500}" --val-every 250 $B_EXTRA || exit 1
fi
if [ "${TRAIN_NOGUIDE:-0}" = "1" ]; then
  python -W ignore scripts/train_eval.py $COMMON --stage b_noguide --steps "${STEPS_B:-1500}" --val-every 250 $B_EXTRA || exit 1
else
  [ -f "$RUN/vnet_noguide.pt" ] || cp "$BASE/vnet_noguide.pt" "$BASE/log_vnet_noguide.jsonl" "$RUN/"
fi
# Stage C (residual graph diffusion) is optional: STAGE_C=1; NOISE_CONSTRAINTS=1 = stage S5c (projection + penalties)
if [ "${STAGE_C:-0}" = "1" ]; then
  python -W ignore scripts/train_eval.py $COMMON --stage c --steps "${STEPS_C:-800}" --val-every 200 \
    --noise-constraints "${NOISE_CONSTRAINTS:-0}" --noise-water "${NOISE_WATER:-1}" \
    --noise-feas-weight "${NOISE_FEAS_WEIGHT:-1.0}" --noise-mean-weight "${NOISE_MEAN_WEIGHT:-1.0}" \
    --noise-mean-every "${NOISE_MEAN_EVERY:-5}" --noise-mean-members "${NOISE_MEAN_MEMBERS:-2}" \
    --noise-mean-steps "${NOISE_MEAN_STEPS:-4}" || exit 1
fi
python -W ignore scripts/train_eval.py --run "$RUN" --threads "$THREADS" --stage eval --eval-stride "${EVAL_STRIDE:-16}" \
  --max-lead-steps 5 --n-ens-cases "${N_ENS_CASES:-24}" --members "${MEMBERS:-8}" --sample-steps "${SAMPLE_STEPS:-20}" || exit 1
if [ "${DIAGNOSE:-1}" = "1" ]; then
  python -W ignore scripts/diagnose_guide.py --run "$RUN" --threads "$THREADS" || exit 1
fi
echo PIPELINE_DONE
