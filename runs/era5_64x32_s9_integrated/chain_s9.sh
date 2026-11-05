#!/usr/bin/env bash
# Stage stage9 (2026-11-05 13:40 KST): integrated structure (감자 2026-11-03 "결과 확인 후 이점들을 모은 통합 구조",
# 2026-11-05 13:22 KST "통합 9단계가 가장 기대되는데 언제 시작되나요?").  Only components with a confirmed advantage:
#   recipe   stage7b: 10 y data (2008-2017), Stage B 4500 steps, 2-step rollout from 1500, Stage A forecast-state
#            perturbation (stage3); guide frozen in Stage B (stage12c rejected fine-tuning)
#   V-Net    stage12b width cs/cv 64/16; stage4 PIGT fusion (1 layer, operator tokens); stage14a previous-change input
#   guide    stage11a connection/curvature operators (stage3 operators + tendency head)
#   noise    stage5c Stage C with projection + feasibility/mean constraints (800 steps)
# Excluded (ledger verdicts): PDE loss (needs the stage5a closure guide), ODE (stage8), fcn21 output (stage10),
# simplicial blocks (11b), EMA (12a), 3-step curriculum (12d, conditional: hurts 24 h), operator layer (13a),
# spectral loss (13b), nudging (14b).  Normaliser = stage7's (identical to stage7b).  Order as stage7: guide-free
# V-Net first (it is the guide-free pair and the Stage A perturbation model), Stage A, Stage B, Stage C, eval,
# diagnose.  Compare with stage7b (same data and steps) + ClimODE.  Design: docs/design/Stage9_Integrated_Structure_KO.md
cd "$(dirname "$0")/../.."
S7=runs/era5_64x32_s7_data10y
S7B=runs/era5_64x32_s7b_steps3x
RUN=runs/era5_64x32_s9_integrated
LOG=$RUN/chain_s9.log
echo "[$(date -u +%H:%M:%S)] stage9 integrated structure start" >> $LOG
[ -f $RUN/normalizer.pt ] || cp $S7/normalizer.pt $RUN/
[ -f "$RUN/config.json" ] || python - "$S7B/config.json" "$RUN/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c.update(cs=64, cv=16, fusion="pigt", pigt_layers=1, pigt_ops=1, curvature_ops=1, prev_state=1)
json.dump(c, open(sys.argv[2], "w"), indent=2)
PY
COMMON="--run $RUN --threads 4 --batch 4 --lr 1e-3"
B="--steps 4500 --val-every 250 --rollout-steps 2 --rollout-start 1500 --rollout-weight 1.0"
{
python -W ignore scripts/train_eval.py $COMMON --stage b_noguide $B &&
python -W ignore scripts/train_eval.py $COMMON --stage a --steps 1000 --val-every 200 \
  --perturb-model $RUN/vnet_noguide.pt --perturb-prob 0.3 &&
python -W ignore scripts/train_eval.py $COMMON --stage b $B &&
python -W ignore scripts/train_eval.py $COMMON --stage c --steps 800 --val-every 200 --noise-constraints 1 \
  --noise-water 1 --noise-feas-weight 1.0 --noise-mean-weight 1.0 --noise-mean-every 5 --noise-mean-members 2 \
  --noise-mean-steps 4 &&
python -W ignore scripts/train_eval.py --run $RUN --threads 4 --stage eval --eval-stride 16 --max-lead-steps 5 \
  --n-ens-cases 24 --members 8 --sample-steps 20 &&
python -W ignore scripts/diagnose_guide.py --run $RUN --threads 4 && echo PIPELINE_DONE
} >> $RUN/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] stage9 pipeline exit $? CHAIN_S9_DONE" >> $LOG
