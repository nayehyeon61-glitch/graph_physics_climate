#!/usr/bin/env bash
# Stage stage20 (2026-11-07 23:30 KST): integrated version of the stage17-19 advantages (감자 2026-11-07 23:10 KST:
# "마지막으로 이후 이점들을 모아서 stage 20으로 해서 통합 버젼 실험해보죠").
#   1. scripts/stage20_compose.py picks the configuration: the stage17-19 run with the largest mean gain against the
#      same-budget control stage17-ctl among the runs that pass the adoption rule against it (each run's config carries
#      the components adopted along its chain path).  Its guide (Stage A + PINN preprocessing + graph embedding) is
#      reused as is.
#   2. Paired training with the budget in budget.env (default 3000 steps from stage9b): guide-free pair, guided model
#      (+ the chosen run's Stage B options: guide-head fine-tuning, path PINN), Stage C residual diffusion with the
#      stage9b noise constraints, full evaluation with ensembles (CRPS), diagnose, 2020/2022.
#   3. Same-budget control (stage9b recipe, delta, both arms) unless the budget equals stage17-19's (then stage17-ctl
#      is the control and the chosen run's Stage B is reused).
#   4. Decisions: vs the same-budget control (adoption rule), vs stage9b and vs the chosen 1000-step run (reported),
#      comparison table with ClimODE: reports/compare_stage20.md.
# Design: docs/design/Stage20_Integrated_KO.md
cd "$(dirname "$0")/../.."
S9B=runs/era5_64x32_s9b_merged
CTL17=runs/era5_64x32_s17_ctl
RUN=runs/era5_64x32_s20_integrated
CTL=runs/era5_64x32_s20_ctl
LOG=$RUN/chain_s20.log
echo "[$(date -u +%H:%M:%S)] stage20 waits for CHAIN_S19_DONE" >> $LOG
until grep -q "CHAIN_S19_DONE" runs/era5_64x32_s19_gpinn/chain_s19.log 2>/dev/null; do sleep 300; done
. $RUN/budget.env
eval "$(python scripts/stage20_compose.py --out $RUN)"
echo "[$(date -u +%H:%M:%S)] stage20 start: chosen ${CHOSEN:-none} (${SRC:-}), $STEPS steps lr $LR init $INIT, B flags '$BFLAGS'" >> $LOG
if [ -z "$SRC" ]; then
  echo "stage20: no stage17-19 run passes the adoption rule against stage17-ctl -> the integrated model stays stage9b" >> $LOG
  echo "[$(date -u +%H:%M:%S)] stage20 done CHAIN_S20_DONE" >> $LOG
  exit 0
fi
REUSE=0; [ "$STEPS" = 1000 ] && [ "$INIT" = s9b ] && [ "$LR" = 3e-4 ] && REUSE=1
for f in normalizer.pt column_norm.json; do [ -f $RUN/$f ] || cp $S9B/$f $RUN/; done
for f in $FILES; do [ -f $RUN/$f ] || cp $SRC/$f $RUN/; done
[ -f $RUN/config.json ] || cp $SRC/config.json $RUN/config.json
if [ $REUSE = 1 ]; then  # same budget as stage17-19: the chosen run's Stage B is the stage20 Stage B
  for f in vnet.pt vnet_done.json log_vnet.jsonl vnet_noguide.pt vnet_noguide_done.json log_vnet_noguide.jsonl \
           guide_finetuned.pt; do [ -f $SRC/$f ] && [ ! -f $RUN/$f ] && cp $SRC/$f $RUN/; done
  CTL=$CTL17
fi
IV=""; IP=""
[ "$INIT" = s9b ] && IV="--init-vnet $S9B/vnet.pt" && IP="--init-vnet $S9B/vnet_noguide.pt"
FT="--threads 4 --batch 4 --lr $LR --steps $STEPS --val-every 250 --rollout-steps 2 --rollout-start $ROLLOUT_START --rollout-weight 1.0"
EV="--threads 4 --stage eval --eval-stride 16 --max-lead-steps 5"
C="--threads 4 --batch 4 --lr 1e-3 --stage c --steps 800 --val-every 200 --noise-constraints 1 --noise-water 1 \
   --noise-feas-weight 1.0 --noise-mean-weight 1.0 --noise-mean-every 5 --noise-mean-members 2 --noise-mean-steps 4"
{
python -W ignore scripts/train_eval.py --run $RUN $FT --stage b_noguide $IP &&
python -W ignore scripts/train_eval.py --run $RUN $FT --stage b $IV $BFLAGS &&
python -W ignore scripts/train_eval.py --run $RUN $C &&
python -W ignore scripts/train_eval.py --run $RUN $EV --n-ens-cases 24 --members 8 --sample-steps 20 &&
python -W ignore scripts/diagnose_guide.py --run $RUN --threads 4 && echo PIPELINE_DONE
} >> $RUN/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] stage20 pipeline exit $?" >> $LOG
for Y in 2020 2022; do
  python -W ignore scripts/train_eval.py --run $RUN $EV --n-ens-cases 0 --eval-data data/era5_wb2_64x32_${Y}test.npz \
    --eval-column-data data/era5_wb2_64x32_column10_${Y}test.npz --eval-tag $Y --max-cases 88 > $RUN/eval_$Y.log 2>&1
  echo "[$(date -u +%H:%M:%S)] stage20 eval_$Y exit $?" >> $LOG
done
if [ $REUSE = 0 ]; then  # same-budget control: stage9b recipe (delta, no new structure), both arms
  mkdir -p $CTL
  for f in normalizer.pt guide.pt a_done.json column_norm.json config.json; do [ -f $CTL/$f ] || cp $S9B/$f $CTL/; done
  {
  python -W ignore scripts/train_eval.py --run $CTL $FT --stage b_noguide $IP &&
  python -W ignore scripts/train_eval.py --run $CTL $FT --stage b $IV &&
  python -W ignore scripts/train_eval.py --run $CTL $EV --n-ens-cases 0 && echo PIPELINE_DONE
  } >> $CTL/pipeline.log 2>&1
  echo "[$(date -u +%H:%M:%S)] stage20 control pipeline exit $?" >> $LOG
  for Y in 2020 2022; do
    python -W ignore scripts/train_eval.py --run $CTL $EV --n-ens-cases 0 --eval-data data/era5_wb2_64x32_${Y}test.npz \
      --eval-column-data data/era5_wb2_64x32_column10_${Y}test.npz --eval-tag $Y --max-cases 88 > $CTL/eval_$Y.log 2>&1
    echo "[$(date -u +%H:%M:%S)] stage20 control eval_$Y exit $?" >> $LOG
  done
fi
echo "stage20 vs same-budget control $CTL:" >> $LOG
if python scripts/decide_adoption.py $RUN/eval.json $CTL/eval.json --min-mean-gain 0.5 >> $LOG; then
  echo "stage20 ADOPTED" >> $LOG
else
  echo "stage20 REJECTED" >> $LOG
fi
echo "stage20 vs stage9b (reported; budget differs):" >> $LOG
python scripts/decide_adoption.py $RUN/eval.json $S9B/eval.json --min-mean-gain 0.5 >> $LOG
echo "stage20 vs $CHOSEN ($SRC, 1000 steps):" >> $LOG
python scripts/decide_adoption.py $RUN/eval.json $SRC/eval.json --min-mean-gain 0.5 >> $LOG
python scripts/compare_runs.py --out reports/compare_stage20.md "stage20=$RUN:lie_guided" \
  "같은 예산 대조=$CTL:lie_guided" "$CHOSEN=$SRC:lie_guided" "stage9b=$S9B:lie_guided" \
  "stage20 guide 없음=$RUN:no_guide" "ClimODE=runs/climode_2019_trainmode:climode" >> $LOG 2>&1
echo "[$(date -u +%H:%M:%S)] stage20 done CHAIN_S20_DONE" >> $LOG
