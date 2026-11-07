#!/usr/bin/env bash
# External benchmark, 2022 test year (감자 2026-11-04 22:05 KST: "25-26년도의 모델은 없나요?"): the 2025 models that
# WeatherBench2 hosts at 64x32 (Aurora, Nature 2025; FGN = WeatherNext 2, 2025) only cover 2022, so our best runs and
# ClimODE are also scored on 2022 (88 init times 2022-01-02..12-16 00 UTC every 4 days, leads 24-120 h, ERA5 64x32
# truth from data/era5_wb2_64x32_2022test.npz; Aurora has no 2022-01-01 00 UTC forecast, hence the 01-02 start)
# next to runs/wb2_<model>_2022 (aurora, fgn_mean, graphcast, pangu, hres, ens_mean; scripts/eval_wb2_forecasts.py).
# Not a stage (no model change): evaluation only.  Runs after the 2020 benchmark chain so it never competes with training.
cd "$(dirname "$0")/../.."
LOG=runs/bench_wb2_2022/chain_bench2022.log
D=data/era5_wb2_64x32_2022test.npz
until grep -q CHAIN_BENCH2020_DONE runs/bench_wb2_2020/chain_bench2020.log 2>/dev/null; do sleep 60; done
echo "[$(date -u +%H:%M:%S)] 2020 benchmark finished -> 2022 benchmark evaluation" >> $LOG
for RUN in runs/era5_64x32_s9_integrated runs/era5_64x32_s7b_steps3x runs/era5_64x32_s12b_width64 runs/era5_64x32_s14a_prev runs/era5_64x32_physops_roll; do
  [ -f $RUN/vnet_done.json ] || continue
  python -W ignore scripts/train_eval.py --run $RUN --threads 4 --stage eval --eval-stride 16 --max-lead-steps 5 \
    --n-ens-cases 0 --eval-data $D --eval-tag 2022 --max-cases 88 >> $LOG 2>&1
  echo "[$(date -u +%H:%M:%S)] $RUN eval_2022 exit $?" >> $LOG
done
python -W ignore scripts/eval_climode.py --data $D --out runs/climode_2022_trainmode --train-mode --batch-ics 2 \
  --threads 4 --max-cases 88 --members 0 >> $LOG 2>&1
echo "[$(date -u +%H:%M:%S)] ClimODE 2022 exit $? CHAIN_BENCH2022_DONE" >> $LOG
