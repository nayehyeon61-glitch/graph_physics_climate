#!/usr/bin/env bash
# External benchmark, 2020 test year (감자 2026-11-04 21:32 KST: "climate ode 말고 그 수준에서 최신 비교 가능한 모델들과도
# 비교"): our best runs and ClimODE are scored on the WeatherBench2 2020 protocol (88 init times 2020-01-01..12-14
# 00 UTC every 4 days, leads 24-120 h, ERA5 64x32 truth from data/era5_wb2_64x32_2020test.npz) so that they sit next
# to the WB2 64x32 forecasts of GraphCast, Keisler GNN, SphericalCNN, NeuralGCM, FuXi, GenCast (ens. mean), Pangu,
# HRES, IFS ENS mean (runs/wb2_<model>_2020, scored by scripts/eval_wb2_forecasts.py with the same cases).
# Not a stage (no model change): evaluation only.  Runs after the stage9 chain (requeued 2026-11-05 13:40 KST;
# stage7c runs after both benchmarks and scores itself on 2020/2022) so it never competes with training.
cd "$(dirname "$0")/../.."
LOG=runs/bench_wb2_2020/chain_bench2020.log
D=data/era5_wb2_64x32_2020test.npz
until grep -q CHAIN_S9_DONE runs/era5_64x32_s9_integrated/chain_s9.log 2>/dev/null; do sleep 60; done
echo "[$(date -u +%H:%M:%S)] stage9 chain finished -> 2020 benchmark evaluation" >> $LOG
for RUN in runs/era5_64x32_s9_integrated runs/era5_64x32_s7b_steps3x runs/era5_64x32_s12b_width64 runs/era5_64x32_s14a_prev runs/era5_64x32_physops_roll; do
  [ -f $RUN/vnet_done.json ] || continue
  python -W ignore scripts/train_eval.py --run $RUN --threads 4 --stage eval --eval-stride 16 --max-lead-steps 5 \
    --n-ens-cases 0 --eval-data $D --eval-tag 2020 --max-cases 88 >> $LOG 2>&1
  echo "[$(date -u +%H:%M:%S)] $RUN eval_2020 exit $?" >> $LOG
done
python -W ignore scripts/eval_climode.py --data $D --out runs/climode_2020_trainmode --train-mode --batch-ics 2 \
  --threads 4 --max-cases 88 --members 0 >> $LOG 2>&1
echo "[$(date -u +%H:%M:%S)] ClimODE 2020 exit $? CHAIN_BENCH2020_DONE" >> $LOG
