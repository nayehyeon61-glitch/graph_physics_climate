#!/usr/bin/env bash
# Stage stage13a (2026-11-04 10:00 KST, 감자 제안 09:28 KST): continuous neural-operator layer after the discrete mesh
# V-Net — Lie guide -> mesh GNN -> **spherical neural operator** (neuraloperator SphericalConv / torch-harmonics SHT,
# all degrees l < 32, hidden 32, 2 FNO blocks, residual on the delta, zero-initialised).  Everything else = stage3
# (data 2016–2017, stage3 guide.pt and normaliser reused, Stage A skipped, Stage B 1500 steps, rollout 2 from 500);
# the guide-free baseline is retrained with the same layer.  Hypothesis: a global, resolution-independent spectral
# operator restores the synoptic scales (zonal wavenumbers 8–15) that the local mesh operators damp from 24 h on.
# Queued after stage12d; ≈3 h.  Compare with stage3 (+ ClimODE).
cd "$(dirname "$0")/../.."
S3=runs/era5_64x32_physops_roll
RUN=runs/era5_64x32_s13a_sfno
LOG=$RUN/chain_s13a.log
until grep -q CHAIN_S12D_DONE runs/era5_64x32_s12d_roll3/chain_s12d.log 2>/dev/null; do sleep 60; done
echo "[$(date -u +%H:%M:%S)] stage12d chain finished -> stage13a (spherical neural operator layer)" >> $LOG
for f in normalizer.pt guide.pt log_a.jsonl; do [ -f "$RUN/$f" ] || cp "$S3/$f" "$RUN/"; done
[ -f "$RUN/config.json" ] || python - "$S3/config.json" "$RUN/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c.update(neural_op="sht", no_hidden=32, no_layers=2, no_modes=[32, 64])
json.dump(c, open(sys.argv[2], "w"), indent=2)
PY
COMMON="--run $RUN --threads 4 --batch 4 --lr 1e-3"
B="--steps 1500 --val-every 250 --rollout-steps 2 --rollout-start 500 --rollout-weight 1.0"
{
python -W ignore scripts/train_eval.py $COMMON --stage b $B &&
python -W ignore scripts/train_eval.py $COMMON --stage b_noguide $B &&
python -W ignore scripts/train_eval.py --run $RUN --threads 4 --stage eval --eval-stride 16 --max-lead-steps 5 \
  --n-ens-cases 24 --members 8 --sample-steps 20 &&
python -W ignore scripts/diagnose_guide.py --run $RUN --threads 4 && echo PIPELINE_DONE
} > $RUN/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] stage13a pipeline exit $? CHAIN_S13A_DONE" >> $LOG
