#!/usr/bin/env bash
# Stage stage22 (2026-11-09 16:07 KST; 감자 16:00 KST "https://github.com/MPAS-Dev/MPAS-Model 이것처럼 기후에 대한 보르노이와
# 들로네도 참고해 보세요" -> decision card "A: stage22 TRiSK -> guide"): the MPAS / TRiSK C-grid quantities of the flow
# (Ringler, Thuburn, Klemp, Skamarock 2010; Skamarock et al. 2012) computed on the guide's finest mesh (icosahedral
# nodes = Voronoi cells, triangles = dual vertices) as fixed guide inputs through a zero-initialised adapter:
# edge normal velocity -> TRiSK kinetic energy K_i, vertex absolute vorticity / PV q_v = (zeta + f) / h (h = Phi_500 / g),
# the PV-flux (nonlinear Coriolis) term, the Bernoulli gradient grad(K + Phi), their sum (vector-invariant tendency),
# the 500 hPa mass-flux divergence and the edge-flux advective tendencies of t850 / q700 (gpclimate/nn/trisk_ops.py).
# Parent = the stage19 sub-stage with the largest mean gain among those adopted against the stage18 base (else that
# base); config = parent + trisk_ops; Stage A = the parent's recipe (heads + adapter only, trunk frozen, 17c
# perturbations, the parent's PINN weight) from the parent's guide; Stage B = paired fine-tuning (V-Net from stage9b,
# 1000 steps) with the parent's Stage B flags.  Decided against the parent (and the base when different).
# Design: docs/design/Stage22_TRiSK_Guide_Features_KO.md, reference docs/design/MPAS_Dual_Mesh_Reference_KO.md
cd "$(dirname "$0")/../.."
S9B=runs/era5_64x32_s9b_merged
S17A=runs/era5_64x32_s17a_dyn
S19=runs/era5_64x32_s19_gpinn
S21=runs/era5_64x32_s21_physfirst
RUN=runs/era5_64x32_s22_trisk
LOG=$RUN/chain_s22.log
echo "[$(date -u +%H:%M:%S)] stage22 waits for CHAIN_S19_DONE" >> $LOG
until grep -q "CHAIN_S19_DONE" $S19/chain_s19.log 2>/dev/null; do sleep 120; done
BASE=$(cat $S21/base.txt); BFLAGS=$(cat $S21/bflags.txt)
PARENT=$(python scripts/pick_parent.py --base $BASE runs/era5_64x32_s19a_gpinn runs/era5_64x32_s19b_gpinn \
         runs/era5_64x32_s19c_gpinn)
PA=""; PB=""
case $PARENT in *s19*) PA="--pinn-weight 0.05";; esac
case $PARENT in *s19c*) PB="--pinn-weight-b 0.02";; esac
echo "$PARENT" > $RUN/parent.txt; echo "$BASE" > $RUN/base.txt; echo "$BFLAGS $PB" > $RUN/bflags.txt
echo "$PB" > $RUN/extra_b.txt
echo "[$(date -u +%H:%M:%S)] stage22 start: parent $PARENT (A '$PA', B '$BFLAGS $PB'), base $BASE" >> $LOG
for f in normalizer.pt column_norm.json; do [ -f $RUN/$f ] || cp $S9B/$f $RUN/; done
for f in vnet_noguide.pt vnet_noguide_done.json log_vnet_noguide.jsonl tv_eps.json graph_embed.npy pinn_scales.json; do
  if [ ! -f $RUN/$f ] && [ -f $PARENT/$f ]; then cp $PARENT/$f $RUN/; fi
done
[ -f $RUN/config.json ] || python - "$PARENT/config.json" "$RUN/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c.update(trisk_ops=1)
json.dump(c, open(sys.argv[2], "w"), indent=2)
PY
FT="--threads 4 --batch 4 --lr 3e-4 --steps 1000 --val-every 250 --rollout-steps 2 --rollout-start 0 --rollout-weight 1.0"
EV="--threads 4 --stage eval --eval-stride 16 --max-lead-steps 5"
A="--threads 4 --batch 4 --lr 5e-4 --stage a --steps 1000 --val-every 200 --init-guide $PARENT/guide.pt --guide-heads-only 1 \
   --perturb-model $S17A/vnet_noguide.pt --perturb-prob 0.3 $PA"
{
python -W ignore scripts/train_eval.py --run $RUN $A &&
python -W ignore scripts/train_eval.py --run $RUN $FT --stage b --init-vnet $S9B/vnet.pt $BFLAGS $PB &&
python -W ignore scripts/train_eval.py --run $RUN $EV --n-ens-cases 0 &&
python -W ignore scripts/diagnose_guide.py --run $RUN --threads 4 && echo PIPELINE_DONE
} >> $RUN/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] stage22 pipeline exit $?" >> $LOG
for Y in 2020 2022; do
  python -W ignore scripts/train_eval.py --run $RUN $EV --n-ens-cases 0 --eval-data data/era5_wb2_64x32_${Y}test.npz \
    --eval-column-data data/era5_wb2_64x32_column10_${Y}test.npz --eval-tag $Y --max-cases 88 > $RUN/eval_$Y.log 2>&1
  echo "[$(date -u +%H:%M:%S)] stage22 eval_$Y exit $?" >> $LOG
done
echo "stage22 physics metrics (run, parent):" >> $LOG
python scripts/core_physics_summary.py $RUN/eval.json $PARENT/eval.json >> $LOG 2>&1
echo "stage22 vs parent $PARENT:" >> $LOG
OKP=0
if python scripts/decide_adoption.py $RUN/eval.json $PARENT/eval.json --min-mean-gain 0.5 >> $LOG; then OKP=1; fi
OKB=$OKP
if [ "$BASE" != "$PARENT" ]; then
  echo "stage22 vs base $BASE:" >> $LOG
  OKB=0
  if python scripts/decide_adoption.py $RUN/eval.json $BASE/eval.json --min-mean-gain 0.5 >> $LOG; then OKB=1; fi
fi
if [ $OKP = 1 ] && [ $OKB = 1 ]; then echo "stage22 ADOPTED" >> $LOG; else echo "stage22 REJECTED" >> $LOG; fi
echo "[$(date -u +%H:%M:%S)] stage22 done CHAIN_S22_DONE" >> $LOG
