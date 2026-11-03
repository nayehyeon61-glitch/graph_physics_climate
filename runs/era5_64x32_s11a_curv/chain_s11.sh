#!/usr/bin/env bash
# Stage chain stage11 (2026-11-03, 감자: "lie 들의 어긋남 즉 connection 과 curvature 에 대한 graph의 확장적 연결").
#   stage11a runs/era5_64x32_s11a_curv  stage3 recipe (base11, gated fusion, physics operators, tendency head,
#            Stage A forecast-state perturbation, Stage B 2-step rollout from 500, A 1000 / B 1500 steps, 2016-2017)
#            + exact connection / curvature operators of the flow as extra guide inputs (--curvature-ops 1:
#            DEC vorticity / divergence / absolute curvature, strain, Okubo-Weiss, Helmholtz psi/chi, rotational and
#            divergent wind, Lie brackets of the wind levels).  Design: docs/design/Connection_Curvature_Graph_KO.md.
# The run's config.json is created here from the stage3 config with curvature_ops=1 (run_guide_ablation.sh is not
# edited while other chains execute it); the stage3 normaliser is reused (same data).  Compare with stage3 (+ ClimODE).
cd "$(dirname "$0")/../.."
RUN=runs/era5_64x32_s11a_curv
S3=runs/era5_64x32_physops_roll
LOG=$RUN/chain_s11.log
until grep -q CHAIN_S10_DONE runs/era5_64x32_s10a_fcn21/chain_s10.log 2>/dev/null; do sleep 60; done
echo "[$(date -u +%H:%M:%S)] stage10 chain finished -> stage11a connection/curvature operators" >> $LOG
[ -f $RUN/config.json ] || python - $S3/config.json $RUN/config.json <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c.update(curvature_ops=1, integrator=c.get("integrator", "delta"), ode_steps=c.get("ode_steps", 4),
         ode_solver=c.get("ode_solver", "euler"), ode_physics=c.get("ode_physics", 0), spec=c.get("spec", "base11"))
json.dump(c, open(sys.argv[2], "w"), indent=2)
PY
[ -f $RUN/normalizer.pt ] || cp $S3/normalizer.pt $RUN/
RUN=$RUN BASE=$S3 PERTURB_MODEL=$S3/vnet_noguide.pt PERTURB_PROB=0.3 ROLLOUT_STEPS=2 ROLLOUT_START=500 \
  bash scripts/run_guide_ablation.sh > $RUN/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%S)] stage11a pipeline exit $? CHAIN_S11_DONE" >> $LOG
