#!/usr/bin/env python
"""CPU smoke test: Stage A -> Stage B -> Stage C on a tiny ERA5 subset (WeatherBench2 64x32).

This checks that the pipeline runs end to end, that the gauge-equivariance
errors are at floating-point level, and that the training losses move.  A few
dozen CPU steps on a few weeks of data cannot establish forecast skill; the
persistence comparison is reported only as a sanity reference.

    python scripts/smoke_test.py --fetch            # downloads ~0.2 GB once
    python scripts/smoke_test.py --data data/era5_wb2_64x32_smoke.npz
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpclimate.data.era5_wb2 import DEFAULT_SPEC, ERA5Pairs, fetch_wb2_subset, fit_normalizer  # noqa: E402
from gpclimate.data.variables import merge_state  # noqa: E402
from gpclimate.external import repos  # noqa: E402
from gpclimate.geometry.graphs import SphereGraphs  # noqa: E402
from gpclimate.models.diffusion import (DiffusionConfig, GraphResidualDiffusion, ResidualDenoiser,  # noqa: E402
                                        assemble_ensemble)
from gpclimate.models.dynamics_vnet import LieGuidedForecaster, VNetConfig  # noqa: E402
from gpclimate.models.lie_physical_gnn import GuideConfig  # noqa: E402
from gpclimate.nn.so2 import Field  # noqa: E402
from gpclimate.training.losses import (field_mse, gauge_equivariance_report, global_tendency_consistency,  # noqa: E402
                                       perturb, random_gauges, sample_sigma)
from gpclimate.training.metrics import wb2_scores  # noqa: E402

PERIODS = {"train": ("2018-01-01", "2018-01-24T18"), "calib": ("2019-01-01", "2019-01-10T18"),
           "test": ("2020-01-01", "2020-01-10T18")}
EVAL_VARS = ["z500", "t850", "t2m", "msl", "u850", "v850", "u10", "v10"]


def n_params(m: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable_only))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/era5_wb2_64x32_smoke.npz"))
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--mesh", type=int, nargs=3, default=[4, 3, 2])
    ap.add_argument("--lead", type=int, default=4, help="lead in 6h steps (4 = 24 h, PG-Flow daily step)")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps-a", type=int, default=40)
    ap.add_argument("--steps-b", type=int, default=40)
    ap.add_argument("--steps-c", type=int, default=40)
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--sample-steps", type=int, default=10)
    ap.add_argument("--n-test", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablation", action="store_true", help="also train a V-Net without the physics guide")
    ap.add_argument("--out", default=str(ROOT / "reports/smoke_test.json"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    report: dict = {"args": vars(args), "repos": repos.available(), "torch": torch.__version__}
    t_start = time.time()

    if args.fetch or not Path(args.data).exists():
        log(f"fetching ERA5 subset from {PERIODS} ...")
        t0 = time.time()
        fetch_wb2_subset(args.data, PERIODS)
        report["fetch_seconds"] = time.time() - t0
    spec = DEFAULT_SPEC
    train = ERA5Pairs(args.data, "train", spec, args.lead)
    calib = ERA5Pairs(args.data, "calib", spec, args.lead)
    test = ERA5Pairs(args.data, "test", spec, args.lead)
    report["data"] = {"grid": [len(train.lat), len(train.lon)], "pairs": {"train": len(train), "calib": len(calib),
                      "test": len(test)}, "periods": PERIODS, "lead_hours": 6 * args.lead,
                      "scalars": list(spec.scalars), "vectors": [list(p) for p in spec.vectors]}
    log(f"data: {report['data']}")

    graphs = SphereGraphs(train.lat, train.lon, mesh_splits=args.mesh)
    g = graphs.view()
    w = g.grid_area_weight
    report["graph"] = graphs.summary()
    log(f"graph: {report['graph']}")

    norm = fit_normalizer(train)

    def prep(ds: ERA5Pairs, idx):
        x0, x1, c = ds.batch(idx)
        return norm.state(x0), norm.state(x1), norm.cond(c), x0

    guide_cfg = GuideConfig(cs=32, cv=8, depth=(1, 1, 1, 1, 1), n_basis=8, hidden=48)
    vnet_cfg = VNetConfig(cs=32, cv=8, depth=(1, 1, 1, 1, 1), heads=2, ds=8, dv=4, n_basis=8, hidden=48)
    model = LieGuidedForecaster(spec, guide_cfg, vnet_cfg)
    model.delta_scale_s.copy_(norm.delta_s)
    model.delta_scale_v.copy_(norm.delta_v)
    guide = model.guide
    report["params"] = {"guide": n_params(guide), "vnet": n_params(model.vnet)}

    # ------------------------------------------------------------- equivariance (untrained, random init)
    x0n, x1n, cn, _ = prep(train, [0])
    lv = {"P0": "0", "P1": "1", "P2": "2", "P1u": "1", "P0u": "0", "rec": "grid", "aux": "grid"}

    def guide_fn(x, c, gv):
        feats = guide(x, c, torch.zeros(1), gv)
        rec, aux = guide.heads(feats, gv)
        return {**feats, "rec": rec, "aux": Field(aux, aux.new_zeros(*aux.shape[:-1], 0).to(torch.complex64))}

    def forecaster_fn(x, c, gv):
        f, d = model(x, c, gv)
        return {"forecast": f, "delta": d}

    # (the untrained forecaster is skipped: its delta head starts at zero, which would pass trivially)
    eq = {"guide": gauge_equivariance_report(guide_fn, x0n, cn, graphs, lv)}
    log(f"equivariance (untrained): {eq}")

    # ------------------------------------------------------------- Stage A
    opt = torch.optim.Adam(guide.parameters(), lr=args.lr)
    hist_a = []
    gen = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    for step in range(args.steps_a):
        idx = rng.integers(0, len(train), args.batch)
        x0n, _, cn, x0 = prep(train, idx)
        aux_t = train.aux_targets(x0) / norm.aux_std
        tg, tm = random_gauges(graphs, gen)  # T_g: random gauge (local frame) transformation
        gv = graphs.gauge_transformed(tm, tg)
        xg, cg = x0n.gauge(tg), cn.gauge(tg)
        sigma = sample_sigma(args.batch, 0.5, generator=gen)
        feats = guide(perturb(xg, sigma, gen), cg, sigma, gv)
        rec, aux = guide.heads(feats, gv)
        l_rec = field_mse(rec, xg, w)  # target T_g(X): keep the gauge change, remove the perturbation
        l_aux = ((aux - aux_t) ** 2 * w[None, :, None]).mean() / w.mean()
        loss = l_rec + 0.5 * l_aux
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(guide.parameters(), 1.0)
        opt.step()
        hist_a.append([float(l_rec), float(l_aux)])
        if step % 10 == 0 or step == args.steps_a - 1:
            log(f"A step {step:3d}  L_rec {float(l_rec):.4f}  L_aux {float(l_aux):.4f}")
    t_a = time.time() - t0
    with torch.no_grad():
        idx = list(range(0, len(calib), max(1, len(calib) // 4)))[:4]
        x0n, _, cn, x0 = prep(calib, idx)
        feats = guide(x0n, cn, torch.zeros(len(idx)), g)
        rec, aux = guide.heads(feats, g)
        val_rec = float(field_mse(rec, x0n, w))
        val_aux = float((((aux - calib.aux_targets(x0) / norm.aux_std) ** 2) * w[None, :, None]).mean() / w.mean())
    report["stage_a"] = {"steps": args.steps_a, "seconds": t_a, "sec_per_step": t_a / max(1, args.steps_a),
                         "loss_first5": np.mean(hist_a[:5], 0).tolist(), "loss_last5": np.mean(hist_a[-5:], 0).tolist(),
                         "calib_rec_mse_sigma0": val_rec, "calib_aux_mse": val_aux}
    log(f"stage A: {report['stage_a']}")

    # ------------------------------------------------------------- Stage B (guide weights frozen)
    def train_stage_b(m: LieGuidedForecaster, tag: str):
        m.freeze_guide()
        opt = torch.optim.Adam([p for p in m.vnet.parameters() if p.requires_grad], lr=args.lr)
        rng_b = np.random.default_rng(args.seed + 1)  # identical batches for the main model and the ablation
        hist = []
        t0 = time.time()
        for step in range(args.steps_b):
            idx = rng_b.integers(0, len(train), args.batch)
            x0n, x1n, cn, _ = prep(train, idx)
            fc, d = m(x0n, cn, g)
            target = Field((x1n.s - x0n.s) / m.delta_scale_s, (x1n.v - x0n.v) / m.delta_scale_v)
            l_f = field_mse(d, target, w)
            l_phys = global_tendency_consistency(d.s * m.delta_scale_s, x1n.s - x0n.s, w)
            loss = l_f + 0.1 * l_phys
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.vnet.parameters(), 1.0)
            opt.step()
            hist.append([float(l_f), float(l_phys)])
            if step % 10 == 0 or step == args.steps_b - 1:
                log(f"B[{tag}] step {step:3d}  L_forecast {float(l_f):.4f}  L_phys {float(l_phys):.5f}")
        t_b = time.time() - t0
        out = {"steps": args.steps_b, "seconds": t_b, "sec_per_step": t_b / max(1, args.steps_b),
               "loss_first5": np.mean(hist[:5], 0).tolist(), "loss_last5": np.mean(hist[-5:], 0).tolist()}
        if m.vnet.fusion is not None:
            out["fusion_gates_mean_abs"] = [float(f.beta_s.abs().mean()) for f in m.vnet.fusion]
        m.eval()
        return out

    report["stage_b"] = train_stage_b(model, "lie-guided")
    log(f"stage B: {report['stage_b']}")
    ablation = None
    if args.ablation:  # design doc section 10, first row: plain Dynamics GNN vs Lie-guided model
        torch.manual_seed(args.seed)
        ablation = LieGuidedForecaster(spec, guide_cfg, VNetConfig(**{**vnet_cfg.__dict__, "use_guide": False}))
        ablation.delta_scale_s.copy_(norm.delta_s)
        ablation.delta_scale_v.copy_(norm.delta_v)
        report["params"]["vnet_no_guide"] = n_params(ablation.vnet)
        report["stage_b_no_guide"] = train_stage_b(ablation, "no-guide")
        for p in ablation.parameters():
            p.requires_grad_(False)
    model.eval()

    # ------------------------------------------------------------- Stage C (A + B frozen), trained on the calib period
    for p in model.parameters():
        p.requires_grad_(False)
    dcfg = DiffusionConfig(cs=32, cv=8, depth=2, n_basis=8, hidden=48, n_train_steps=1000,
                           sample_steps=args.sample_steps)
    diff = GraphResidualDiffusion(ResidualDenoiser(spec, dcfg, guide_dims=(guide_cfg.cs, guide_cfg.cv)), 1000)
    report["params"]["diffusion"] = n_params(diff)

    def context(x0n, cn):
        with torch.no_grad():
            P = model.physics(x0n, cn, g)
            xbar, _ = model(x0n, cn, g)
        return xbar, {"xbar": xbar, "x": x0n, "cond": cn, "P0u": P["P0u"]}

    with torch.no_grad():  # residual scale from a period the deterministic model did not train on
        idx = list(range(len(calib)))
        res_s, res_v = [], []
        for i in range(0, len(idx), 4):
            x0n, x1n, cn, _ = prep(calib, idx[i:i + 4])
            xbar, _ = model(x0n, cn, g)
            res_s.append(x1n.s - xbar.s)
            res_v.append(x1n.v - xbar.v)
        rs, rv = torch.cat(res_s), torch.cat(res_v)
        diff.res_scale_s.copy_(rs.std((0, 1)) + 1e-6)
        diff.res_scale_v.copy_(torch.sqrt((rv.abs() ** 2).mean((0, 1)) / 2) + 1e-6)
    opt = torch.optim.Adam(diff.denoiser.parameters(), lr=args.lr)
    hist_c = []
    t0 = time.time()
    for step in range(args.steps_c):
        idx = rng.integers(0, len(calib), args.batch)
        x0n, x1n, cn, _ = prep(calib, idx)
        xbar, ctx = context(x0n, cn)
        loss = diff.loss(diff.to_residual(x1n, xbar), ctx, g, w)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(diff.parameters(), 1.0)
        opt.step()
        hist_c.append(float(loss))
        if step % 10 == 0 or step == args.steps_c - 1:
            log(f"C step {step:3d}  L_eps {float(loss):.4f}")
    t_c = time.time() - t0
    report["stage_c"] = {"steps": args.steps_c, "seconds": t_c, "sec_per_step": t_c / max(1, args.steps_c),
                         "loss_first5": float(np.mean(hist_c[:5])), "loss_last5": float(np.mean(hist_c[-5:]))}
    log(f"stage C: {report['stage_c']}")

    # denoiser equivariance (fixed diffusion time, after training)
    x0n, x1n, cn, _ = prep(test, [0])
    xbar, ctx = context(x0n, cn)
    r = diff.to_residual(x1n, xbar)

    def denoiser_fn(rr, c, gv):
        P = model.physics(ctx["x"] if gv is g else ctx_g["x"], c, gv)
        cc = {"xbar": ctx["xbar"] if gv is g else ctx_g["xbar"], "x": ctx["x"] if gv is g else ctx_g["x"],
              "cond": c, "P0u": P["P0u"]}
        return {"eps": diff.denoiser(rr, torch.tensor([500]), cc, gv)}

    gen_eq = torch.Generator().manual_seed(123)
    tg, tm = random_gauges(graphs, gen_eq)
    ctx_g = {"x": ctx["x"].gauge(tg), "xbar": ctx["xbar"].gauge(tg)}
    with torch.no_grad():
        e0 = denoiser_fn(r, cn, g)["eps"]
        gv = graphs.gauge_transformed(tm, tg)
        e1 = denoiser_fn(r.gauge(tg), cn.gauge(tg), gv)["eps"]
    from gpclimate.training.losses import relative_error
    eq["denoiser"] = {"eps": relative_error(e1, e0.gauge(tg))}

    # trained-model equivariance
    eq["forecaster_trained"] = gauge_equivariance_report(forecaster_fn, x0n, cn, graphs,
                                                         {"forecast": "grid", "delta": "grid"}, seed=7)
    report["equivariance_rel_error"] = eq
    log(f"equivariance: {eq}")

    # ------------------------------------------------------------- test evaluation (WeatherBench2 metrics)
    n_test = min(args.n_test, len(test))
    idx = list(np.linspace(0, len(test) - 1, n_test).round().astype(int))
    H, W = len(test.lat), len(test.lon)
    det, pers, truth, ens, abl = {}, {}, {}, {}, {}
    t0 = time.time()
    with torch.no_grad():
        for i in idx:
            x0n, x1n, cn, _ = prep(test, [i])
            xbar, ctx = context(x0n, cn)
            members = assemble_ensemble(xbar, diff.sample(ctx, g, args.members, args.sample_steps, seed=int(i)))
            for name, f in (("det", xbar), ("pers", x0n), ("truth", x1n)):
                phys = merge_state(norm.state_inv(f), spec, H, W)
                tgt = {"det": det, "pers": pers, "truth": truth}[name]
                for k in EVAL_VARS:
                    tgt.setdefault(k, []).append(phys[k][0].numpy())
            if ablation is not None:
                fa, _ = ablation(x0n, cn, g)
                phys = merge_state(norm.state_inv(fa), spec, H, W)
                for k in EVAL_VARS:
                    abl.setdefault(k, []).append(phys[k][0].numpy())
            for m, f in enumerate(members):
                phys = merge_state(norm.state_inv(f), spec, H, W)
                for k in EVAL_VARS:
                    ens.setdefault(k, [[] for _ in range(args.members)])[m].append(phys[k][0].numpy())
        # 2-step rollout sanity check (physics features recomputed from each predicted state)
        x0n, _, cn, _ = prep(test, [0])
        roll = model.rollout(x0n, [cn, cn], g)
        roll_finite = all(bool(torch.isfinite(f.s).all() and torch.isfinite(f.v.real).all()) for f in roll)
    t_eval = time.time() - t0
    times = test.times[np.array(idx) + args.lead].astype("datetime64[ns]")
    stack = lambda d: {k: np.stack(v, 0) for k, v in d.items()}  # noqa: E731
    sc_det = wb2_scores(stack(det), stack(truth), test.lat, test.lon, times,
                        ensemble={k: np.stack([np.stack(mm, 0) for mm in v], 0) for k, v in ens.items()})
    sc_per = wb2_scores(stack(pers), stack(truth), test.lat, test.lon, times)
    report["test"] = {"n_cases": n_test, "members": args.members, "sample_steps": args.sample_steps,
                      "seconds": t_eval, "valid_times": [str(t) for t in times.astype("datetime64[h]")],
                      "rmse_deterministic": sc_det["rmse"], "rmse_persistence": sc_per["rmse"],
                      "crps_ensemble": sc_det["crps"], "ens_mean_rmse": sc_det["ens_mean_rmse"],
                      "ens_spread": sc_det["ens_spread"], "rollout_2step_finite": roll_finite}
    if ablation is not None:
        report["test"]["rmse_no_guide"] = wb2_scores(stack(abl), stack(truth), test.lat, test.lon, times)["rmse"]
    report["total_seconds"] = time.time() - t_start
    log(f"test: {json.dumps(report['test'], indent=1)}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    log(f"report written to {args.out}")


if __name__ == "__main__":
    main()
