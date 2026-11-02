#!/usr/bin/env python
"""Staged training and evaluation on an ERA5 subset (CPU or GPU).

    python scripts/train_eval.py --stage a          # Lie Physical GNN (guide)
    python scripts/train_eval.py --stage b          # Dynamics V-Net with frozen guide
    python scripts/train_eval.py --stage b_noguide  # ablation: same V-Net without physics guide
    python scripts/train_eval.py --stage c          # residual graph diffusion (A + B frozen)
    python scripts/train_eval.py --stage eval       # test-year rollout RMSE, ensemble CRPS, baselines

Each stage writes checkpoints and JSON-lines logs into ``--run``; later stages
load what earlier stages saved, so stages can run as separate processes.
Splits are time-disjoint: train (A, B), calib (model selection for A/B, Stage C
training, residual scale), test (evaluation only).
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

from gpclimate.data.era5_wb2 import DEFAULT_SPEC, ERA5Pairs, Normalizer, fit_normalizer, fit_tendency_scale  # noqa: E402
from gpclimate.data.variables import merge_state  # noqa: E402
from gpclimate.geometry.graphs import SphereGraphs  # noqa: E402
from gpclimate.models.diffusion import (DiffusionConfig, GraphResidualDiffusion, ResidualDenoiser,  # noqa: E402
                                        assemble_ensemble)
from gpclimate.models.dynamics_vnet import LieGuidedForecaster, VNetConfig  # noqa: E402
from gpclimate.models.lie_physical_gnn import GuideConfig  # noqa: E402
from gpclimate.nn.so2 import Field  # noqa: E402
from gpclimate.training.losses import (field_mse, global_tendency_consistency, perturb, random_gauges,  # noqa: E402
                                       sample_sigma)
from gpclimate.training.metrics import wb2_scores  # noqa: E402

EVAL_VARS = ["z500", "t850", "q700", "t2m", "msl", "u850", "v850", "u500", "v500", "u10", "v10"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def cosine_lr(step: int, total: int, lr: float, warmup: int = 100, floor: float = 0.02) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return lr * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))


class Run:
    """Everything the stages share: data, graphs, normaliser, model builders, checkpoint paths."""

    def __init__(self, args):
        self.args = args
        self.dir = Path(args.run)
        self.dir.mkdir(parents=True, exist_ok=True)
        cfg_path = self.dir / "config.json"
        if cfg_path.exists():
            self.cfg = json.load(open(cfg_path))
        else:
            self.cfg = {k: getattr(args, k) for k in ("data", "mesh", "lead", "cs", "cv", "depth", "heads", "ds", "dv",
                                                      "hidden", "n_basis", "guide_cs", "guide_cv", "diff_cs",
                                                      "diff_cv", "diff_depth", "physics_ops", "tend_head")}
            json.dump(self.cfg, open(cfg_path, "w"), indent=2)
        c = self.cfg
        self.spec = DEFAULT_SPEC
        self.train = ERA5Pairs(c["data"], "train", self.spec, c["lead"])
        self.calib = ERA5Pairs(c["data"], "calib", self.spec, c["lead"])
        self.test = ERA5Pairs(c["data"], "test", self.spec, c["lead"])
        self.graphs = SphereGraphs(self.train.lat, self.train.lon, mesh_splits=c["mesh"])
        self.g = self.graphs.view()
        self.w = self.g.grid_area_weight
        npath = self.dir / "normalizer.pt"
        if npath.exists():
            self.norm = Normalizer(**torch.load(npath))
            if self.norm.tend_s is None:  # runs created before the tendency head existed
                fit_tendency_scale(self.train, self.norm)
                self.save_normalizer(npath)
        else:
            self.norm = fit_normalizer(self.train)
            self.save_normalizer(npath)
        # flags added later default to off so older run directories keep loading their checkpoints
        self.guide_cfg = GuideConfig(cs=c["guide_cs"], cv=c["guide_cv"], depth=tuple(c["depth"]), n_basis=c["n_basis"],
                                     hidden=c["hidden"], physics_ops=bool(c.get("physics_ops", False)),
                                     tend_head=bool(c.get("tend_head", False)))
        self.vnet_kwargs = dict(cs=c["cs"], cv=c["cv"], depth=tuple(c["depth"]), heads=c["heads"], ds=c["ds"],
                                dv=c["dv"], n_basis=c["n_basis"], hidden=c["hidden"])
        self.diff_cfg = DiffusionConfig(cs=c["diff_cs"], cv=c["diff_cv"], depth=c["diff_depth"], n_basis=c["n_basis"],
                                        hidden=c["hidden"])

    def save_normalizer(self, path):
        torch.save({k: getattr(self.norm, k) for k in self.norm.__dataclass_fields__
                    if getattr(self.norm, k) is not None}, path)

    # ---------------------------------------------------------------- data helpers
    def prep(self, ds: ERA5Pairs, idx):
        x0, x1, c = ds.batch(idx)
        return self.norm.state(x0), self.norm.state(x1), self.norm.cond(c), x0

    def cond_at(self, ds: ERA5Pairs, idx):
        return self.norm.cond(ds.cond_field(torch.as_tensor(idx, dtype=torch.long)))

    # ---------------------------------------------------------------- models / checkpoints
    def forecaster(self, use_guide: bool) -> LieGuidedForecaster:
        m = LieGuidedForecaster(self.spec, self.guide_cfg, VNetConfig(use_guide=use_guide, **self.vnet_kwargs))
        m.delta_scale_s.copy_(self.norm.delta_s)
        m.delta_scale_v.copy_(self.norm.delta_v)
        if m.guide is not None and m.guide.phys_ops is not None:
            m.guide.phys_ops.v_scale.copy_(self.norm.v_scale)  # m/s per normalised unit (vorticity / 2 Omega)
        return m

    def load_guide(self, m: LieGuidedForecaster):
        m.guide.load_state_dict(torch.load(self.dir / "guide.pt"))

    def load_vnet(self, m: LieGuidedForecaster, name: str):
        m.vnet.load_state_dict(torch.load(self.dir / f"{name}.pt"))

    def diffusion(self) -> GraphResidualDiffusion:
        return GraphResidualDiffusion(ResidualDenoiser(self.spec, self.diff_cfg,
                                                       guide_dims=(self.guide_cfg.cs, self.guide_cfg.cv)), 1000)

    def jlog(self, stage: str, rec: dict):
        with open(self.dir / f"log_{stage}.jsonl", "a") as fh:
            fh.write(json.dumps(rec, default=float) + "\n")


# ---------------------------------------------------------------------------- Stage A
def stage_a(run: Run):
    a, g, w = run.args, run.g, run.w
    model = run.forecaster(True)
    guide = model.guide
    opt = torch.optim.Adam(guide.parameters(), lr=a.lr)
    gen = torch.Generator().manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    val_idx = list(np.linspace(0, len(run.calib) - 1, 16).round().astype(int))
    best, t0 = float("inf"), time.time()
    log(f"stage A: guide params {n_params(guide)}, steps {a.steps}, batch {a.batch}")
    for step in range(a.steps):
        for pg in opt.param_groups:
            pg["lr"] = cosine_lr(step, a.steps, a.lr)
        idx = rng.integers(0, len(run.train), a.batch)
        x0n, _, cn, x0 = run.prep(run.train, idx)
        aux_t = run.train.aux_targets(x0) / run.norm.aux_std
        tg, tm = random_gauges(run.graphs, gen)
        gv = run.graphs.gauge_transformed(tm, tg)
        xg, cg = x0n.gauge(tg), cn.gauge(tg)
        sigma = sample_sigma(a.batch, a.sigma_max, generator=gen)
        feats = guide(perturb(xg, sigma, gen), cg, sigma, gv)
        rec, aux, tend = guide.heads_all(feats, gv)
        l_rec = field_mse(rec, xg, w)
        l_aux = ((aux - aux_t) ** 2 * w[None, :, None]).mean() / w.mean()
        loss = l_rec + a.aux_weight * l_aux
        l_tend = torch.zeros(())
        if tend is not None:
            tend_t = run.norm.tendency(x0, run.train.next_state(idx)).gauge(tg)
            l_tend = field_mse(tend, tend_t, w)
            loss = loss + a.tend_weight * l_tend
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(guide.parameters(), 1.0)
        opt.step()
        if step % 25 == 0:
            run.jlog("a", {"step": step, "l_rec": float(l_rec), "l_aux": float(l_aux), "l_tend": float(l_tend),
                           "t": time.time() - t0})
        if step % a.val_every == 0 or step == a.steps - 1:
            guide.eval()
            with torch.no_grad():
                x0n, _, cn, x0 = run.prep(run.calib, val_idx)
                feats = guide(x0n, cn, torch.zeros(len(val_idx)), g)
                rec, aux, tend = guide.heads_all(feats, g)
                v_rec = float(field_mse(rec, x0n, w))
                v_aux = float((((aux - run.calib.aux_targets(x0) / run.norm.aux_std) ** 2) * w[None, :, None]).mean()
                              / w.mean())
                v_tend = 0.0
                if tend is not None:
                    v_tend = float(field_mse(tend, run.norm.tendency(x0, run.calib.next_state(val_idx)), w))
            guide.train()
            score = v_rec + a.aux_weight * v_aux + a.tend_weight * v_tend
            if score < best:
                best = score
                torch.save(guide.state_dict(), run.dir / "guide.pt")
            run.jlog("a", {"step": step, "val_rec": v_rec, "val_aux": v_aux, "val_tend": v_tend, "best": best,
                           "t": time.time() - t0})
            log(f"A {step:5d}  L_rec {float(l_rec):.4f} L_aux {float(l_aux):.4f} L_tend {float(l_tend):.4f}"
                f" | val rec {v_rec:.4f} aux {v_aux:.4f} tend {v_tend:.4f} | {(time.time() - t0) / (step + 1):.2f} s/step")
    log(f"stage A done in {(time.time() - t0) / 60:.1f} min, best val score {best:.4f}")


# ---------------------------------------------------------------------------- Stage B
def stage_b(run: Run, use_guide: bool):
    a, g, w = run.args, run.g, run.w
    name = "vnet" if use_guide else "vnet_noguide"
    model = run.forecaster(use_guide)
    if use_guide:
        run.load_guide(model)
        model.freeze_guide()
    params = [p for p in model.vnet.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=a.lr)
    rng = np.random.default_rng(a.seed + 1)
    val_idx = list(np.linspace(0, len(run.calib) - 1, 48).round().astype(int))
    best, t0 = float("inf"), time.time()
    log(f"stage B[{name}]: vnet params {n_params(model.vnet)}, steps {a.steps}, batch {a.batch}")
    for step in range(a.steps):
        for pg in opt.param_groups:
            pg["lr"] = cosine_lr(step, a.steps, a.lr)
        idx = rng.integers(0, len(run.train), a.batch)
        x0n, x1n, cn, _ = run.prep(run.train, idx)
        _, d = model(x0n, cn, g)
        target = Field((x1n.s - x0n.s) / model.delta_scale_s, (x1n.v - x0n.v) / model.delta_scale_v)
        l_f = field_mse(d, target, w)
        l_phys = global_tendency_consistency(d.s * model.delta_scale_s, x1n.s - x0n.s, w)
        loss = l_f + a.phys_weight * l_phys
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 25 == 0:
            run.jlog(name, {"step": step, "l_f": float(l_f), "l_phys": float(l_phys), "t": time.time() - t0})
        if step % a.val_every == 0 or step == a.steps - 1:
            model.eval()
            with torch.no_grad():
                tot = 0.0
                for i in range(0, len(val_idx), 8):
                    x0n, x1n, cn, _ = run.prep(run.calib, val_idx[i:i + 8])
                    fc, _ = model(x0n, cn, g)
                    tot += float(field_mse(fc, x1n, w)) * len(val_idx[i:i + 8])
                v_mse = tot / len(val_idx)
                x0n, x1n, cn, _ = run.prep(run.calib, val_idx[:8])
                pers = float(field_mse(x0n, x1n, w))
            model.train()
            if use_guide:
                model.guide.eval()
            if v_mse < best:
                best = v_mse
                torch.save(model.vnet.state_dict(), run.dir / f"{name}.pt")
            run.jlog(name, {"step": step, "val_mse": v_mse, "val_persistence_mse": pers, "best": best,
                            "t": time.time() - t0})
            log(f"B[{name}] {step:5d}  L_f {float(l_f):.4f} L_phys {float(l_phys):.5f} | val MSE {v_mse:.4f}"
                f" (persistence {pers:.4f}) | {(time.time() - t0) / (step + 1):.2f} s/step")
    log(f"stage B[{name}] done in {(time.time() - t0) / 60:.1f} min, best val MSE {best:.4f}")


# ---------------------------------------------------------------------------- Stage C
def stage_c(run: Run):
    a, g, w = run.args, run.g, run.w
    model = run.forecaster(True)
    run.load_guide(model)
    run.load_vnet(model, "vnet")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    diff = run.diffusion()
    rng = np.random.default_rng(a.seed + 2)
    with torch.no_grad():  # residual scale on the calib period (not seen by the deterministic model)
        idx = list(np.linspace(0, len(run.calib) - 1, 160).round().astype(int))
        rs, rv = [], []
        for i in range(0, len(idx), 8):
            x0n, x1n, cn, _ = run.prep(run.calib, idx[i:i + 8])
            xbar, _ = model(x0n, cn, g)
            rs.append(x1n.s - xbar.s)
            rv.append(x1n.v - xbar.v)
        rs, rv = torch.cat(rs), torch.cat(rv)
        diff.res_scale_s.copy_(rs.std((0, 1)) + 1e-6)
        diff.res_scale_v.copy_(torch.sqrt((rv.abs() ** 2).mean((0, 1)) / 2) + 1e-6)
    log(f"stage C: denoiser params {n_params(diff.denoiser)}, residual scale s {diff.res_scale_s.tolist()}")

    def context(x0n, cn):
        with torch.no_grad():
            P = model.physics(x0n, cn, g)
            xbar, _ = model(x0n, cn, g)
        return xbar, {"xbar": xbar, "x": x0n, "cond": cn, "P0u": P["P0u"]}

    opt = torch.optim.Adam(diff.denoiser.parameters(), lr=a.lr)
    t0, hist = time.time(), []
    for step in range(a.steps):
        for pg in opt.param_groups:
            pg["lr"] = cosine_lr(step, a.steps, a.lr)
        idx = rng.integers(0, len(run.calib), a.batch)
        x0n, x1n, cn, _ = run.prep(run.calib, idx)
        xbar, ctx = context(x0n, cn)
        loss = diff.loss(diff.to_residual(x1n, xbar), ctx, g, w)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(diff.parameters(), 1.0)
        opt.step()
        hist.append(float(loss))
        if step % 25 == 0:
            run.jlog("c", {"step": step, "l_eps": float(np.mean(hist[-25:])), "t": time.time() - t0})
        if step % a.val_every == 0 or step == a.steps - 1:
            log(f"C {step:5d}  L_eps(avg25) {np.mean(hist[-25:]):.4f} | {(time.time() - t0) / (step + 1):.2f} s/step")
    torch.save(diff.state_dict(), run.dir / "diffusion.pt")
    log(f"stage C done in {(time.time() - t0) / 60:.1f} min")


# ---------------------------------------------------------------------------- evaluation
def stage_eval(run: Run):
    a, g, test = run.args, run.g, run.test
    H, W, lead = len(test.lat), len(test.lon), run.cfg["lead"]
    K = a.max_lead_steps
    models = {}
    m = run.forecaster(True)
    run.load_guide(m)
    run.load_vnet(m, "vnet")
    models["lie_guided"] = m.eval()
    if (run.dir / "vnet_noguide.pt").exists():
        m2 = run.forecaster(False)
        run.load_vnet(m2, "vnet_noguide")
        models["no_guide"] = m2.eval()
    diff = None
    if (run.dir / "diffusion.pt").exists():
        diff = run.diffusion()
        diff.load_state_dict(torch.load(run.dir / "diffusion.pt"))
        diff.eval()
    # climatology baseline: per-grid-point mean of the training years (physical units)
    clim = run.train.state.mean(0, keepdim=True)  # [1, C, H, W]
    clim_f = run.norm.state(__import__("gpclimate.data.variables", fromlist=["split_state"]).split_state(
        clim, run.spec, run.train.names))

    last = len(test) - 1 - (K - 1) * lead
    cases = list(range(0, last, a.eval_stride))
    log(f"eval: {len(cases)} init times, leads {[lead * 6 * k for k in range(1, K + 1)]} h, models {list(models)}")
    truth = {k: {} for k in range(1, K + 1)}
    preds = {name: {k: {} for k in range(1, K + 1)} for name in list(models) + ["persistence", "climatology"]}
    valid_times = {k: [] for k in range(1, K + 1)}
    t0 = time.time()

    def add(store, k, f):
        phys = merge_state(run.norm.state_inv(f), run.spec, H, W)
        for v in EVAL_VARS:
            store[k].setdefault(v, []).append(phys[v][0].numpy())

    with torch.no_grad():
        for i in cases:
            x0n, _, _, _ = run.prep(test, [i])
            states = {name: x0n for name in models}
            for k in range(1, K + 1):
                j = i + (k - 1) * lead
                cn = run.cond_at(test, [j])
                _, x1n, _, _ = run.prep(test, [j])
                add(truth, k, x1n)
                valid_times[k].append(test.times[j + lead])
                add(preds["persistence"], k, x0n)
                add(preds["climatology"], k, clim_f)
                for name, mdl in models.items():
                    states[name], _ = mdl(states[name], cn, g)
                    add(preds[name], k, states[name])
    t_det = time.time() - t0
    log(f"deterministic rollouts done in {t_det / 60:.1f} min")

    stack = lambda d: {v: np.stack(x, 0) for v, x in d.items()}  # noqa: E731
    results = {"n_cases": len(cases), "lead_hours": {}, "rmse": {}, "finite": {}}
    for k in range(1, K + 1):
        times = np.asarray(valid_times[k]).astype("datetime64[ns]")
        results["lead_hours"][k] = 6 * lead * k
        results["rmse"][k] = {}
        for name, st in preds.items():
            arr = stack(st[k])
            results["finite"].setdefault(name, {})[k] = bool(all(np.isfinite(x).all() for x in arr.values()))
            results["rmse"][k][name] = wb2_scores(arr, stack(truth[k]), test.lat, test.lon, times)["rmse"]

    # ensemble at the first lead: diffusion vs. deterministic + independent Gaussian noise of the residual scale
    if diff is not None:
        ens_cases = cases[: a.n_ens_cases]
        mdl = models["lie_guided"]
        ens, gauss, tr, det = {}, {}, {}, {}
        tt = []
        gen = torch.Generator().manual_seed(a.seed + 3)
        t1 = time.time()
        with torch.no_grad():
            for i in ens_cases:
                x0n, x1n, cn, _ = run.prep(test, [i])
                P = mdl.physics(x0n, cn, g)
                xbar, _ = mdl(x0n, cn, g)
                ctx = {"xbar": xbar, "x": x0n, "cond": cn, "P0u": P["P0u"]}
                members = assemble_ensemble(xbar, diff.sample(ctx, g, a.members, a.sample_steps, seed=int(i)))
                noise = []
                for _ in range(a.members):
                    es = torch.randn(xbar.s.shape, generator=gen) * diff.res_scale_s
                    ev = torch.complex(torch.randn(xbar.v.shape, generator=gen),
                                       torch.randn(xbar.v.shape, generator=gen)) * diff.res_scale_v
                    noise.append(Field(xbar.s + es, xbar.v + ev))
                tt.append(test.times[i + lead])
                for store, f in ((tr, x1n), (det, xbar)):
                    phys = merge_state(run.norm.state_inv(f), run.spec, H, W)
                    for v in EVAL_VARS:
                        store.setdefault(v, []).append(phys[v][0].numpy())
                for store, mem in ((ens, members), (gauss, noise)):
                    for mi, f in enumerate(mem):
                        phys = merge_state(run.norm.state_inv(f), run.spec, H, W)
                        for v in EVAL_VARS:
                            store.setdefault(v, [[] for _ in range(a.members)])[mi].append(phys[v][0].numpy())
        times = np.asarray(tt).astype("datetime64[ns]")
        trs = stack(tr)
        out = {"n_cases": len(ens_cases), "members": a.members, "sample_steps": a.sample_steps,
               "seconds": time.time() - t1, "deterministic_rmse": wb2_scores(stack(det), trs, test.lat, test.lon, times)["rmse"]}
        for label, store in (("diffusion", ens), ("gaussian", gauss)):
            sc = wb2_scores(stack(det), trs, test.lat, test.lon, times,
                            ensemble={v: np.stack([np.stack(mm, 0) for mm in x], 0) for v, x in store.items()})
            out[label] = {"crps": sc["crps"], "ens_mean_rmse": sc["ens_mean_rmse"], "spread": sc["ens_spread"]}
        results["ensemble_lead1"] = out
        log(f"ensemble eval done in {(time.time() - t1) / 60:.1f} min")

    results["seconds_deterministic"] = t_det
    results["params"] = {n: n_params(m.vnet) for n, m in models.items()}
    results["params"]["guide"] = n_params(models["lie_guided"].guide)
    if diff is not None:
        results["params"]["diffusion"] = n_params(diff.denoiser)
    results["config"] = run.cfg
    results["args"] = vars(a)
    with open(run.dir / "eval.json", "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    write_markdown(run, results)
    log(f"eval written to {run.dir / 'eval.json'}")


def write_markdown(run: Run, r: dict):
    lines = [f"# 평가 결과: {run.dir.name}", "",
             f"테스트 기간 {len(run.test.times)} 시점 중 {r['n_cases']}개 초기 시각 (stride {run.args.eval_stride} × 6 h). "
             f"지표: WeatherBench2 면적가중 RMSE (물리 단위)."]
    names = list(next(iter(r["rmse"].values())).keys())
    for k, lh in r["lead_hours"].items():
        lines += ["", f"## Lead {lh} h", "", "| 변수 | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for v in EVAL_VARS:
            lines.append(f"| {v} | " + " | ".join(f"{r['rmse'][k][n][v]:.4g}" for n in names) + " |")
    if "ensemble_lead1" in r:
        e = r["ensemble_lead1"]
        lines += ["", f"## 앙상블 (lead {r['lead_hours'][1]} h, {e['n_cases']} 사례, {e['members']} 멤버, "
                      f"{e['sample_steps']} 샘플링 step)", "",
                  "| 변수 | 결정론 RMSE | diffusion CRPS | diffusion 앙상블평균 RMSE | diffusion spread | "
                  "Gaussian CRPS | Gaussian 앙상블평균 RMSE | Gaussian spread |", "|---|---|---|---|---|---|---|---|"]
        for v in EVAL_VARS:
            d, gs = e["diffusion"], e["gaussian"]
            lines.append(f"| {v} | {e['deterministic_rmse'][v]:.4g} | {d['crps'][v]:.4g} | {d['ens_mean_rmse'][v]:.4g} | "
                         f"{d['spread'][v]:.4g} | {gs['crps'][v]:.4g} | {gs['ens_mean_rmse'][v]:.4g} | {gs['spread'][v]:.4g} |")
    (run.dir / "eval.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "runs/era5_64x32"))
    ap.add_argument("--data", default=str(ROOT / "data/era5_wb2_64x32_2016_2019.npz"))
    ap.add_argument("--stage", required=True, choices=["a", "b", "b_noguide", "c", "eval"])
    ap.add_argument("--mesh", type=int, nargs=3, default=[4, 3, 2])
    ap.add_argument("--lead", type=int, default=4)
    ap.add_argument("--cs", type=int, default=48)
    ap.add_argument("--cv", type=int, default=12)
    ap.add_argument("--guide-cs", type=int, default=32)
    ap.add_argument("--guide-cv", type=int, default=8)
    ap.add_argument("--depth", type=int, nargs=5, default=[1, 1, 1, 1, 1])
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ds", type=int, default=8)
    ap.add_argument("--dv", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--n-basis", type=int, default=8)
    ap.add_argument("--diff-cs", type=int, default=32)
    ap.add_argument("--diff-cv", type=int, default=8)
    ap.add_argument("--diff-depth", type=int, default=2)
    ap.add_argument("--physics-ops", type=int, default=1, choices=[0, 1],
                    help="guide: fixed equivariant physics operators as mesh-0 inputs (stored in config.json)")
    ap.add_argument("--tend-head", type=int, default=1, choices=[0, 1],
                    help="guide: Stage-A one-step (6 h) tendency head (stored in config.json)")
    ap.add_argument("--tend-weight", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--sigma-max", type=float, default=0.5)
    ap.add_argument("--aux-weight", type=float, default=0.5)
    ap.add_argument("--phys-weight", type=float, default=0.1)
    ap.add_argument("--eval-stride", type=int, default=16, help="init-time stride in 6 h steps")
    ap.add_argument("--max-lead-steps", type=int, default=5)
    ap.add_argument("--n-ens-cases", type=int, default=24)
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--sample-steps", type=int, default=20)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    run = Run(args)
    log(f"run {run.dir}: pairs train {len(run.train)} calib {len(run.calib)} test {len(run.test)}; "
        f"graph {run.graphs.summary()['mesh_nodes']}")
    {"a": stage_a, "b": lambda r: stage_b(r, True), "b_noguide": lambda r: stage_b(r, False), "c": stage_c,
     "eval": stage_eval}[args.stage](run)


if __name__ == "__main__":
    main()
