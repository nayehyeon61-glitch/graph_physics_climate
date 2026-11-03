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

from gpclimate.data.era5_wb2 import SPECS, ERA5Pairs, Normalizer, fit_normalizer, fit_tendency_scale  # noqa: E402
from gpclimate.data.variables import merge_state  # noqa: E402
from gpclimate.geometry.graphs import SphereGraphs  # noqa: E402
from gpclimate.models.diffusion import (DiffusionConfig, FeasibilityProjection, GraphResidualDiffusion,  # noqa: E402
                                        ResidualDenoiser)
from gpclimate.models.dynamics_vnet import LieGuidedForecaster, VNetConfig  # noqa: E402
from gpclimate.models.lie_physical_gnn import GuideConfig  # noqa: E402
from gpclimate.nn.primitive2d import DEFAULT_TERMS, resolved_fraction  # noqa: E402
from gpclimate.nn.so2 import Field  # noqa: E402
from gpclimate.training.losses import (field_mse, global_tendency_consistency, perturb, random_gauges,  # noqa: E402
                                       sample_sigma)
from gpclimate.training.metrics import wb2_scores  # noqa: E402



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
            self.cfg = {k: getattr(args, k) for k in ("data", "spec", "mesh", "lead", "cs", "cv", "depth", "heads", "ds", "dv",
                                                      "hidden", "n_basis", "guide_cs", "guide_cv", "diff_cs",
                                                      "diff_cv", "diff_depth", "physics_ops", "tend_head", "fusion",
                                                      "pigt_layers", "pigt_ops", "closure_tend", "resolved_terms",
                                                      "resolved_coef", "integrator", "ode_steps", "ode_solver",
                                                      "ode_physics")}
            json.dump(self.cfg, open(cfg_path, "w"), indent=2)
        c = self.cfg
        # stage10: the state feature matrix is a named spec stored in config.json (older runs: base11)
        self.spec = SPECS[c.get("spec", "base11")]
        self.eval_vars = self.spec.state_channels()  # every state channel is scored (RMSE per variable)
        log(f"feature matrix {c.get('spec', 'base11')}: {self.spec.n_s} scalar + {self.spec.n_v} vector-pair state "
            f"channels ({len(self.eval_vars)} variables), {self.spec.n_cond_s} scalar + {self.spec.n_cond_v} vector conditions")
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
                                     tend_head=bool(c.get("tend_head", False)),
                                     closure_tend=bool(c.get("closure_tend", False)),
                                     resolved_terms=tuple(c.get("resolved_terms") or DEFAULT_TERMS),
                                     resolved_coef=c.get("resolved_coef"))
        self.vnet_kwargs = dict(cs=c["cs"], cv=c["cv"], depth=tuple(c["depth"]), heads=c["heads"], ds=c["ds"],
                                dv=c["dv"], n_basis=c["n_basis"], hidden=c["hidden"],
                                fusion=c.get("fusion", "gated"), pigt_layers=int(c.get("pigt_layers", 2)),
                                pigt_ops=bool(c.get("pigt_ops", True)), integrator=c.get("integrator", "delta"),
                                ode_steps=int(c.get("ode_steps", 4)), ode_solver=c.get("ode_solver", "euler"),
                                ode_physics=bool(c.get("ode_physics", False)))
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
        m.lead_steps = int(self.cfg["lead"])
        if self.norm.tend_s is not None:  # stage S8 physics drift scale (6 h tendency)
            m.tend_scale_s.copy_(self.norm.tend_s)
            m.tend_scale_v.copy_(self.norm.tend_v)
        if m.guide is not None and m.guide.phys_ops is not None:
            m.guide.phys_ops.v_scale.copy_(self.norm.v_scale)  # m/s per normalised unit (vorticity / 2 Omega)
        if getattr(m.vnet, "phys_ops", None) is not None:
            m.vnet.phys_ops.v_scale.copy_(self.norm.v_scale)
        if m.guide is not None and m.guide.resolved is not None:
            m.guide.resolved.set_scales(self.norm.s_mean, self.norm.s_std, self.norm.v_scale, self.norm.tend_s,
                                        self.norm.tend_v)
        return m

    def load_guide(self, m: LieGuidedForecaster):
        m.guide.load_state_dict(torch.load(self.dir / "guide.pt"))

    def load_vnet(self, m: LieGuidedForecaster, name: str):
        m.vnet.load_state_dict(torch.load(self.dir / f"{name}.pt"))

    def diffusion(self, constraints: bool = False, water: bool = True) -> GraphResidualDiffusion:
        """Residual diffusion; ``constraints`` (stage S5c) attaches the feasibility projection P_C."""
        proj = FeasibilityProjection(self.spec, self.norm.s_mean, self.norm.s_std, water) if constraints else None
        return GraphResidualDiffusion(ResidualDenoiser(self.spec, self.diff_cfg,
                                                       guide_dims=(self.guide_cfg.cs, self.guide_cfg.cv)), 1000, proj)

    def load_diffusion(self) -> GraphResidualDiffusion:
        sd = torch.load(self.dir / "diffusion.pt")
        diff = self.diffusion(constraints="project.s_mean" in sd, water=bool(sd.get("project.water", True)))
        diff.load_state_dict(sd)
        return diff

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
    lead = run.cfg["lead"]
    # optional structured perturbation: with probability perturb_prob the input is the ``lead``-step forecast of a
    # guide-free V-Net (valid at t), so the guide also learns to read its own forecast states (rollout robustness)
    fm = None
    if getattr(a, "perturb_model", None):
        fm = run.forecaster(False)
        fm.vnet.load_state_dict(torch.load(a.perturb_model))
        fm.eval()
        log(f"stage A: forecast-state perturbation from {a.perturb_model} with p={a.perturb_prob}")
    w_m = torch.ones(g.n_mesh[0])  # icosahedral nodes are nearly uniform in area
    log(f"stage A: guide params {n_params(guide)}, steps {a.steps}, batch {a.batch}"
        + (f", closure head with resolved terms {guide.cfg.resolved_terms}" if guide.closure_out is not None else ""))
    n_fc = 0
    for step in range(a.steps):
        for pg in opt.param_groups:
            pg["lr"] = cosine_lr(step, a.steps, a.lr)
        idx = rng.integers(lead if fm is not None else 0, len(run.train), a.batch)
        x0n, _, cn, x0 = run.prep(run.train, idx)
        xin = x0n
        if fm is not None and rng.random() < a.perturb_prob:
            with torch.no_grad():
                xp0n, _, cpn, _ = run.prep(run.train, idx - lead)
                xin, _ = fm(xp0n, cpn, g)  # forecast valid at idx; targets stay the true state
            n_fc += 1
        aux_t = run.train.aux_targets(x0) / run.norm.aux_std
        tg, tm = random_gauges(run.graphs, gen)
        gv = run.graphs.gauge_transformed(tm, tg)
        xg, cg = x0n.gauge(tg), cn.gauge(tg)
        sigma = sample_sigma(a.batch, a.sigma_max, generator=gen)
        feats = guide(perturb(xin.gauge(tg), sigma, gen), cg, sigma, gv)
        rec, aux, tend = guide.heads_all(feats, gv)
        l_rec = field_mse(rec, xg, w)
        l_aux = ((aux - aux_t) ** 2 * w[None, :, None]).mean() / w.mean()
        loss = l_rec + a.aux_weight * l_aux
        l_tend = torch.zeros(())
        if tend is not None:
            tend_t = run.norm.tendency(x0, run.train.next_state(idx)).gauge(tg)
            l_tend = field_mse(tend, tend_t, w)
            loss = loss + a.tend_weight * l_tend
        l_clos = torch.zeros(())
        if guide.closure_out is not None:  # S5a: closure = observed tendency - fixed resolved dynamics, on mesh 0
            tend_m = guide.resolved.pool(run.norm.tendency(x0, run.train.next_state(idx)).gauge(tg), gv)
            clos_t = tend_m - guide.resolved_tendency(xg, gv)
            l_clos = field_mse(guide.closure(feats), clos_t, w_m)
            loss = loss + a.tend_weight * l_clos
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(guide.parameters(), 1.0)
        opt.step()
        if step % 25 == 0:
            run.jlog("a", {"step": step, "l_rec": float(l_rec), "l_aux": float(l_aux), "l_tend": float(l_tend),
                           "l_clos": float(l_clos), "t": time.time() - t0})
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
                v_clos, v_full, rfrac = 0.0, 0.0, None
                if guide.closure_out is not None:
                    tend_m = guide.resolved.pool(run.norm.tendency(x0, run.calib.next_state(val_idx)), g)
                    F = guide.resolved_tendency(x0n, g)
                    S = guide.closure(feats)
                    v_clos = float(field_mse(S, tend_m - F, w_m))  # == error of the full tendency F + S
                    v_full = float(field_mse(F, tend_m, w_m))  # resolved dynamics alone (no closure)
                    rfrac = resolved_fraction(tend_m, F, w_m)
            guide.train()
            score = v_rec + a.aux_weight * v_aux + a.tend_weight * (v_tend + v_clos)
            if score < best:
                best = score
                torch.save(guide.state_dict(), run.dir / "guide.pt")
            run.jlog("a", {"step": step, "val_rec": v_rec, "val_aux": v_aux, "val_tend": v_tend, "val_clos": v_clos,
                           "val_resolved_only": v_full, "resolved_fraction": rfrac, "best": best,
                           "t": time.time() - t0})
            log(f"A {step:5d}  L_rec {float(l_rec):.4f} L_aux {float(l_aux):.4f} L_tend {float(l_tend):.4f}"
                f" L_clos {float(l_clos):.4f} | val rec {v_rec:.4f} aux {v_aux:.4f} tend {v_tend:.4f} clos {v_clos:.4f}"
                f" (resolved only {v_full:.4f}) | {(time.time() - t0) / (step + 1):.2f} s/step")
    log(f"stage A done in {(time.time() - t0) / 60:.1f} min, best val score {best:.4f}"
        + (f", forecast-perturbed batches {n_fc}" if fm is not None else ""))


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
    K, k_start, lead = int(getattr(a, "rollout_steps", 1)), int(getattr(a, "rollout_start", 0)), run.cfg["lead"]
    lam = float(getattr(a, "rollout_weight", 1.0))
    val_idx2 = [i for i in val_idx if i < len(run.calib) - lead]  # 2-step validation needs X_{t+2 lead}

    def scaled(f: Field) -> Field:
        return Field(f.s / model.delta_scale_s, f.v / model.delta_scale_v)

    lam_pde = float(getattr(a, "pde_weight", 0.0))
    use_pde = lam_pde > 0 and use_guide and model.guide.closure_out is not None
    w_m = torch.ones(g.n_mesh[0])

    def pde_residual(x0n: Field, fc: Field, cn: Field) -> torch.Tensor:
        """S5b: mean 6 h tendency of the lead-time forecast vs the guide's physics tendency F + S at the midpoint.

        (X_hat - X_t) / lead  (normalised, per data step)  should equal  F(X_bar) + S_theta(X_bar),  X_bar the
        midpoint state (trapezoidal rule); gradients reach X_hat through F, S_theta and the pooling.
        """
        guide = model.guide
        tend_hat = Field((fc.s - x0n.s) / (lead * run.norm.tend_s), (fc.v - x0n.v) / (lead * run.norm.tend_v))
        xbar = Field(0.5 * (x0n.s + fc.s), 0.5 * (x0n.v + fc.v))
        feats = guide(xbar, cn, torch.zeros(x0n.s.shape[0]), g)
        return field_mse(guide.resolved.pool(tend_hat, g), guide.tendency_mesh(feats, xbar, g), w_m)

    log(f"stage B[{name}]: vnet params {n_params(model.vnet)}, steps {a.steps}, batch {a.batch}, "
        f"rollout steps {K} from step {k_start} (weight {lam})" + (f", PDE residual weight {lam_pde}" if use_pde else ""))
    if model.vnet.cfg.integrator == "ode":  # stage S8
        vc = model.vnet.cfg
        log(f"stage B[{name}]: graph neural ODE dynamics, {vc.ode_steps} sub-steps of {6 * lead // vc.ode_steps} h, "
            f"solver {vc.ode_solver}, physics drift {'on' if vc.ode_physics else 'off'}")
    for step in range(a.steps):
        for pg in opt.param_groups:
            pg["lr"] = cosine_lr(step, a.steps, a.lr)
        k_now = K if step >= k_start else 1
        idx = rng.integers(0, len(run.train) - (k_now - 1) * lead, a.batch)
        states, conds = run.train.sequence(idx, k_now)
        xs = [run.norm.state(x) for x in states]
        cs = [run.norm.cond(c) for c in conds]
        f, d = model(xs[0], cs[0], g)
        l_f = field_mse(scaled(f), scaled(xs[1]), w)  # == delta MSE of the one-step target
        l_phys = global_tendency_consistency(d.s * model.delta_scale_s, xs[1].s - xs[0].s, w)
        l_pde = pde_residual(xs[0], f, cs[0]) if use_pde else torch.zeros(())
        l_roll = torch.zeros(())
        for k in range(2, k_now + 1):  # autoregressive steps; physics features recomputed from the forecast
            f, _ = model(f, cs[k - 1], g)
            l_roll = l_roll + field_mse(scaled(f), scaled(xs[k]), w)
        loss = l_f + a.phys_weight * l_phys + lam * l_roll + lam_pde * l_pde
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 25 == 0:
            run.jlog(name, {"step": step, "l_f": float(l_f), "l_phys": float(l_phys), "l_roll": float(l_roll),
                            "l_pde": float(l_pde), "k": k_now, "t": time.time() - t0})
        if step % a.val_every == 0 or step == a.steps - 1:
            model.eval()
            with torch.no_grad():
                tot, tot2 = 0.0, 0.0
                for i in range(0, len(val_idx), 8):
                    x0n, x1n, cn, _ = run.prep(run.calib, val_idx[i:i + 8])
                    fc, _ = model(x0n, cn, g)
                    tot += float(field_mse(fc, x1n, w)) * len(val_idx[i:i + 8])
                for i in range(0, len(val_idx2), 8):
                    states, conds = run.calib.sequence(val_idx2[i:i + 8], 2)
                    fc = run.norm.state(states[0])
                    for k in range(2):
                        fc, _ = model(fc, run.norm.cond(conds[k]), g)
                    tot2 += float(field_mse(fc, run.norm.state(states[2]), w)) * len(val_idx2[i:i + 8])
                v_mse, v_mse2 = tot / len(val_idx), tot2 / len(val_idx2)
                x0n, x1n, cn, _ = run.prep(run.calib, val_idx[:8])
                pers = float(field_mse(x0n, x1n, w))
            model.train()
            if use_guide:
                model.guide.eval()
            score = v_mse + (v_mse2 if K >= 2 else 0.0)  # with rollout training select on 1-step + 2-step error
            if score < best:
                best = score
                torch.save(model.vnet.state_dict(), run.dir / f"{name}.pt")
            run.jlog(name, {"step": step, "val_mse": v_mse, "val_mse2": v_mse2, "val_persistence_mse": pers,
                            "best": best, "attention_mass": model.vnet.attention_mass(), "t": time.time() - t0})
            log(f"B[{name}] {step:5d}  L_f {float(l_f):.4f} L_roll {float(l_roll):.4f} L_phys {float(l_phys):.5f}"
                + (f" L_pde {float(l_pde):.4f}" if use_pde else "")
                + f" | val MSE {v_mse:.4f} 2-step {v_mse2:.4f} (persistence {pers:.4f})"
                f" | {(time.time() - t0) / (step + 1):.2f} s/step")
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
    diff = run.diffusion(constraints=bool(a.noise_constraints), water=bool(a.noise_water))
    rng = np.random.default_rng(a.seed + 2)
    gen = torch.Generator().manual_seed(a.seed + 4)
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
    if diff.project is not None:  # stage S5c
        log(f"stage C: feasibility projection (q700 >= 0, mass, water={bool(a.noise_water)}), "
            f"feas weight {a.noise_feas_weight}, mean constraint weight {a.noise_mean_weight} every "
            f"{a.noise_mean_every} steps ({a.noise_mean_members} members x {a.noise_mean_steps} sampling steps)")

    def context(x0n, cn):
        with torch.no_grad():
            P = model.physics(x0n, cn, g)
            xbar, _ = model(x0n, cn, g)
        return xbar, {"xbar": xbar, "x": x0n, "cond": cn, "P0u": P["P0u"]}

    opt = torch.optim.Adam(diff.denoiser.parameters(), lr=a.lr)
    t0, hist, hist_f, hist_m = time.time(), [], [], []
    for step in range(a.steps):
        for pg in opt.param_groups:
            pg["lr"] = cosine_lr(step, a.steps, a.lr)
        idx = rng.integers(0, len(run.calib), a.batch)
        x0n, x1n, cn, _ = run.prep(run.calib, idx)
        xbar, ctx = context(x0n, cn)
        ls = diff.losses(diff.to_residual(x1n, xbar), ctx, g, w)
        loss = ls["dsm"]
        if diff.project is not None:  # stage S5c: L_DSM + lambda_feas L_feas + lambda_mean L_mean
            loss = loss + a.noise_feas_weight * ls["feas"]
            hist_f.append(float(ls["feas"]))
            if a.noise_mean_weight > 0 and step % a.noise_mean_every == 0:
                l_mean = diff.mean_loss(ctx, g, w, a.noise_mean_members, a.noise_mean_steps, gen)
                loss = loss + a.noise_mean_weight * l_mean
                hist_m.append(float(l_mean))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(diff.parameters(), 1.0)
        opt.step()
        hist.append(float(ls["dsm"]))
        if step % 25 == 0:
            rec = {"step": step, "l_eps": float(np.mean(hist[-25:])), "t": time.time() - t0}
            if hist_f:
                rec["l_feas"] = float(np.mean(hist_f[-25:]))
            if hist_m:
                rec["l_mean"] = float(np.mean(hist_m[-5:]))
            run.jlog("c", rec)
        if step % a.val_every == 0 or step == a.steps - 1:
            extra = (f" L_feas {np.mean(hist_f[-25:]):.4f}" if hist_f else "") + (
                f" L_mean {np.mean(hist_m[-5:]):+.4f}" if hist_m else "")
            log(f"C {step:5d}  L_eps(avg25) {np.mean(hist[-25:]):.4f}{extra} | {(time.time() - t0) / (step + 1):.2f} s/step")
    torch.save(diff.state_dict(), run.dir / "diffusion.pt")
    log(f"stage C done in {(time.time() - t0) / 60:.1f} min")


# ---------------------------------------------------------------------------- evaluation
class PhysicsOnlyForecaster(torch.nn.Module):
    """Stage S8b diagnostic: integrate the frozen guide's tendency alone (same sub-steps, V-Net switched off)."""

    def __init__(self, base: LieGuidedForecaster):
        super().__init__()
        self.base = base
        self.guide, self.vnet = base.guide, base.vnet

    def forward(self, state: Field, cond: Field, g):
        f = self.base.physics_only(state, cond, g)
        return f, Field((f.s - state.s) / self.base.delta_scale_s, (f.v - state.v) / self.base.delta_scale_v)


def stage_eval(run: Run):
    a, g, test = run.args, run.g, run.test
    H, W, lead = len(test.lat), len(test.lon), run.cfg["lead"]
    K = a.max_lead_steps
    models = {}
    m = run.forecaster(True)
    run.load_guide(m)
    run.load_vnet(m, "vnet")
    models["lie_guided"] = m.eval()
    if m.vnet.cfg.integrator == "ode" and m.vnet.cfg.ode_physics:  # stage S8b diagnostic: drift alone, V-Net off
        models["physics_only"] = PhysicsOnlyForecaster(m).eval()
    if (run.dir / "vnet_noguide.pt").exists():
        m2 = run.forecaster(False)
        run.load_vnet(m2, "vnet_noguide")
        models["no_guide"] = m2.eval()
    diff = None
    if (run.dir / "diffusion.pt").exists():
        diff = run.load_diffusion().eval()
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
        for v in run.eval_vars:
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
        ens, ens_raw, gauss, tr, det = {}, {}, {}, {}, {}
        tt = []
        gen = torch.Generator().manual_seed(a.seed + 3)
        t1 = time.time()
        # stage S5c diagnostics: constraint violations of raw / projected members and the ensemble-mean bias
        viol = {k: {"neg_q_fraction": [], "mass_drift_pa": [], "water_drift_rel": []} for k in ("raw", "projected", "gaussian")}
        viol_proj = FeasibilityProjection(run.spec, run.norm.s_mean, run.norm.s_std, True)
        bias = {k: [] for k in ("diffusion", "gaussian")}
        with torch.no_grad():
            for i in ens_cases:
                x0n, x1n, cn, _ = run.prep(test, [i])
                P = mdl.physics(x0n, cn, g)
                xbar, _ = mdl(x0n, cn, g)
                ctx = {"xbar": xbar, "x": x0n, "cond": cn, "P0u": P["P0u"]}
                res = diff.sample(ctx, g, a.members, a.sample_steps, seed=int(i))
                raw = diff.assemble(xbar, res, run.w, project=False)
                members = diff.assemble(xbar, res, run.w, project=True) if diff.project is not None else raw
                noise = []
                for _ in range(a.members):
                    es = torch.randn(xbar.s.shape, generator=gen) * diff.res_scale_s
                    ev = torch.complex(torch.randn(xbar.v.shape, generator=gen),
                                       torch.randn(xbar.v.shape, generator=gen)) * diff.res_scale_v
                    noise.append(Field(xbar.s + es, xbar.v + ev))
                tt.append(test.times[i + lead])
                for store, f in ((tr, x1n), (det, xbar)):
                    phys = merge_state(run.norm.state_inv(f), run.spec, H, W)
                    for v in run.eval_vars:
                        store.setdefault(v, []).append(phys[v][0].numpy())
                stores = [(ens, members), (gauss, noise)] + ([(ens_raw, raw)] if diff.project is not None else [])
                for store, mem in stores:
                    for mi, f in enumerate(mem):
                        phys = merge_state(run.norm.state_inv(f), run.spec, H, W)
                        for v in run.eval_vars:
                            store.setdefault(v, [[] for _ in range(a.members)])[mi].append(phys[v][0].numpy())
                for key, mem in (("raw", raw), ("projected", members), ("gaussian", noise)):
                    for f in mem:
                        for kk, val in viol_proj.violations(f, xbar, run.w).items():
                            viol[key][kk].append(float(val.abs().mean()) if kk != "neg_q_fraction" else float(val.mean()))
                for key, mem in (("diffusion", members), ("gaussian", noise)):
                    mean_s = torch.stack([f.s for f in mem]).mean(0)
                    mean_v = torch.stack([f.v for f in mem]).mean(0)
                    # physical-unit area-weighted RMS of (ensemble mean - deterministic) per variable
                    d = Field((mean_s - xbar.s) * run.norm.s_std, (mean_v - xbar.v) * run.norm.v_scale)
                    phys = merge_state(d, run.spec, H, W)
                    bias[key].append({v: float(np.sqrt(np.average((phys[v][0].numpy() ** 2).mean(-1),
                                                                   weights=np.cos(np.deg2rad(test.lat)))))
                                      for v in run.eval_vars})
        times = np.asarray(tt).astype("datetime64[ns]")
        trs = stack(tr)
        out = {"n_cases": len(ens_cases), "members": a.members, "sample_steps": a.sample_steps,
               "seconds": time.time() - t1, "deterministic_rmse": wb2_scores(stack(det), trs, test.lat, test.lon, times)["rmse"]}
        for label, store in (("diffusion", ens), ("gaussian", gauss)) + ((("diffusion_raw", ens_raw),) if ens_raw else ()):
            sc = wb2_scores(stack(det), trs, test.lat, test.lon, times,
                            ensemble={v: np.stack([np.stack(mm, 0) for mm in x], 0) for v, x in store.items()})
            out[label] = {"crps": sc["crps"], "ens_mean_rmse": sc["ens_mean_rmse"], "spread": sc["ens_spread"]}
        out["constraints"] = {"projection": diff.project is not None,
                              **{k: {kk: float(np.mean(vv)) for kk, vv in d.items()} for k, d in viol.items()}}
        out["ens_mean_minus_det_rms"] = {k: {v: float(np.mean([b[v] for b in bl])) for v in run.eval_vars}
                                         for k, bl in bias.items()}
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
        for v in run.eval_vars:
            lines.append(f"| {v} | " + " | ".join(f"{r['rmse'][k][n][v]:.4g}" for n in names) + " |")
    if "ensemble_lead1" in r:
        e = r["ensemble_lead1"]
        lines += ["", f"## 앙상블 (lead {r['lead_hours'][1]} h, {e['n_cases']} 사례, {e['members']} 멤버, "
                      f"{e['sample_steps']} 샘플링 step)", "",
                  "| 변수 | 결정론 RMSE | diffusion CRPS | diffusion 앙상블평균 RMSE | diffusion spread | "
                  "Gaussian CRPS | Gaussian 앙상블평균 RMSE | Gaussian spread |", "|---|---|---|---|---|---|---|---|"]
        for v in run.eval_vars:
            d, gs = e["diffusion"], e["gaussian"]
            lines.append(f"| {v} | {e['deterministic_rmse'][v]:.4g} | {d['crps'][v]:.4g} | {d['ens_mean_rmse'][v]:.4g} | "
                         f"{d['spread'][v]:.4g} | {gs['crps'][v]:.4g} | {gs['ens_mean_rmse'][v]:.4g} | {gs['spread'][v]:.4g} |")
        if "constraints" in e:
            c = e["constraints"]
            lines += ["", f"### 허용조건 진단 (사영 {'켜짐' if c['projection'] else '꺼짐'})", "",
                      "| 멤버 | q700<0 면적비 | 질량 drift |⟨msl−μ⟩| (Pa) | 수분 drift |W/W_μ−1| |", "|---|---|---|---|"]
            for k in ("raw", "projected", "gaussian"):
                lines.append(f"| {k} | {c[k]['neg_q_fraction']:.4f} | {c[k]['mass_drift_pa']:.3g} | {c[k]['water_drift_rel']:.3g} |")
            if "diffusion_raw" in e:
                dr = e["diffusion_raw"]
                lines += ["", "| 변수 | 사영 전 CRPS | 사영 후 CRPS | 사영 전 앙상블평균 RMSE | 사영 후 앙상블평균 RMSE |", "|---|---|---|---|---|"]
                for v in run.eval_vars:
                    lines.append(f"| {v} | {dr['crps'][v]:.4g} | {d['crps'][v]:.4g} | {dr['ens_mean_rmse'][v]:.4g} | {d['ens_mean_rmse'][v]:.4g} |")
            b = e["ens_mean_minus_det_rms"]
            lines += ["", "| 변수 | 앙상블평균−결정론 RMS (diffusion) | 〃 (Gaussian) |", "|---|---|---|"]
            for v in run.eval_vars:
                lines.append(f"| {v} | {b['diffusion'][v]:.4g} | {b['gaussian'][v]:.4g} |")
    (run.dir / "eval.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "runs/era5_64x32"))
    ap.add_argument("--data", default=str(ROOT / "data/era5_wb2_64x32_2016_2019.npz"))
    ap.add_argument("--spec", default="base11", choices=sorted(SPECS),
                    help="state feature matrix: base11 (5 scalars + 3 wind pairs) or fcn21 (FourCastNet-20 channels "
                         "+ q700 = 13 scalars + 4 wind pairs; stage10; stored in config.json at run creation)")
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
    ap.add_argument("--closure-tend", type=int, default=0, choices=[0, 1],
                    help="guide (stage S5a): tendency = fixed resolved dynamics + learned closure on mesh 0 "
                         "(stored in config.json)")
    ap.add_argument("--resolved-terms", default=",".join(DEFAULT_TERMS),
                    help="comma-separated resolved terms kept in F (stage S5a; stored in config.json)")
    ap.add_argument("--resolved-coef", default=None,
                    help="JSON written by scripts/resolved_fraction.py --fit with the fixed calibration coefficients "
                         "of the resolved terms (stage S5a; stored in config.json)")
    ap.add_argument("--pde-weight", type=float, default=0.0,
                    help="Stage B (stage S5b): weight of the PDE-residual loss between the forecast tendency and "
                         "the guide's F + S_theta at the midpoint state (needs a closure-head guide)")
    ap.add_argument("--noise-constraints", type=int, default=0, choices=[0, 1],
                    help="Stage C (stage S5c): feasibility projection P_C on generated states + feasibility penalty "
                         "+ ensemble-mean constraint (stored in diffusion.pt)")
    ap.add_argument("--noise-water", type=int, default=1, choices=[0, 1], help="S5c: also conserve column water")
    ap.add_argument("--noise-feas-weight", type=float, default=1.0)
    ap.add_argument("--noise-mean-weight", type=float, default=1.0)
    ap.add_argument("--noise-mean-every", type=int, default=5, help="S5c: steps between mean-constraint updates")
    ap.add_argument("--noise-mean-members", type=int, default=2)
    ap.add_argument("--noise-mean-steps", type=int, default=4, help="S5c: sampling steps inside the mean constraint")
    ap.add_argument("--fusion", default="gated", choices=["gated", "pigt"],
                    help="V-Net physics fusion: gated cross-attention or the physics-integrated graph transformer")
    ap.add_argument("--pigt-layers", type=int, default=2)
    ap.add_argument("--pigt-ops", type=int, default=1, choices=[0, 1])
    ap.add_argument("--integrator", default="delta", choices=["delta", "ode"],
                    help="dynamics layer (stage S8): one-shot lead-time change or a graph neural ODE integrated with "
                         "--ode-steps fixed sub-steps (stored in config.json)")
    ap.add_argument("--ode-steps", type=int, default=4)
    ap.add_argument("--ode-solver", default="euler", choices=["euler", "heun"])
    ap.add_argument("--ode-physics", type=int, default=0, choices=[0, 1],
                    help="stage S8: add the frozen guide's 6 h tendency as the known drift of the ODE")
    ap.add_argument("--perturb-model", default=None,
                    help="Stage A: vnet state dict of a guide-free forecaster whose lead-time forecast replaces the "
                         "input with probability --perturb-prob (structured, forecast-like perturbations)")
    ap.add_argument("--perturb-prob", type=float, default=0.3)
    ap.add_argument("--rollout-steps", type=int, default=1, help="Stage B: autoregressive steps in the loss")
    ap.add_argument("--rollout-start", type=int, default=0, help="Stage B: first step that uses the rollout loss")
    ap.add_argument("--rollout-weight", type=float, default=1.0)
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
    args.resolved_terms = [t for t in str(args.resolved_terms).split(',') if t]
    if args.resolved_coef and not isinstance(args.resolved_coef, dict):
        args.resolved_coef = json.load(open(args.resolved_coef))["coef"]
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    run = Run(args)
    log(f"run {run.dir}: pairs train {len(run.train)} calib {len(run.calib)} test {len(run.test)}; "
        f"graph {run.graphs.summary()['mesh_nodes']}")
    {"a": stage_a, "b": lambda r: stage_b(r, True), "b_noguide": lambda r: stage_b(r, False), "c": stage_c,
     "eval": stage_eval}[args.stage](run)


if __name__ == "__main__":
    main()
