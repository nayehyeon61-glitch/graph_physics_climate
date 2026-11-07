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
import copy
import json
import math
import sys
import time
from typing import Optional
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpclimate.data.era5_wb2 import (COLUMN_SETS, SPECS, ColumnInputs, ERA5Pairs, Normalizer,  # noqa: E402
                                     fit_normalizer, fit_tendency_scale)
from gpclimate.data.variables import merge_state  # noqa: E402
from gpclimate.geometry.graphs import SphereGraphs  # noqa: E402
from gpclimate.models.diffusion import (DiffusionConfig, FeasibilityProjection, GraphResidualDiffusion,  # noqa: E402
                                        ResidualDenoiser)
from gpclimate.models.dynamics_vnet import LieGuidedForecaster, VNetConfig  # noqa: E402
from gpclimate.models.lie_physical_gnn import GuideConfig  # noqa: E402
from gpclimate.nn.primitive2d import DEFAULT_TERMS, resolved_fraction  # noqa: E402
from gpclimate.nn.so2 import Field  # noqa: E402
from gpclimate.training.losses import (field_mse, global_tendency_consistency, perturb, random_gauges,  # noqa: E402
                                       sample_sigma, spectral_energy_loss)
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


def parse_rollout_schedule(text: str):
    """stage12: Stage B rollout curriculum ``"500:2,1000:3"`` -> [(500, 2), (1000, 3)] (from step 500 two autoregressive
    steps in the loss, from step 1000 three).  Empty string = no schedule (``--rollout-steps/--rollout-start`` apply)."""
    out = []
    for item in [t for t in str(text or "").split(",") if t.strip()]:
        step, k = item.split(":")
        out.append((int(step), int(k)))
    out.sort()
    if any(k < 1 for _, k in out):
        raise ValueError(f"rollout schedule needs k >= 1: {text}")
    return out


def focus_weights(spec, focus: str, weight: float):
    """stage10b: per-channel loss weights — 1.0 for the state channels of the ``focus`` spec, ``weight`` for the rest.
    Returns ``None`` (unweighted) when no focus is set."""
    if not focus:
        return None
    fs = SPECS[focus]
    ws = torch.tensor([1.0 if n in fs.scalars else weight for n in spec.scalars])
    wv = torch.tensor([1.0 if pair in fs.vectors else weight for pair in spec.vectors])
    return ws, wv


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
                                                      "ode_physics", "curvature_ops", "loss_focus",
                                                      "loss_focus_weight", "simplicial", "neural_op", "no_hidden",
                                                      "no_layers", "no_modes", "prev_state", "nudge", "nudge_carry",
                                                      "nudge_alpha_init")}
            # stage15a keys (absent from runs created before; tests build bare Namespaces)
            self.cfg.update(column_inputs=getattr(args, "column_inputs", "") or "",
                            column_data=getattr(args, "column_data", "") or "")
            # stage15b keys (intermediate-time auxiliary loss)
            self.cfg.update(inter_times=int(getattr(args, "inter_times", 0) or 0),
                            inter_weight=float(getattr(args, "inter_weight", 1.0)))
            # stage18b key (previous-state input along the ODE path)
            self.cfg.update(prev_path=int(getattr(args, "prev_path", 0) or 0))
            # stage17 keys (dynamic guide: sub-step forcing, learnable guide-drift gain)
            self.cfg.update(ode_path_forcing=int(getattr(args, "ode_path_forcing", 0) or 0),
                            ode_drift_gain=int(getattr(args, "ode_drift_gain", 0) or 0),
                            ode_drift=getattr(args, "ode_drift", "tend") or "tend",
                            flow_head=int(getattr(args, "flow_head", 0) or 0))
            # stage18 keys (graph tendency core)
            self.cfg.update(tendency_core=int(getattr(args, "tendency_core", 0) or 0),
                            core_tokens=int(getattr(args, "core_tokens", 1) if getattr(args, "core_tokens", None) is not None
                                            else 1))
            json.dump(self.cfg, open(cfg_path, "w"), indent=2)
        c = self.cfg
        # stage10: the state feature matrix is a named spec stored in config.json (older runs: base11)
        self.spec = SPECS[c.get("spec", "base11")]
        self.eval_vars = self.spec.state_channels()  # every state channel is scored (RMSE per variable)
        # stage10b: channel-weighted training losses (focus spec channels 1.0, the rest loss_focus_weight);
        # validation / eval stay unweighted so runs remain comparable
        self.loss_w = focus_weights(self.spec, c.get("loss_focus", "") or "", float(c.get("loss_focus_weight", 0.2)))
        if self.loss_w is not None:
            log(f"loss focus {c['loss_focus']}: scalar weights {self.loss_w[0].tolist()}, vector weights {self.loss_w[1].tolist()}")
        log(f"feature matrix {c.get('spec', 'base11')}: {self.spec.n_s} scalar + {self.spec.n_v} vector-pair state "
            f"channels ({len(self.eval_vars)} variables), {self.spec.n_cond_s} scalar + {self.spec.n_cond_v} vector conditions")
        self.train = ERA5Pairs(c["data"], "train", self.spec, c["lead"])
        self.calib = ERA5Pairs(c["data"], "calib", self.spec, c["lead"])
        self.test = ERA5Pairs(c["data"], "test", self.spec, c["lead"])
        if getattr(args, "eval_data", ""):  # external benchmark year (e.g. 2020 for the WeatherBench2 leaderboard)
            self.test = ERA5Pairs(args.eval_data, "test", self.spec, c["lead"])
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
                                     resolved_coef=c.get("resolved_coef"),
                                     curvature_ops=bool(c.get("curvature_ops", False)),
                                     simplicial=int(c.get("simplicial", 0)),
                                     flow_head=bool(c.get("flow_head", False)))
        self.vnet_kwargs = dict(cs=c["cs"], cv=c["cv"], depth=tuple(c["depth"]), heads=c["heads"], ds=c["ds"],
                                dv=c["dv"], n_basis=c["n_basis"], hidden=c["hidden"],
                                fusion=c.get("fusion", "gated"), pigt_layers=int(c.get("pigt_layers", 2)),
                                pigt_ops=bool(c.get("pigt_ops", True)), integrator=c.get("integrator", "delta"),
                                ode_steps=int(c.get("ode_steps", 4)), ode_solver=c.get("ode_solver", "euler"),
                                ode_physics=bool(c.get("ode_physics", False)),
                                neural_op=c.get("neural_op", "") or "", no_hidden=int(c.get("no_hidden", 16)),
                                no_layers=int(c.get("no_layers", 2)), no_modes=tuple(c.get("no_modes") or (16, 32)),
                                prev_state=bool(c.get("prev_state", False)), nudge=bool(c.get("nudge", False)),
                                nudge_carry=bool(c.get("nudge_carry", True) if c.get("nudge_carry") is not None else True),
                                nudge_alpha_init=float(c.get("nudge_alpha_init") or 0.12),
                                ode_path_forcing=bool(c.get("ode_path_forcing", False)),
                                ode_drift_gain=bool(c.get("ode_drift_gain", False)),
                                ode_drift=c.get("ode_drift", "tend") or "tend",
                                tendency_core=bool(c.get("tendency_core", False)),
                                core_tokens=bool(c.get("core_tokens", True) if c.get("core_tokens") is not None else True),
                                prev_path=bool(c.get("prev_path", False)))
        if self.vnet_kwargs["prev_path"] and not (self.vnet_kwargs["prev_state"] and self.vnet_kwargs["integrator"] == "ode"):
            raise ValueError("prev_path needs the previous-state input (prev_state) and the ODE integrator")
        if self.vnet_kwargs["ode_drift_gain"] and not self.vnet_kwargs["ode_physics"]:
            raise ValueError("ode_drift_gain needs ode_physics (the gain multiplies the guide's tendency drift)")
        # stage15a: column channels as V-Net inputs at the initial time only (normalised with train-split statistics,
        # cached in column_norm.json so an external benchmark year uses the same scaling)
        self.use_col = bool(c.get("column_inputs"))
        if self.use_col:
            self._init_columns(args)
            self.vnet_kwargs.update(col_s=len(COLUMN_SETS[c["column_inputs"]][0]),
                                    col_v=len(COLUMN_SETS[c["column_inputs"]][1]))
        # stage15b: auxiliary head for the intermediate times X_{t + j 6 h}, j = 1..inter_times (training loss only)
        self.inter_times = int(c.get("inter_times", 0) or 0)
        self.inter_weight = float(c.get("inter_weight", 1.0) if c.get("inter_weight") is not None else 1.0)
        if self.inter_times:
            if self.inter_times >= int(c["lead"]):
                raise ValueError(f"inter_times must be < lead ({c['lead']} data steps)")
            if c.get("integrator", "delta") == "ode":  # stage17: supervise the integration path itself
                if int(c.get("ode_steps", 4)) != int(c["lead"]) or self.inter_times != int(c["lead"]) - 1:
                    raise ValueError("ODE path supervision needs ode_steps == lead (6 h sub-steps) and "
                                     "inter_times == lead - 1")
                log(f"ODE path supervision: {self.inter_times} path states "
                    f"({', '.join(f'{6 * j} h' for j in range(1, self.inter_times + 1))}), weight {self.inter_weight}")
            elif c.get("integrator", "delta") != "delta":
                raise ValueError("inter_times needs the delta or ode integrator")
            else:
                self.vnet_kwargs.update(inter_times=self.inter_times)
                log(f"intermediate-time loss: {self.inter_times} auxiliary targets "
                    f"({', '.join(f'{6 * j} h' for j in range(1, self.inter_times + 1))}), weight {self.inter_weight}")
        self.use_prev = bool(c.get("prev_state", False))  # stage14: V-Net input includes X_t - X_{t-lead}
        self.use_nudge = bool(c.get("nudge", False))  # stage14b: first step relaxes toward X_t - M(X_{t-lead})
        # earliest training index whose previous states exist (prev: X_{t-lead}; nudge pre-forecast: X_{t-2 lead})
        self.min_idx = int(c["lead"]) * (int(self.use_prev) + int(self.use_nudge))
        self.diff_cfg = DiffusionConfig(cs=c["diff_cs"], cv=c["diff_cv"], depth=c["diff_depth"], n_basis=c["n_basis"],
                                        hidden=c["hidden"])

    def _init_columns(self, args):
        c = self.cfg
        which, path = c["column_inputs"], c["column_data"]
        spath = self.dir / "column_norm.json"
        stats = json.load(open(spath)) if spath.exists() else None
        self.train.columns = ColumnInputs(path, "train", self.train.times, which, stats)
        if stats is None:
            json.dump(self.train.columns.stats, open(spath, "w"), indent=2)
        stats = self.train.columns.stats
        self.calib.columns = ColumnInputs(path, "calib", self.calib.times, which, stats)
        if getattr(args, "eval_data", ""):
            ext = getattr(args, "eval_column_data", "") or ""
            if not ext:
                raise ValueError("this run uses column inputs: pass --eval-column-data for the --eval-data year")
            self.test.columns = ColumnInputs(ext, "test", self.test.times, which, stats)
        else:
            self.test.columns = ColumnInputs(path, "test", self.test.times, which, stats)
        cs = self.train.columns
        log(f"column inputs {which}: {cs.n_s} scalar + {cs.n_v} vector channels from {path} (initial time only)")

    def save_normalizer(self, path):
        torch.save({k: getattr(self.norm, k) for k in self.norm.__dataclass_fields__
                    if getattr(self.norm, k) is not None}, path)

    # ---------------------------------------------------------------- data helpers
    def prep(self, ds: ERA5Pairs, idx):
        x0, x1, c = ds.batch(idx)
        return self.norm.state(x0), self.norm.state(x1), self.norm.cond(c), x0

    def cond_at(self, ds: ERA5Pairs, idx):
        return self.norm.cond(ds.cond_field(torch.as_tensor(idx, dtype=torch.long)))

    def col(self, ds: ERA5Pairs, idx):
        """stage15a: normalised column channels at ``idx`` (None when the run does not use them)."""
        return ds.columns.field(idx) if self.use_col else None

    def inter_targets(self, ds: ERA5Pairs, idx):
        """stage15b: normalised states X_{t + j 6 h}, j = 1..inter_times (empty when the run has no such head)."""
        return [self.norm.state(ds.state_after(idx, j)) for j in range(1, self.inter_times + 1)]

    def prev(self, ds: ERA5Pairs, idx):
        """stage14: normalised state one lead before ``idx`` (None when the run does not use it); stage18b
        (``prev_path``): the list of states one lead before each 6 h sub-step of the lead."""
        if not self.use_prev:
            return None
        if self.cfg.get("prev_path"):
            return [self.norm.state(x) for x in ds.prev_path(idx)]
        return self.norm.state(ds.prev_state(idx))

    def increment(self, model, ds: ERA5Pairs, idx, state_n: Field):
        """stage14b: nudging increment X_t - M(X_{t-lead}) (delta units) from ``model``'s own pre-forecast of the
        initial time; zero for the first ``lead`` times of a split (no previous state); None when the run does not
        nudge or ``model`` has no forecaster (physics-only diagnostic)."""
        if not self.use_nudge or not hasattr(model, "increment"):
            return None
        idx = torch.as_tensor(idx, dtype=torch.long)
        lead = int(self.cfg["lead"])
        j = torch.clamp(idx - lead, min=0)
        prev_n = self.norm.state(ds.prev_state(idx))
        prev_c = self.norm.cond(ds.cond_field(j))
        pp = self.norm.state(ds.prev_state(j)) if self.use_prev else None
        e = model.increment(state_n, prev_n, prev_c, self.g, pp)
        ok = (idx >= lead).to(e.s)[:, None, None]
        return Field(e.s * ok, e.v * ok)

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
        if m.guide is not None and getattr(m.guide, "curv_ops", None) is not None:
            m.guide.curv_ops.v_scale.copy_(self.norm.v_scale)
        if getattr(m.vnet, "phys_ops", None) is not None:
            m.vnet.phys_ops.v_scale.copy_(self.norm.v_scale)
        if m.guide is not None and getattr(m.guide, "flow_out", None) is not None:  # stage17c flow in m/s
            m.guide.flow_v_scale.copy_(self.norm.v_scale)
        if m.guide is not None and m.guide.resolved is not None:
            m.guide.resolved.set_scales(self.norm.s_mean, self.norm.s_std, self.norm.v_scale, self.norm.tend_s,
                                        self.norm.tend_v)
        return m

    def load_guide(self, m: LieGuidedForecaster, finetuned: bool = True):
        """Stage A guide ``guide.pt``; when Stage B fine-tuned the guide (``--guide-lr``, stage12) the matching
        ``guide_finetuned.pt`` is preferred for inference (Stage B itself starts from ``guide.pt``)."""
        ft = self.dir / "guide_finetuned.pt"
        m.guide.load_state_dict(torch.load(ft if finetuned and ft.exists() else self.dir / "guide.pt"))

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


# guide read-out heads (grid adapter, reconstruction, aux, tendency/source, flow): Stage A --guide-heads-only (stage17c)
# and Stage B --guide-lr with --guide-heads-only (stage18c) train only these, so the features P stay unchanged
GUIDE_HEADS = ("head_dec.", "rec_out.", "aux_out.", "tend_out.", "flow_out.")


def next_prev(model, start):
    """The ``prev`` input of the next rollout lead (stage18b path list when the model follows the ODE path)."""
    return model.next_prev(start) if hasattr(model, "next_prev") else start


# ---------------------------------------------------------------------------- Stage A
def stage_a(run: Run):
    a, g, w = run.args, run.g, run.w
    done_path = run.dir / "a_done.json"
    if int(getattr(a, "resume", 1)) and done_path.exists() and (run.dir / "guide.pt").exists() \
            and int(json.load(open(done_path)).get("steps", -1)) == int(a.steps):
        log(f"stage A already finished in this run directory; skipping (delete {done_path.name} to retrain)")
        return
    model = run.forecaster(True)
    guide = model.guide
    init_guide = getattr(a, "init_guide", "") or ""
    if init_guide:  # stage17c: start from a trained guide (new heads keep their initialisation)
        missing, unexpected = guide.load_state_dict(torch.load(init_guide), strict=False)
        if unexpected:
            raise ValueError(f"--init-guide {init_guide}: unexpected keys {unexpected}")
        log(f"stage A: guide initialised from {init_guide}" + (f" (new: {missing})" if missing else ""))
    use_flow = getattr(guide, "flow_out", None) is not None
    if use_flow:
        log("stage A: Lie transport guide — tendency = semi-Lagrangian transport by the guide flow + source head")

    def step_tendency(feats, x_norm, gview, source):
        """stage17c: guide 6 h tendency (normalised tendency units) = transport by the guide's flow + source."""
        xt = guide.transport(feats, x_norm, gview)
        return Field((xt.s - x_norm.s) / run.norm.tend_s + source.s, (xt.v - x_norm.v) / run.norm.tend_v + source.v)

    if int(getattr(a, "guide_heads_only", 0) or 0):  # stage17c: the trunk (features P) stays exactly as it was
        heads = GUIDE_HEADS
        for n_, p_ in guide.named_parameters():
            p_.requires_grad_(n_.startswith(heads))
        log(f"stage A: trunk frozen, training the heads only "
            f"({sum(p_.numel() for p_ in guide.parameters() if p_.requires_grad)} trainable of {n_params(guide)})")
    opt = torch.optim.Adam([p_ for p_ in guide.parameters() if p_.requires_grad], lr=a.lr)
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
        l_rec = field_mse(rec, xg, w, run.loss_w)
        l_aux = ((aux - aux_t) ** 2 * w[None, :, None]).mean() / w.mean()
        loss = l_rec + a.aux_weight * l_aux
        l_tend = torch.zeros(())
        if tend is not None:
            tend_t = run.norm.tendency(x0, run.train.next_state(idx)).gauge(tg)
            if use_flow:
                tend = step_tendency(feats, xg, gv, tend)
            l_tend = field_mse(tend, tend_t, w, run.loss_w)
            loss = loss + a.tend_weight * l_tend
        l_clos = torch.zeros(())
        if guide.closure_out is not None:  # S5a: closure = observed tendency - fixed resolved dynamics, on mesh 0
            tend_m = guide.resolved.pool(run.norm.tendency(x0, run.train.next_state(idx)).gauge(tg), gv)
            clos_t = tend_m - guide.resolved_tendency(xg, gv)
            l_clos = field_mse(guide.closure(feats), clos_t, w_m)
            loss = loss + a.tend_weight * l_clos
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p_ for p_ in guide.parameters() if p_.requires_grad], 1.0)
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
                v_tend, v_tr, v_zero = 0.0, None, None
                if tend is not None:
                    tend_true = run.norm.tendency(x0, run.calib.next_state(val_idx))
                    if use_flow:
                        zero = Field(torch.zeros_like(tend.s), torch.zeros_like(tend.v))
                        v_tr = float(field_mse(step_tendency(feats, x0n, g, zero), tend_true, w))  # transport alone
                        v_zero = float(field_mse(zero, tend_true, w))  # persistence
                        tend = step_tendency(feats, x0n, g, tend)
                    v_tend = float(field_mse(tend, tend_true, w))
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
                           **({"val_transport_only": v_tr, "val_persistence": v_zero} if use_flow else {}),
                           "t": time.time() - t0})
            log(f"A {step:5d}  L_rec {float(l_rec):.4f} L_aux {float(l_aux):.4f} L_tend {float(l_tend):.4f}"
                f" L_clos {float(l_clos):.4f} | val rec {v_rec:.4f} aux {v_aux:.4f} tend {v_tend:.4f} clos {v_clos:.4f}"
                f" (resolved only {v_full:.4f})"
                + (f" transport only {v_tr:.4f} persistence {v_zero:.4f}" if use_flow else "")
                + f" | {(time.time() - t0) / (step + 1):.2f} s/step")
    json.dump({"steps": int(a.steps), "best": best, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
              open(done_path, "w"))
    log(f"stage A done in {(time.time() - t0) / 60:.1f} min, best val score {best:.4f}"
        + (f", forecast-perturbed batches {n_fc}" if fm is not None else ""))


# ---------------------------------------------------------------------------- Stage B
def stage_b(run: Run, use_guide: bool):
    a, g, w = run.args, run.g, run.w
    name = "vnet" if use_guide else "vnet_noguide"
    model = run.forecaster(use_guide)
    # stage12 options (all off by default so earlier recipes are reproduced exactly):
    #   --guide-lr      fine-tune the guide weights in Stage B with their own (smaller) cosine lr -> guide_finetuned.pt
    #   --ema           exponential moving average of the trained weights for validation and the saved checkpoint
    #   --rollout-schedule  curriculum of autoregressive loss steps, e.g. "500:2,1000:3"
    guide_lr = float(getattr(a, "guide_lr", 0.0) or 0.0) if use_guide else 0.0
    if use_guide:
        run.load_guide(model, finetuned=False)
        model.freeze_guide()
        if guide_lr > 0:  # the guide has no dropout / batch norm, so eval mode only marks it as "not Stage A"
            heads_only = bool(int(getattr(a, "guide_heads_only", 0) or 0))  # stage18c: physics heads only
            for n_, p in model.guide.named_parameters():
                p.requires_grad_(n_.startswith(GUIDE_HEADS) if heads_only else True)
            if heads_only:
                log(f"stage B[{name}]: guide fine-tuning restricted to the heads {GUIDE_HEADS} "
                    f"({sum(p.numel() for p in model.guide.parameters() if p.requires_grad)} params; trunk / P frozen)")
    groups = [{"params": [p for p in model.vnet.parameters() if p.requires_grad], "lr": a.lr, "base_lr": a.lr}]
    if guide_lr > 0:
        groups.append({"params": [p for p in model.guide.parameters() if p.requires_grad], "lr": guide_lr,
                       "base_lr": guide_lr})
    params = [p for gp in groups for p in gp["params"]]
    opt = torch.optim.Adam(groups, lr=a.lr)
    ema_decay = float(getattr(a, "ema", 0.0) or 0.0)
    ema_model = None
    if ema_decay > 0:
        ema_model = copy.deepcopy(model).eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)

    def ema_update(step: int):
        d = min(ema_decay, (1.0 + step) / (10.0 + step))  # warm start so the first steps are not dominated by init
        with torch.no_grad():
            for mod_e, mod in ((ema_model.vnet, model.vnet),) + (((ema_model.guide, model.guide),) if guide_lr > 0 else ()):
                for pe, pm in zip(mod_e.parameters(), mod.parameters()):
                    pe.mul_(d).add_(pm.detach(), alpha=1.0 - d)
                for be, bm in zip(mod_e.buffers(), mod.buffers()):
                    be.copy_(bm)

    rng = np.random.default_rng(a.seed + 1)
    val_idx = list(np.linspace(0, len(run.calib) - 1, 48).round().astype(int))
    best, t0 = float("inf"), time.time()
    K, k_start, lead = int(getattr(a, "rollout_steps", 1)), int(getattr(a, "rollout_start", 0)), run.cfg["lead"]
    sched = parse_rollout_schedule(getattr(a, "rollout_schedule", "") or "")
    if sched:  # curriculum overrides --rollout-steps / --rollout-start
        K, k_start = max(k for _, k in sched), sched[0][0]

    def rollout_steps_at(step: int) -> int:
        if not sched:
            return K if step >= k_start else 1
        k = 1
        for s0, kk in sched:
            if step >= s0:
                k = kk
        return k

    lam = float(getattr(a, "rollout_weight", 1.0))
    val_idx2 = [i for i in val_idx if i < len(run.calib) - lead]  # 2-step validation needs X_{t+2 lead}

    def scaled(f: Field) -> Field:
        return Field(f.s / model.delta_scale_s, f.v / model.delta_scale_v)

    lam_pde = float(getattr(a, "pde_weight", 0.0))
    lam_spec = float(getattr(a, "spec_energy_weight", 0.0) or 0.0)  # stage13b: spectral energy continuity
    grid_hw, grid_lat = g.extras["grid_hw"], g.extras["grid_lat"]
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
        f"rollout steps {K} from step {k_start} (weight {lam})" + (f", PDE residual weight {lam_pde}" if use_pde else "")
        + (f", rollout schedule {sched}" if sched else "") + (f", EMA decay {ema_decay}" if ema_model is not None else "")
        + (f", guide fine-tuning lr {guide_lr} ({n_params(model.guide)} params)" if guide_lr > 0 else "")
        + (f", spectral energy loss weight {lam_spec}" if lam_spec > 0 else "")
        + (f", neural operator {model.vnet.cfg.neural_op} ({n_params(model.vnet.cont_op)} params)"
           if model.vnet.cont_op is not None else ""))
    if model.vnet.cfg.integrator == "ode":  # stage S8
        vc = model.vnet.cfg
        log(f"stage B[{name}]: graph neural ODE dynamics, {vc.ode_steps} sub-steps of {6 * lead // vc.ode_steps} h, "
            f"solver {vc.ode_solver}, physics drift {'on' if vc.ode_physics else 'off'}"
            + (" (learned gain, init 0)" if vc.ode_drift_gain else "")
            + (", sub-step solar forcing" if vc.ode_path_forcing else "")
            + (", graph tendency core (beta-scaled guide transport + conservative face flux + source"
               + (", core tokens C in PIGT)" if vc.core_tokens else ")") if vc.tendency_core else ""))
    # restart safety: a finished Stage B is skipped, an interrupted one resumes from its last validation checkpoint
    start_step = 0
    ck_path, done_path = run.dir / f"{name}_train_state.pt", run.dir / f"{name}_done.json"
    if int(getattr(a, "resume", 1)) and done_path.exists() and (run.dir / f"{name}.pt").exists() \
            and int(json.load(open(done_path)).get("steps", -1)) == int(a.steps):
        log(f"stage B[{name}] already finished in this run directory; skipping (delete {done_path.name} to retrain)")
        return
    if int(getattr(a, "resume", 1)) and ck_path.exists():
        ck = torch.load(ck_path, weights_only=False)
        model.vnet.load_state_dict(ck["vnet"])
        if guide_lr > 0 and "guide" in ck:
            model.guide.load_state_dict(ck["guide"])
        if ema_model is not None and "ema_vnet" in ck:
            ema_model.vnet.load_state_dict(ck["ema_vnet"])
            if guide_lr > 0 and "ema_guide" in ck:
                ema_model.guide.load_state_dict(ck["ema_guide"])
        opt.load_state_dict(ck["opt"])
        best, start_step = float(ck["best"]), int(ck["step"])
        rng.bit_generator.state = ck["rng"]
        torch.set_rng_state(ck["torch_rng"])
        t0 = time.time() - float(ck.get("elapsed", 0.0))
        lp = run.dir / f"log_{name}.jsonl"  # rows at or after the resume step are logged again
        if lp.exists():
            rows = [line for line in open(lp) if json.loads(line)["step"] < start_step]
            open(lp, "w").writelines(rows)
        log(f"stage B[{name}]: resumed from step {start_step} (best {best:.4f}, {time.time() - t0:.0f} s elapsed before)")
    init_vnet = getattr(a, "init_vnet", "") or ""
    if start_step == 0 and init_vnet:  # stage17: paired fine-tuning from a trained V-Net (new parameters keep their init)
        missing, unexpected = model.vnet.load_state_dict(torch.load(init_vnet), strict=False)
        if unexpected:
            raise ValueError(f"--init-vnet {init_vnet}: unexpected keys {unexpected}")
        if ema_model is not None:
            ema_model.vnet.load_state_dict(model.vnet.state_dict())
        log(f"stage B[{name}]: V-Net initialised from {init_vnet}" + (f" (new: {missing})" if missing else ""))
    for step in range(start_step, a.steps):
        for pg in opt.param_groups:
            pg["lr"] = cosine_lr(step, a.steps, pg.get("base_lr", a.lr))
        k_now = rollout_steps_at(step)
        idx = rng.integers(run.min_idx, len(run.train) - (k_now - 1) * lead, a.batch)
        states, conds = run.train.sequence(idx, k_now)
        xs = [run.norm.state(x) for x in states]
        cs = [run.norm.cond(c) for c in conds]
        x_prev = run.prev(run.train, idx)
        f, d = model(xs[0], cs[0], g, prev=x_prev, nudge=run.increment(model, run.train, idx, xs[0]),
                     col=run.col(run.train, idx))
        x_prev, nud = next_prev(model, xs[0]), model.carried_nudge()
        l_f = field_mse(scaled(f), scaled(xs[1]), w, run.loss_w)  # == delta MSE of the one-step target
        l_int = torch.zeros(())
        if run.inter_times:  # stage15b: intermediate times of the first lead, same delta units as L_f
            mids = model.inter_forecasts(xs[0])
            l_int = sum(field_mse(scaled(fa), scaled(ta), w, run.loss_w)
                        for fa, ta in zip(mids, run.inter_targets(run.train, idx))) / run.inter_times
        l_phys = global_tendency_consistency(d.s * model.delta_scale_s, xs[1].s - xs[0].s, w)
        l_pde = pde_residual(xs[0], f, cs[0]) if use_pde else torch.zeros(())
        l_roll = torch.zeros(())
        l_spec = spectral_energy_loss(f, xs[1], *grid_hw, grid_lat) if lam_spec > 0 else torch.zeros(())
        for k in range(2, k_now + 1):  # autoregressive steps; physics features recomputed from the forecast
            f_new, _ = model(f, cs[k - 1], g, prev=x_prev, nudge=nud)
            x_prev, f, nud = next_prev(model, f), f_new, model.carried_nudge()
            l_roll = l_roll + field_mse(scaled(f), scaled(xs[k]), w, run.loss_w)
            if lam_spec > 0:
                l_spec = l_spec + spectral_energy_loss(f, xs[k], *grid_hw, grid_lat)
        loss = l_f + a.phys_weight * l_phys + lam * l_roll + lam_pde * l_pde + lam_spec * l_spec + run.inter_weight * l_int
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if ema_model is not None:
            ema_update(step)
        if step % 25 == 0:
            run.jlog(name, {"step": step, "l_f": float(l_f), "l_phys": float(l_phys), "l_roll": float(l_roll),
                            "l_pde": float(l_pde), "l_spec": float(l_spec), "k": k_now, "t": time.time() - t0,
                            **({"l_int": float(l_int)} if run.inter_times else {})})
        if step % a.val_every == 0 or step == a.steps - 1:
            model.eval()
            vm = ema_model if ema_model is not None else model  # validated / saved weights
            with torch.no_grad():
                tot, tot2 = 0.0, 0.0
                tot_int = np.zeros(run.inter_times)
                for i in range(0, len(val_idx), 8):
                    x0n, x1n, cn, _ = run.prep(run.calib, val_idx[i:i + 8])
                    fc, _ = vm(x0n, cn, g, prev=run.prev(run.calib, val_idx[i:i + 8]),
                               nudge=run.increment(vm, run.calib, val_idx[i:i + 8], x0n),
                               col=run.col(run.calib, val_idx[i:i + 8]))
                    tot += float(field_mse(fc, x1n, w)) * len(val_idx[i:i + 8])
                    if run.inter_times:  # stage15b: MSE of the intermediate-time head (diagnostic, not selected on)
                        for j, (fa, ta) in enumerate(zip(vm.inter_forecasts(x0n),
                                                         run.inter_targets(run.calib, val_idx[i:i + 8]))):
                            tot_int[j] += float(field_mse(fa, ta, w)) * len(val_idx[i:i + 8])
                for i in range(0, len(val_idx2), 8):
                    states, conds = run.calib.sequence(val_idx2[i:i + 8], 2)
                    fc, pv = run.norm.state(states[0]), run.prev(run.calib, val_idx2[i:i + 8])
                    nud = run.increment(vm, run.calib, val_idx2[i:i + 8], fc)
                    cl = run.col(run.calib, val_idx2[i:i + 8])
                    for k in range(2):
                        fc_new, _ = vm(fc, run.norm.cond(conds[k]), g, prev=pv, nudge=nud, col=cl)
                        pv, fc, nud, cl = next_prev(vm, fc), fc_new, vm.carried_nudge(), None
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
                torch.save(vm.vnet.state_dict(), run.dir / f"{name}.pt")
                if guide_lr > 0:
                    torch.save(vm.guide.state_dict(), run.dir / "guide_finetuned.pt")
            nb = vm.vnet.nudge_block
            v_int = (tot_int / len(val_idx)).tolist()
            run.jlog(name, {"step": step, "val_mse": v_mse, "val_mse2": v_mse2, "val_persistence_mse": pers,
                            **({"val_inter_mse": v_int} if run.inter_times else {}),
                            "best": best, "attention_mass": model.vnet.attention_mass(), "t": time.time() - t0,
                            **({"nudge_alpha": nb.last_alpha.tolist()} if nb is not None and nb.last_alpha is not None else {}),
                            **({"drift_gain": vm.vnet.drift_gain_s.tolist() + vm.vnet.drift_gain_v.tolist()}
                               if vm.vnet.drift_gain_s is not None else {}),
                            **({"core": vm.last_core} if getattr(vm, "last_core", None) else {})})
            log(f"B[{name}] {step:5d}  L_f {float(l_f):.4f} L_roll {float(l_roll):.4f} L_phys {float(l_phys):.5f}"
                + (f" L_pde {float(l_pde):.4f}" if use_pde else "")
                + (f" L_int {float(l_int):.4f}" if run.inter_times else "")
                + f" | val MSE {v_mse:.4f} 2-step {v_mse2:.4f} (persistence {pers:.4f})"
                + (" 6/12/18 h " + "/".join(f"{x:.4f}" for x in v_int) if run.inter_times else "")
                + (" gain " + "/".join(f"{float(x):.2f}" for x in torch.cat([vm.vnet.drift_gain_s, vm.vnet.drift_gain_v]))
                   if vm.vnet.drift_gain_s is not None else "")
                + (" core beta " + "/".join(f"{b:.2f}" for b in vm.last_core["beta"])
                   + f" rms T {vm.last_core['transport']:.3f} F {vm.last_core['flux']:.3f} S {vm.last_core['source']:.3f}"
                   if getattr(vm, "last_core", None) else "")
                + f" | {(time.time() - t0) / (step + 1):.2f} s/step")
            state = {"step": step + 1, "vnet": model.vnet.state_dict(), "opt": opt.state_dict(), "best": best,
                     "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(), "elapsed": time.time() - t0}
            if guide_lr > 0:
                state["guide"] = model.guide.state_dict()
            if ema_model is not None:
                state["ema_vnet"] = ema_model.vnet.state_dict()
                if guide_lr > 0:
                    state["ema_guide"] = ema_model.guide.state_dict()
            torch.save(state, ck_path)
        if int(getattr(a, "stop_after", 0) or 0) and step + 1 >= int(a.stop_after):
            log(f"stage B[{name}]: stopping after step {step} (--stop-after, testing aid)")
            return
    json.dump({"steps": int(a.steps), "best": best, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
              open(done_path, "w"))
    if ck_path.exists():
        ck_path.unlink()
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
            xbar, _ = model(x0n, cn, g, prev=run.prev(run.calib, idx[i:i + 8]),
                            nudge=run.increment(model, run.calib, idx[i:i + 8], x0n),
                            col=run.col(run.calib, idx[i:i + 8]))
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

    def context(x0n, cn, prev=None, nudge=None, col=None):
        with torch.no_grad():
            P = model.physics(x0n, cn, g)
            xbar, _ = model(x0n, cn, g, prev=prev, nudge=nudge, col=col)
        return xbar, {"xbar": xbar, "x": x0n, "cond": cn, "P0u": P["P0u"]}

    opt = torch.optim.Adam(diff.denoiser.parameters(), lr=a.lr)
    t0, hist, hist_f, hist_m = time.time(), [], [], []
    for step in range(a.steps):
        for pg in opt.param_groups:
            pg["lr"] = cosine_lr(step, a.steps, a.lr)
        idx = rng.integers(0, len(run.calib), a.batch)
        x0n, x1n, cn, _ = run.prep(run.calib, idx)
        xbar, ctx = context(x0n, cn, run.prev(run.calib, idx), run.increment(model, run.calib, idx, x0n),
                            run.col(run.calib, idx))
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

    def forward(self, state: Field, cond: Field, g, prev: Optional[Field] = None,
                nudge: Optional[Field] = None, col: Optional[Field] = None):  # prev / nudge / col unused (no V-Net)
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
    if m.vnet.cfg.integrator == "ode" and (m.vnet.cfg.ode_physics or m.vnet.core_on):
        # stage S8b / stage18 diagnostic: the guide's own drift (core: transport by its flow + its source), V-Net off
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
    if getattr(a, "max_cases", 0):  # common init times with external forecasts that end earlier in the year
        cases = cases[: a.max_cases]
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
            prevs = {name: run.prev(test, [i]) for name in models}  # stage14 (first init time: zero change)
            nudges = {name: run.increment(mdl, test, [i], x0n) for name, mdl in models.items()}  # stage14b
            col0 = run.col(test, [i])  # stage15a: initial time only
            for k in range(1, K + 1):
                j = i + (k - 1) * lead
                cn = run.cond_at(test, [j])
                _, x1n, _, _ = run.prep(test, [j])
                add(truth, k, x1n)
                valid_times[k].append(test.times[j + lead])
                add(preds["persistence"], k, x0n)
                add(preds["climatology"], k, clim_f)
                for name, mdl in models.items():
                    new, _ = mdl(states[name], cn, g, prev=prevs[name], nudge=nudges[name],
                                 col=col0 if k == 1 else None)
                    prevs[name], states[name] = next_prev(mdl, states[name]), new
                    nudges[name] = getattr(mdl, "carried_nudge", lambda: None)()
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
    # (--n-ens-cases 0 skips it, e.g. for the external benchmark years; there is nothing to score then)
    if diff is not None and a.n_ens_cases > 0:
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
                xbar, _ = mdl(x0n, cn, g, prev=run.prev(test, [i]), nudge=run.increment(mdl, test, [i], x0n),
                              col=run.col(test, [i]))
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
    tag = f"_{a.eval_tag}" if getattr(a, "eval_tag", "") else ""
    if getattr(a, "eval_data", ""):
        results["eval_data"] = a.eval_data
        results["init_times"] = [str(test.times[i].astype("datetime64[ns]")) for i in cases]
    with open(run.dir / f"eval{tag}.json", "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    write_markdown(run, results, tag)
    log(f"eval written to {run.dir / f'eval{tag}.json'}")


def write_markdown(run: Run, r: dict, tag: str = ""):
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
    (run.dir / f"eval{tag}.md").write_text("\n".join(lines) + "\n")


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
    ap.add_argument("--loss-focus", default="", choices=[""] + sorted(SPECS),
                    help="stage10b: weight the training losses towards this spec's channels (others: --loss-focus-weight)")
    ap.add_argument("--loss-focus-weight", type=float, default=0.2)
    ap.add_argument("--simplicial", type=int, default=0, choices=[0, 1, 2],
                    help="stage11b: simplicial vertex-edge-face message passing blocks in the guide (mesh 0)")
    ap.add_argument("--curvature-ops", type=int, default=0, choices=[0, 1],
                    help="guide: exact connection / curvature operators of the flow (DEC vorticity, divergence, "
                         "strain, Okubo-Weiss, Helmholtz potentials, Lie brackets) as extra mesh-0 inputs "
                         "(stage11a; stored in config.json)")
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
    ap.add_argument("--prev-state", type=int, default=0, choices=[0, 1],
                    help="stage14: V-Net also sees the observed change over the previous lead (X_t - X_{t-lead})")
    ap.add_argument("--nudge", type=int, default=0, choices=[0, 1],
                    help="stage14b: initial-increment nudging — the first step relaxes toward the model's own error "
                         "over the previous lead, X_t - M(X_{t-lead}), with a learned coefficient (stored in config.json)")
    ap.add_argument("--nudge-carry", type=int, default=1, choices=[0, 1],
                    help="stage14b: carry the added term into later rollout steps (red-noise recursion q_k = rho q_{k-1}, "
                         "Evensen 2009 Ch. 12); 0 = first step only")
    ap.add_argument("--nudge-alpha-init", type=float, default=0.12,
                    help="stage14b: initial uniform nudging coefficient = prior rho (scripts/model_error_autocorr.py)")
    ap.add_argument("--column-inputs", default="", choices=[""] + sorted(COLUMN_SETS),
                    help="stage15a: column channels as V-Net inputs at the initial time only (stored in config.json)")
    ap.add_argument("--column-data", default="", help="stage15a: npz with the column channels (same splits/times as --data)")
    ap.add_argument("--inter-times", type=int, default=0,
                    help="stage15b: intermediate data steps (6 h) supervised by an auxiliary head inside the first lead "
                         "(3 = 6/12/18 h for a 24 h lead; 0 = off); training loss only, forecasts unchanged")
    ap.add_argument("--inter-weight", type=float, default=1.0,
                    help="stage15b: weight of the mean intermediate-time loss (same delta units as L_f)")
    ap.add_argument("--neural-op", default="", choices=["", "fft", "sht"],
                    help="stage13: continuous neural-operator layer on the grid after the mesh V-Net ('' = off)")
    ap.add_argument("--no-hidden", type=int, default=16, help="stage13: neural operator hidden channels")
    ap.add_argument("--no-layers", type=int, default=2, help="stage13: number of FNO blocks")
    ap.add_argument("--no-modes", type=int, nargs=2, default=[16, 32], help="stage13: kept modes (lat, lon)")
    ap.add_argument("--spec-energy-weight", type=float, default=0.0,
                    help="stage13b: Stage B weight of the zonal spectral energy loss (0 = off)")
    ap.add_argument("--ode-physics", type=int, default=0, choices=[0, 1],
                    help="stage S8: add the frozen guide's 6 h tendency as the known drift of the ODE")
    ap.add_argument("--ode-path-forcing", type=int, default=0, choices=[0, 1],
                    help="stage17: solar forcing follows the ODE sub-step time (stored in config.json)")
    ap.add_argument("--ode-drift-gain", type=int, default=0, choices=[0, 1],
                    help="stage17b: learnable per-variable gain on the guide drift, zero-initialised (needs --ode-physics)")
    ap.add_argument("--flow-head", type=int, default=0, choices=[0, 1],
                    help="stage17c: Lie transport guide (flow head + semi-Lagrangian transport; stored in config.json)")
    ap.add_argument("--tendency-core", type=int, default=0, choices=[0, 1],
                    help="stage18: graph tendency core (needs --integrator ode, PIGT fusion, --flow-head 1, "
                         "--ode-physics 0)")
    ap.add_argument("--prev-path", type=int, default=0, choices=[0, 1],
                    help="stage18b: previous-state input along the ODE path (needs --prev-state 1, --integrator ode)")
    ap.add_argument("--core-tokens", type=int, default=1, choices=[0, 1],
                    help="stage18: give the guide's core terms to PIGT as a fourth token source C")
    ap.add_argument("--ode-drift", default="tend", choices=["tend", "transport"],
                    help="stage17c: guide drift of the ODE = tendency head (tend) or Lie transport guide (transport)")
    ap.add_argument("--guide-heads-only", type=int, default=0, choices=[0, 1],
                    help="stage17c: Stage A trains only the guide heads (grid adapter, read-outs, tendency, flow); the "
                         "trunk and hence the features P seen by the V-Net stay unchanged.  stage18c: with --guide-lr, "
                         "Stage B fine-tunes only these heads along the forecast path")
    ap.add_argument("--init-guide", default="",
                    help="stage17c: Stage A starts from this guide state dict (new heads keep their initialisation)")
    ap.add_argument("--init-vnet", default="",
                    help="stage17: Stage B starts from this V-Net state dict (paired fine-tuning; fresh optimiser and lr)")
    ap.add_argument("--perturb-model", default=None,
                    help="Stage A: vnet state dict of a guide-free forecaster whose lead-time forecast replaces the "
                         "input with probability --perturb-prob (structured, forecast-like perturbations)")
    ap.add_argument("--perturb-prob", type=float, default=0.3)
    ap.add_argument("--rollout-steps", type=int, default=1, help="Stage B: autoregressive steps in the loss")
    ap.add_argument("--rollout-start", type=int, default=0, help="Stage B: first step that uses the rollout loss")
    ap.add_argument("--rollout-weight", type=float, default=1.0)
    ap.add_argument("--rollout-schedule", default="",
                    help="Stage B (stage12): rollout curriculum 'step:k,step:k', e.g. '500:2,1000:3'; overrides "
                         "--rollout-steps/--rollout-start")
    ap.add_argument("--ema", type=float, default=0.0,
                    help="Stage B (stage12): EMA decay of the trained weights used for validation and the checkpoint (0 = off)")
    ap.add_argument("--guide-lr", type=float, default=0.0,
                    help="Stage B (stage12): fine-tune the guide with this peak lr (0 = frozen); saves guide_finetuned.pt")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--sigma-max", type=float, default=0.5)
    ap.add_argument("--aux-weight", type=float, default=0.5)
    ap.add_argument("--phys-weight", type=float, default=0.1)
    ap.add_argument("--eval-stride", type=int, default=16, help="init-time stride in 6 h steps")
    ap.add_argument("--eval-data", default="", help="score on the test split of this npz instead (benchmark year); "
                    "normaliser, climatology and models stay those of the run")
    ap.add_argument("--eval-column-data", default="", help="stage15a runs: column channels of the --eval-data year")
    ap.add_argument("--eval-tag", default="", help="write eval_<tag>.json / eval_<tag>.md instead of eval.json")
    ap.add_argument("--max-cases", type=int, default=0, help="keep only the first N init times (0 = all)")
    ap.add_argument("--max-lead-steps", type=int, default=5)
    ap.add_argument("--n-ens-cases", type=int, default=24)
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--sample-steps", type=int, default=20)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=int, default=1, choices=[0, 1],
                    help="skip a finished Stage A/B in the run directory and resume an interrupted Stage B from its "
                         "last validation checkpoint (<name>_train_state.pt); 0 = always train from scratch")
    ap.add_argument("--stop-after", type=int, default=0, help="testing aid: leave Stage B after this many steps")
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
