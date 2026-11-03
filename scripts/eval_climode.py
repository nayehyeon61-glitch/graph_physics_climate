#!/usr/bin/env python
"""Score the published ClimODE checkpoint (Verma, Heinonen, Garg, ICLR 2024) on *our* 2019 test protocol.

ClimODE is the "climate ODE" baseline: a neural advection ODE on the 5.625 deg lat/lon grid (32 x 64, same grid
as the WeatherBench2 64x32 data used here) for z500, t850, t2m, u10, v10, with an initial velocity fitted from
the last three 6-hourly frames and an emission network for the Gaussian uncertainty.  The authors' code is
used unchanged from the copy vendored in PG-Flow (``physical_guide_flowmatching/third_party/climode``,
Aalto-QuML/ClimODE @ e729d23, MIT) and the public checkpoint ``checkpoints/ClimODE_global.pt`` of that commit
(trained on 2006-2015, validated 2016, tested 2017-2018 by the authors; 2019 is unseen by it).

What is kept exactly as in the authors' evaluation
  * min-max normalisation with the 2006-2018 extrema of each variable (``data/climode_minmax_2006_2018.json``,
    computed from the same WB2 zarr), inputs in ascending-latitude order, constants (orography [m], land-sea
    mask, lat/lon maps), the velocity fit (``utils.get_delta_u`` + ``utils.optimize_vel`` with the authors'
    ``Optim_velocity`` and Gaussian kernel), hourly Euler integration with 6 h outputs, the emission network.
What differs
  * initial conditions: the 90 init times of ``scripts/train_eval.py --stage eval`` (2019, 00 UTC, every 4 days)
    instead of the authors' 12 UTC every-2-days list of 2017-2018; one IC per forward pass (the authors batch
    two years at the same index), so ``model.eval()`` is used -- the authors' script never calls it and its
    BatchNorm would otherwise normalise with the statistics of a single sample;
  * scores: the same WeatherBench2 area-weighted RMSE as ``eval.json`` of our runs (RMSE per init time, then
    the mean over init times, identical to the authors' definition).
The output ``<out>/eval.json`` has the schema of our run ``eval.json`` (``rmse[lead][model][var]``) so that
``scripts/compare_runs.py`` can place ClimODE next to our models (``"ClimODE=runs/climode_2019:climode"``).

Caveats for the comparison (also written into the report): ClimODE saw 10 training years (2006-2015) and
2.75 M parameters versus our 2 years (2016-2017) and 0.55-0.68 M; it predicts 5 variables, ours 11.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpclimate.training.metrics import wb2_scores  # noqa: E402

CKPT_URL = ("https://raw.githubusercontent.com/Aalto-QuML/ClimODE/e729d23e8799ce0e075699e76d60227d848d8d0c/"
            "checkpoints/ClimODE_global.pt")
VARS = ["z500", "t850", "t2m", "u10", "v10"]
CANDIDATE_DIRS = [
    os.environ.get("CLIMODE_DIR", ""),
    os.path.join(os.environ.get("GPC_REPOS_ROOT", ""), "physical_guide_flowmatching", "third_party", "climode"),
    "/home/claude/nayehyeon61-glitch/physical_guide_flowmatching/third_party/climode",
    str(ROOT.parent / "nayehyeon61-glitch" / "physical_guide_flowmatching" / "third_party" / "climode"),
    str(ROOT.parent / "physical_guide_flowmatching" / "third_party" / "climode"),
]


def find_climode_dir(explicit: str | None) -> str:
    for d in ([explicit] if explicit else []) + CANDIDATE_DIRS:
        if d and os.path.exists(os.path.join(d, "model_function.py")):
            return d
    raise FileNotFoundError("vendored ClimODE code not found; pass --climode-dir (PG-Flow third_party/climode)")


def gauss_kernel_inverse(lat: np.ndarray, lon: np.ndarray) -> torch.Tensor:
    """Vectorised ``utils.get_gauss_kernel``: K_ij = exp(-|pos_i - pos_j|^2 / 2) in degrees, returns K^-1."""
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    pos = torch.tensor(np.stack([LAT.ravel(), LON.ravel()], 1), dtype=torch.float32)
    d2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
    return torch.linalg.inv(torch.exp(-d2 / 2.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/era5_wb2_64x32_2016_2019.npz"))
    ap.add_argument("--minmax", default=str(ROOT / "data/climode_minmax_2006_2018.json"))
    ap.add_argument("--ckpt", default=str(ROOT / "data/ClimODE_global.pt"), help=f"downloaded from {CKPT_URL}")
    ap.add_argument("--climode-dir", default=None)
    ap.add_argument("--out", default=str(ROOT / "runs/climode_2019"))
    ap.add_argument("--lead", type=int, default=4, help="6 h steps per lead unit (4 = 24 h), as in train_eval")
    ap.add_argument("--max-lead-steps", type=int, default=5)
    ap.add_argument("--eval-stride", type=int, default=16)
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--solver", default=None, help="override the checkpoint's ODE solver (default: as saved, euler)")
    ap.add_argument("--train-mode", action="store_true", help="mimic the authors' script (no model.eval())")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0, help="process init times shard::n_shards (parallel CPU processes)")
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--merge-only", action="store_true", help="only merge existing shard files into eval.json")
    ap.add_argument("--batch-ics", type=int, default=1, help="init times per forward pass (authors batch 2 years)")
    ap.add_argument("--mkldnn", action="store_true", help="keep oneDNN convolutions on (off by default, see below)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.backends.mkldnn.enabled = a.mkldnn  # PG-Flow found oneDNN off ~2x faster for these convs on CPU
    # 27% of the checkpoint's conv weights are float32 subnormals (|w| < 1.2e-38); on x86 every multiply with a
    # subnormal takes a microcode assist, which made one 128x128 3x3 conv cost 240 ms instead of 8 ms.  Flushing
    # subnormals to zero changes the outputs by less than float32 resolution and speeds the ODE up ~30x.
    torch.set_flush_denormal(True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cdir = find_climode_dir(a.climode_dir)
    sys.path.insert(0, str(ROOT / "gpclimate/external/climode_shims"))  # torchcubicspline stand-in
    sys.path.insert(0, cdir)
    import model_function  # noqa: E402  (classes referenced by the pickled checkpoint)
    import utils as cu  # noqa: E402

    # ------------------------------------------------------------------ data in ClimODE's conventions
    z = np.load(a.data)
    lat, lon = z["lat"].astype(np.float64), z["lon"].astype(np.float64)
    assert lat[0] < lat[-1], "ClimODE expects ascending latitude"
    H, W = len(lat), len(lon)
    mm = json.load(open(a.minmax))["values"]
    mx = np.array([mm[v]["max"] for v in VARS], np.float32)
    mn = np.array([mm[v]["min"] for v in VARS], np.float32)
    phys_test = np.stack([z[f"test/{v}"] for v in VARS], 1)  # [T, 5, H, W] physical
    phys_prev = np.stack([z[f"calib/{v}"][-2:] for v in VARS], 1)  # last two 2018 frames (velocity fit of IC 0)
    series = np.concatenate([phys_prev, phys_test], 0)
    norm = torch.from_numpy((series - mn[None, :, None, None]) / (mx - mn)[None, :, None, None]).float()
    test_times = z["test/time"].astype("datetime64[ns]")
    const = torch.from_numpy(np.stack([z["static/orog"] / 9.80665, z["static/lsm"]], 0)).float().view(1, 2, H, W)
    lat_map = torch.from_numpy(np.repeat(lat[:, None], W, 1)).float()
    lon_map = torch.from_numpy(np.repeat(lon[None, :], H, 0)).float()

    lead, K = a.lead, a.max_lead_steps
    n_pairs = len(phys_test) - lead
    last = n_pairs - 1 - (K - 1) * lead
    cases = list(range(0, last, a.eval_stride))
    if a.max_cases:
        cases = cases[: a.max_cases]
    cases = cases[a.shard::a.n_shards]
    L = lead * K  # 6 h steps to integrate

    # ------------------------------------------------------------------ model
    if not os.path.exists(a.ckpt):
        import urllib.request
        urllib.request.urlretrieve(CKPT_URL, a.ckpt)
    model = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if a.solver:
        model.method = a.solver
    if not a.train_mode:
        model.eval()
    kernel = gauss_kernel_inverse(lat, lon)
    info = {"class": type(model).__name__, "solver": model.method, "params": sum(p.numel() for p in model.parameters()),
            "eval_mode": not a.train_mode, "ckpt_url": CKPT_URL, "climode_dir": cdir}
    print(f"ClimODE {info}; shard {a.shard}/{a.n_shards}: {len(cases)} init times, "
          f"leads {[6 * lead * k for k in range(1, K + 1)]} h", flush=True)

    # ------------------------------------------------------------------ forecasts of this shard
    shard_file = out / f"shard_{a.shard}of{a.n_shards}.npz"
    if not a.merge_only:
        preds, fit_loss, t0 = [], [], time.time()
        for n0 in range(0, len(cases), a.batch_ics):
            # Several init times share one forward pass exactly as the authors batch the years 2017 and 2018 at the
            # same index: the ODE right-hand side depends on time only through the hour of day, which is 00 UTC for
            # every init time and lead here, so the time grid of the first init time serves the whole batch.
            chunk = cases[n0:n0 + a.batch_ics]
            Bc, i0 = len(chunk), chunk[0]
            torch.manual_seed(a.seed + i0)
            past = torch.stack([norm[i:i + 3] for i in chunk], 0)  # [B, 3, 5, H, W]; series index = i + 2
            delta_u = cu.get_delta_u(past, torch.tensor([i0 - 2, i0 - 1, i0]))  # spline derivative at t_i (6 h spacing)
            x0 = torch.stack([norm[i + 2] for i in chunk], 0).view(Bc, 1, 5, H, W)
            v_x, v_y, losses, _ = cu.optimize_vel(Bc, x0, delta_u, model_function.Optim_velocity, kernel, H, W)
            fit_loss.extend([float(min(losses))] * Bc)
            vel = torch.cat([v_x, v_y], 1).view(Bc, 10, H, W).detach()
            model.update_param([vel, const, lat_map, lon_map])
            T = torch.arange(i0, i0 + L + 1).float()
            with torch.no_grad():
                mean, std, _ = model(T, x0)
            mean = mean.view(L + 1, Bc, 5, H, W).numpy() * (mx - mn)[None, None, :, None, None] + mn[None, None, :, None, None]
            for b in range(Bc):
                preds.append(np.stack([mean[k * lead, b] for k in range(1, K + 1)], 0))  # [K, 5, H, W] physical units
            print(f"ic {n0 + Bc}/{len(cases)} ({time.time() - t0:.0f}s, vel-fit loss {fit_loss[-1]:.2e})", flush=True)
        np.savez_compressed(shard_file, cases=np.array(cases), pred=np.stack(preds, 0).astype(np.float32),
                            fit_loss=np.array(fit_loss), seconds=time.time() - t0)
        print(f"shard {a.shard}/{a.n_shards} written: {shard_file}", flush=True)

    # ------------------------------------------------------------------ merge all shards and score
    shards = [out / f"shard_{s}of{a.n_shards}.npz" for s in range(a.n_shards)]
    missing = [s.name for s in shards if not s.exists()]
    if missing:
        print(f"shards {missing} not finished yet; run again with --merge-only when they are", flush=True)
        return
    parts = [np.load(s) for s in shards]
    cases_m = np.concatenate([p["cases"] for p in parts])
    order = np.argsort(cases_m)
    cases_m = cases_m[order].tolist()
    pred = np.concatenate([p["pred"] for p in parts], 0)[order]  # [n, K, 5, H, W]
    fit_loss = np.concatenate([p["fit_loss"] for p in parts])[order]
    seconds = float(sum(float(p["seconds"]) for p in parts))

    res = {"n_cases": len(cases_m), "lead_hours": {}, "rmse": {}, "finite": {}, "model": info, "n_shards": a.n_shards,
           "init_times": [str(test_times[i]) for i in cases_m], "vel_fit_loss_mean": float(np.mean(fit_loss)),
           "seconds": seconds, "notes": __doc__.split("Caveats")[1].strip()}
    for k in range(1, K + 1):
        idx = [i + k * lead for i in cases_m]
        times = test_times[idx]
        tr = {v: phys_test[idx, c] for c, v in enumerate(VARS)}
        fc = {v: pred[:, k - 1, c] for c, v in enumerate(VARS)}
        pe = {v: phys_test[cases_m, c] for c, v in enumerate(VARS)}
        res["lead_hours"][k] = 6 * lead * k
        res["finite"].setdefault("climode", {})[k] = bool(all(np.isfinite(x).all() for x in fc.values()))
        res["rmse"][k] = {"climode": wb2_scores(fc, tr, lat, lon, times)["rmse"],
                          "persistence": wb2_scores(pe, tr, lat, lon, times)["rmse"]}
    json.dump(res, open(out / "eval.json", "w"), indent=1, default=str)
    lines = [f"# ClimODE (published checkpoint) on the 2019 test protocol: {len(cases_m)} init times\n",
             "| lead | " + " | ".join(f"{v} ClimODE / persistence" for v in VARS) + " |", "|---|" + "---|" * len(VARS)]
    for k in range(1, K + 1):
        r = res["rmse"][k]
        lines.append(f"| {res['lead_hours'][k]} h | " + " | ".join(
            f"{r['climode'][v]:.4g} / {r['persistence'][v]:.4g}" for v in VARS) + " |")
    (out / "eval.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
