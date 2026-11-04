#!/usr/bin/env python
"""Score a published WeatherBench2 forecast dataset (HRES, Pangu-Weather, GraphCast, ...) on *our* test protocol.

WeatherBench2 (Rasp et al. 2024) hosts the forecasts of the operational and ML models regridded to the same
64x32 equiangular conservative grid as our ERA5 data (``gs://weatherbench2/datasets/<model>/...64x32...zarr``,
dims time x prediction_timedelta x [level] x longitude x latitude).  This script takes the init times and leads of
``scripts/train_eval.py --stage eval`` (test split of ``--data``: init every ``--eval-stride`` 6 h steps from the
first time, leads 24 h x 1..K), reads those forecasts and scores them with the same WB2 area-weighted RMSE against
the same ERA5 truth as our runs.  Output ``<out>/eval.json`` has our schema (``rmse[lead][name][var]``), so
``scripts/compare_runs.py "HRES=runs/wb2_hres_2019:hres"`` can place the model next to ours.

Caveats written into every comparison: these models run at 0.25 deg (HRES 0.1 deg) on far more data/compute and
are regridded to 64x32 afterwards, so they are an upper reference, not the same class as 5.625 deg models;
HRES is scored against ERA5 here (WB2's headline scores HRES against its own analysis), which penalises it at
short leads; an ensemble mean (``--name ens_mean``) is smoother than a single forecast.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpclimate.data.era5_wb2 import SPECS, WB2_SOURCES  # noqa: E402
from gpclimate.training.metrics import wb2_scores  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="model key written into eval.json (e.g. hres, pangu)")
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--data", default=str(ROOT / "data/era5_wb2_64x32_2016_2019.npz"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--spec", default="base11")
    ap.add_argument("--lead", type=int, default=4, help="data steps (6 h) per lead")
    ap.add_argument("--max-lead-steps", type=int, default=5)
    ap.add_argument("--eval-stride", type=int, default=16)
    ap.add_argument("--max-cases", type=int, default=0, help="keep only the first N init times (0 = all)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import numcodecs.blosc
    import xarray as xr
    numcodecs.blosc.use_threads = False  # keep the CPU free for training jobs

    z = np.load(a.data)
    lat, lon = z["lat"].astype(np.float64), z["lon"].astype(np.float64)
    times = z[f"{a.split}/time"].astype("datetime64[ns]")
    names = list(SPECS[a.spec].state_channels())
    lead, K = a.lead, a.max_lead_steps
    n_pairs = len(times) - lead
    last = n_pairs - 1 - (K - 1) * lead
    cases = list(range(0, last, a.eval_stride))  # == stage_eval / eval_climode init times
    if a.max_cases:
        cases = cases[: a.max_cases]
    inits = times[cases]

    ds = xr.open_zarr(a.zarr, storage_options={"token": "anon"}, chunks=None)  # lazy, no dask
    # some stores carry stray whitespace in variable names (Aurora 2022: "geopotential\t")
    ds = ds.rename({v: v.strip() for v in ds.data_vars if v != v.strip()})
    assert np.allclose(ds.latitude.values, lat) and np.allclose(ds.longitude.values, lon), "grid mismatch"
    have = set(ds.data_vars)
    names = [n for n in names if WB2_SOURCES[n][0] in have and
             (WB2_SOURCES[n][1] is None or WB2_SOURCES[n][1] in ds.level.values)]
    missing = sorted(set(np.asarray(inits, "datetime64[ns]").tolist()) - set(ds.time.values.astype("datetime64[ns]").tolist()))
    assert not missing, f"{len(missing)} init times missing from {a.zarr}, e.g. {missing[:3]}"
    print(f"{a.name}: {len(cases)} init times {str(inits[0])[:13]}..{str(inits[-1])[:13]}, leads "
          f"{[6 * lead * k for k in range(1, K + 1)]} h, variables {names}", flush=True)

    t0 = time.time()
    res = {"n_cases": len(cases), "lead_hours": {}, "rmse": {}, "finite": {a.name: {}}, "source": a.zarr,
           "init_times": [str(t) for t in inits], "notes": __doc__.split("Caveats")[1].strip()}
    fcs, trs = {}, {}
    for k in range(1, K + 1):
        td = np.timedelta64(6 * lead * k, "h")
        if not np.issubdtype(ds.prediction_timedelta.dtype, np.timedelta64):  # stored as plain integer hours
            assert "hour" in str(ds.prediction_timedelta.attrs.get("units", "hours")), ds.prediction_timedelta.attrs
            td = 6 * lead * k
        idx = [i + k * lead for i in cases]
        fcs[k], trs[k] = {}, {}
        for n in names:
            var, level = WB2_SOURCES[n]
            da = ds[var].sel(time=inits, prediction_timedelta=td)
            if level is not None:
                da = da.sel(level=level)
            fcs[k][n] = da.transpose("time", "latitude", "longitude").values.astype(np.float32)
            trs[k][n] = z[f"{a.split}/{n}"][idx].astype(np.float32)
    # init times with a missing (NaN) forecast at any lead or variable, e.g. Keisler 2020-09-25
    ok = np.ones(len(cases), bool)
    for k in fcs:
        for v in fcs[k].values():
            ok &= np.isfinite(v).reshape(len(cases), -1).all(1)
    if not ok.all():
        dropped = [str(t)[:13] for t in inits[~ok]]
        print(f"dropping {len(dropped)} init times with missing forecasts: {dropped}", flush=True)
        res["dropped_init_times"] = dropped
        res["n_cases"] = int(ok.sum())
        res["init_times"] = [str(t) for t in inits[ok]]
    for k in fcs:
        idx = np.asarray([i + k * lead for i in cases])[ok]
        fc = {n: v[ok] for n, v in fcs[k].items()}
        tr = {n: v[ok] for n, v in trs[k].items()}
        res["lead_hours"][k] = 6 * lead * k
        res["finite"][a.name][k] = bool(all(np.isfinite(v).all() for v in fc.values()))
        res["rmse"][k] = {a.name: wb2_scores(fc, tr, lat, lon, times[idx])["rmse"]}
        print(f"lead {6 * lead * k} h: z500 {res['rmse'][k][a.name].get('z500', float('nan')):.1f}", flush=True)
    res["seconds"] = time.time() - t0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "eval.json", "w"), indent=1, default=str)
    print(f"wrote {out / 'eval.json'}")


if __name__ == "__main__":
    main()
