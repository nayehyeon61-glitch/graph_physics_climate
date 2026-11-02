"""Evaluation through WeatherBench2 metrics (area-weighted RMSE, CRPS, ensemble-mean RMSE, spread)."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from gpclimate.external import repos


def _to_dataset(fields: Dict[str, np.ndarray], lat: np.ndarray, lon: np.ndarray, times: np.ndarray,
                member_dim: bool = False):
    import xarray as xr

    dims = ("number", "time", "latitude", "longitude") if member_dim else ("time", "latitude", "longitude")
    coords = {"time": times.astype("datetime64[ns]"), "latitude": lat, "longitude": lon}
    data = {k: (dims, v.astype(np.float64)) for k, v in fields.items()}
    if member_dim:
        coords["number"] = np.arange(next(iter(fields.values())).shape[0])
    return xr.Dataset(data, coords=coords)


def wb2_scores(forecast: Dict[str, np.ndarray], truth: Dict[str, np.ndarray], lat: np.ndarray, lon: np.ndarray,
               times: np.ndarray, ensemble: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Dict[str, float]]:
    """forecast/truth: name -> [T, lat, lon] (physical units); ensemble: name -> [M, T, lat, lon]."""
    metrics = repos.weatherbench2_metrics()
    fc = _to_dataset(forecast, lat, lon, times)
    tr = _to_dataset(truth, lat, lon, times)
    out: Dict[str, Dict[str, float]] = {}
    rmse = metrics.RMSESqrtBeforeTimeAvg().compute(fc, tr)
    out["rmse"] = {k: float(rmse[k]) for k in forecast}
    if ensemble is not None:
        en = _to_dataset(ensemble, lat, lon, times, member_dim=True)
        crps = metrics.CRPS(ensemble_dim="number").compute(en, tr)
        emr = metrics.EnsembleMeanRMSESqrtBeforeTimeAvg(ensemble_dim="number").compute(en, tr)
        spr = metrics.EnsembleStddevSqrtBeforeTimeAvg(ensemble_dim="number").compute(en, tr)
        out["crps"] = {k: float(crps[k]) for k in ensemble}
        out["ens_mean_rmse"] = {k: float(emr[k]) for k in ensemble}
        out["ens_spread"] = {k: float(spr[k]) for k in ensemble}
    return out
