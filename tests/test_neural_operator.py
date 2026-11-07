"""stage13: continuous neural-operator layer (neuraloperator FNO blocks on the lat-lon grid) after the mesh V-Net,
and the spectral energy-continuity loss: no-op at init, gauge equivariance of the whole forecaster with the layer,
the loss' zero/positivity, and a Stage B smoke run with both switched on."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpclimate.data.variables import VariableSpec  # noqa: E402
from gpclimate.geometry.graphs import SphereGraphs  # noqa: E402
from gpclimate.models.dynamics_vnet import LieGuidedForecaster, VNetConfig  # noqa: E402
from gpclimate.models.lie_physical_gnn import GuideConfig  # noqa: E402
from gpclimate.nn.neural_operator import ContinuousOperatorBlock, field_to_image, image_to_field  # noqa: E402
from gpclimate.nn.so2 import Field  # noqa: E402
from gpclimate.training.losses import (gauge_equivariance_report, relative_error, spectral_energy_loss,  # noqa: E402
                                       zonal_band_power)

torch.manual_seed(0)
SPEC = VariableSpec(scalars=("z500", "t850"), vectors=(("u850", "v850"), ("u10", "v10")),
                    static_scalars=("orog",), static_vectors=(("dox", "doy"),), forcing_scalars=("toa",))
LAT, LON = np.linspace(-82.5, 82.5, 12), np.arange(24) * 15.0


@pytest.fixture(scope="module")
def graphs():
    return SphereGraphs(LAT, LON, mesh_splits=(2, 1, 0))


def _rand(B, N, cs, cv):
    return Field(torch.randn(B, N, cs), torch.complex(torch.randn(B, N, cv), torch.randn(B, N, cv)))


def test_image_roundtrip_and_zero_init(graphs):
    g = graphs.view()
    H, W = g.extras["grid_hw"]
    assert (H, W) == (len(LAT), len(LON)) and g.n_grid == H * W
    f = _rand(2, H * W, 3, 2)
    back = image_to_field(field_to_image(f, H, W), 3, 2)
    assert relative_error(back, f) < 1e-7
    x = _rand(2, H * W, SPEC.n_s + SPEC.n_cond_s, SPEC.n_v + SPEC.n_cond_v)
    d = _rand(2, H * W, SPEC.n_s, SPEC.n_v)
    for conv in ("fft", "sht"):
        if conv == "sht":
            pytest.importorskip("torch_harmonics")
        blk = ContinuousOperatorBlock(SPEC.n_s, SPEC.n_v, SPEC.n_cond_s, SPEC.n_cond_v, hidden=8, layers=2,
                                      modes=(6, 12), conv=conv)
        with torch.no_grad():
            out = blk(x, d, H, W, g.extras["grid_lat"])
        assert out.s.shape == d.s.shape and out.v.shape == d.v.shape
        assert float(out.s.abs().max()) == 0.0 and float(out.v.abs().max()) == 0.0
        torch.nn.init.normal_(blk.proj[-1].weight, std=0.1)  # active block: finite output of the right shape
        out = blk(x, d, H, W, g.extras["grid_lat"])
        assert torch.isfinite(out.s).all() and torch.isfinite(out.v).all() and float(out.s.abs().max()) > 0


def test_forecaster_with_neural_operator_is_gauge_equivariant(graphs):
    model = LieGuidedForecaster(SPEC, GuideConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), hidden=16),
                                VNetConfig(cs=8, cv=4, depth=(1, 1, 1, 1, 1), heads=2, ds=4, dv=2, hidden=16,
                                           neural_op="fft", no_hidden=8, no_layers=2, no_modes=(6, 12)))
    for p in list(model.vnet.head_out.parameters()) + list(model.vnet.cont_op.proj[-1].parameters()):
        torch.nn.init.normal_(p, std=0.1)
    G = graphs.grid.n
    x = _rand(2, G, SPEC.n_s, SPEC.n_v)
    c = _rand(1, G, SPEC.n_cond_s, SPEC.n_cond_v)

    def fn(xx, cc, gv):
        f, d = model(xx, cc, gv)
        return {"forecast": f, "delta": d}

    rep = gauge_equivariance_report(fn, x, c, graphs, {"forecast": "grid", "delta": "grid"})
    assert max(rep.values()) < 2e-5, rep
    # the operator branch is active (output differs from the model without it)
    with torch.no_grad():
        f1, _ = model(x, c, graphs.view())
        model.vnet.cont_op = None
        f0, _ = model(x, c, graphs.view())
    assert relative_error(f1, f0) > 1e-4


def test_spectral_energy_loss_zero_and_positive(graphs):
    g = graphs.view()
    H, W = g.extras["grid_hw"]
    lat = g.extras["grid_lat"]
    f = _rand(3, H * W, 2, 1)
    assert float(spectral_energy_loss(f, f, H, W, lat)) == 0.0
    # a zonally smoothed copy loses small-scale power -> positive loss, with the band powers falling at high k
    img = field_to_image(f, H, W)
    k = torch.ones(1, 1, 1, 5) / 5
    sm = torch.nn.functional.conv2d(torch.nn.functional.pad(img, (2, 2, 0, 0), mode="circular"),
                                    k.expand(img.shape[1], 1, 1, 5), groups=img.shape[1])
    fs = image_to_field(sm, 2, 1)
    assert float(spectral_energy_loss(fs, f, H, W, lat)) > 0.05
    p_true, p_sm = zonal_band_power(f, H, W, lat), zonal_band_power(fs, H, W, lat)
    assert torch.all(p_sm[:, -1] < p_true[:, -1])


def test_stage_b_with_neural_operator_and_spectral_loss(tmp_path):
    from test_train_eval_smoke import _run, _write_data
    data = tmp_path / "synthetic.npz"
    _write_data(data)
    run_dir = tmp_path / "run_no"
    out = _run(run_dir, data, "b_noguide", "--steps", "3", "--val-every", "2", "--neural-op", "fft", "--no-hidden", "8",
               "--no-layers", "1", "--no-modes", "6", "12", "--spec-energy-weight", "0.1")
    assert "neural operator fft" in out and "spectral energy loss weight 0.1" in out
    cfg = json.load(open(run_dir / "config.json"))
    assert cfg["neural_op"] == "fft" and cfg["no_modes"] == [6, 12]
    rows = [json.loads(l) for l in open(run_dir / "log_vnet_noguide.jsonl")]
    assert rows[0]["l_spec"] > 0 and np.isfinite(rows[-1]["val_mse"])
    sd = torch.load(run_dir / "vnet_noguide.pt")
    assert any(k.startswith("cont_op.") for k in sd)
