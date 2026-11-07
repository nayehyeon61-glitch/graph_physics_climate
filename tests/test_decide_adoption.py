"""Adoption rule used by the stage9b merge chain: >= 3 of 5 leads better and a minimum mean gain."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from decide_adoption import decide  # noqa: E402


def _ev(scale):
    return {"rmse": {str(k): {"lie_guided": {"z500": 100.0 * k * s, "t850": 1.0 * k * s}}
                     for k, s in zip(range(1, 6), scale)}}


def test_decide_adoption():
    base = _ev([1, 1, 1, 1, 1])
    good = decide(_ev([0.9, 0.95, 0.97, 0.99, 1.01]), base, min_mean_gain=0.5)
    assert good["adopt"] and good["improved_leads"] == 4 and good["gain_pct"]["1"] == pytest.approx(10.0)
    noise = decide(_ev([0.998, 0.998, 0.999, 1.001, 1.004]), base, min_mean_gain=0.5)
    assert noise["improved_leads"] == 3 and not noise["adopt"]  # stage14b-like: within noise
    assert not decide(_ev([1.02, 0.99, 1.01, 1.0, 1.03]), base)["adopt"]
    with pytest.raises(ValueError):
        decide({**_ev([1] * 5), "init_times": ["a"]}, {**base, "init_times": ["b"]})
