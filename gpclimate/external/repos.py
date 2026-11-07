"""Bridges to the research repositories connected to this project.

The model is assembled from code that already lives in the user's forks
instead of re-implementing it.  This module is the single place that knows
where those forks are and how to import from them without dragging in heavy,
unrelated package side effects.

Resolution order for every repository (first hit wins):

1. an installed distribution (``pip install -e <fork>``),
2. ``$GPC_REPOS_ROOT/<dir>`` (e.g. the sibling clones of a project workspace),
3. ``<this repo>/third_party/<dir>`` (git submodules pinned in ``.gitmodules``),
4. ``<parent of this repo>/<dir>`` (side-by-side checkouts).

Pinned fork commits are listed in ``third_party/README.md``.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import os
import sys
import types
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterator, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# repository key -> checkout directory name
REPO_DIRS: Dict[str, str] = {
    "physicsnemo": "physicsnemo",
    "LieConv": "LieConv",
    "lie-transformer": "lie-transformer",
    "graphtransformer": "graphtransformer",
    "weatherbench2": "weatherbench2",
    "e3nn": "e3nn",
    "pytorch_geometric": "pytorch_geometric",
}


class MissingRepositoryError(ImportError):
    pass


def _candidate_roots() -> Iterator[Path]:
    env = os.environ.get("GPC_REPOS_ROOT")
    if env:
        yield Path(env)
    yield PROJECT_ROOT / "third_party"
    yield PROJECT_ROOT.parent


@lru_cache(maxsize=None)
def repo_path(key: str) -> Optional[Path]:
    """Return the checkout directory of a connected repository, if present."""
    name = REPO_DIRS[key]
    for root in _candidate_roots():
        p = root / name
        if p.is_dir() and any(p.iterdir()):
            return p
    return None


def require_repo(key: str) -> Path:
    p = repo_path(key)
    if p is None:
        raise MissingRepositoryError(
            f"Repository '{key}' not found. Run `git submodule update --init third_party/{REPO_DIRS[key]}` "
            f"or set GPC_REPOS_ROOT to a directory containing a '{REPO_DIRS[key]}' checkout."
        )
    return p


def ensure_importable(key: str, top_level_module: str, subdir: str = "") -> None:
    """Make ``top_level_module`` importable, adding the fork checkout to sys.path if needed."""
    if importlib.util.find_spec(top_level_module) is not None:
        return
    root = require_repo(key) / subdir
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    importlib.invalidate_caches()
    if importlib.util.find_spec(top_level_module) is None:
        raise MissingRepositoryError(f"'{top_level_module}' not importable from {root}")


@contextlib.contextmanager
def _temporary_modules(mods: Dict[str, types.ModuleType]):
    """Insert stub modules for the duration of a single module exec."""
    added = [k for k in mods if k not in sys.modules]
    for k in added:
        sys.modules[k] = mods[k]
    try:
        yield
    finally:
        for k in added:
            sys.modules.pop(k, None)


def _load_file_module(qualname: str, path: Path, stubs: Optional[Dict[str, types.ModuleType]] = None):
    spec = importlib.util.spec_from_file_location(qualname, path)
    if spec is None or spec.loader is None:
        raise MissingRepositoryError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    with _temporary_modules(stubs or {}):
        spec.loader.exec_module(mod)
    sys.modules[qualname] = mod
    return mod


# --------------------------------------------------------------------------- physicsnemo
@lru_cache(maxsize=None)
def physicsnemo_icosahedral_mesh():
    """physicsnemo ``models/graphcast/utils/icosahedral_mesh.py`` (GraphCast mesh hierarchy).

    A full ``import physicsnemo`` pulls in NVIDIA warp, hydra, timm, ... .  The
    mesh module itself only needs numpy + scipy and imports a single helper
    (``check_version_spec``) from ``physicsnemo.core``; when physicsnemo is not
    installed we execute the file from the fork checkout with that one helper
    provided, so the code that runs is the fork's own code.
    """
    try:  # a full physicsnemo install
        from physicsnemo.models.graphcast.utils import icosahedral_mesh  # type: ignore

        return icosahedral_mesh
    except Exception:
        pass
    path = require_repo("physicsnemo") / "physicsnemo/models/graphcast/utils/icosahedral_mesh.py"
    pkg = types.ModuleType("physicsnemo")
    pkg.__path__ = []  # type: ignore[attr-defined]
    core = types.ModuleType("physicsnemo.core")
    core.__path__ = []  # type: ignore[attr-defined]
    vc = types.ModuleType("physicsnemo.core.version_check")

    def check_version_spec(distribution_name, spec="0.0.0", *, error_msg=None, hard_fail=False):
        ok = importlib.util.find_spec(distribution_name) is not None
        if not ok and hard_fail:
            raise ImportError(error_msg or f"{distribution_name} is required")
        return ok

    vc.check_version_spec = check_version_spec  # type: ignore[attr-defined]
    return _load_file_module(
        "gpclimate._ext.physicsnemo_icosahedral_mesh",
        path,
        {"physicsnemo": pkg, "physicsnemo.core": core, "physicsnemo.core.version_check": vc},
    )


@lru_cache(maxsize=None)
def physicsnemo_graph_utils():
    """physicsnemo ``models/graphcast/utils/graph_utils.py`` (lat/lon <-> xyz helpers; numpy+torch only)."""
    try:
        from physicsnemo.models.graphcast.utils import graph_utils  # type: ignore

        return graph_utils
    except Exception:
        pass
    path = require_repo("physicsnemo") / "physicsnemo/models/graphcast/utils/graph_utils.py"
    return _load_file_module("gpclimate._ext.physicsnemo_graph_utils", path)


# --------------------------------------------------------------------------- LieConv / LieTransformer
@lru_cache(maxsize=None)
def lieconv_SO2():
    """``lie_conv.lieGroups.SO2`` from the LieConv fork (exp/log/lift for the gauge group)."""
    ensure_importable("LieConv", "lie_conv")
    from lie_conv.lieGroups import SO2  # type: ignore

    return SO2


@lru_cache(maxsize=None)
def lieconv_Swish():
    ensure_importable("LieConv", "lie_conv")
    from lie_conv.lieConv import Swish  # type: ignore

    return Swish


@lru_cache(maxsize=None)
def lietransformer_MultiheadLinear():
    """``eqv_transformer.multihead_neural.MultiheadLinear`` (per-head weights, LieTransformer)."""
    ensure_importable("LieConv", "lie_conv")
    ensure_importable("lie-transformer", "eqv_transformer")
    from eqv_transformer.multihead_neural import MultiheadLinear  # type: ignore

    return MultiheadLinear


# --------------------------------------------------------------------------- WeatherBench2
@lru_cache(maxsize=None)
def weatherbench2_metrics():
    """``weatherbench2.metrics`` (needs xarray + scipy; apache-beam is NOT needed for metrics)."""
    ensure_importable("weatherbench2", "weatherbench2")
    from weatherbench2 import metrics  # type: ignore

    return metrics


def available() -> Dict[str, Optional[str]]:
    """Which connected repositories were found (for logs / smoke-test report)."""
    out: Dict[str, Optional[str]] = {}
    for key in REPO_DIRS:
        p = repo_path(key)
        out[key] = str(p) if p else None
    return out
