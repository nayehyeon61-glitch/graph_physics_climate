#!/usr/bin/env bash
# Set up the connected repositories and Python dependencies for gpclimate.
#
# Path-imported forks (LieConv, lie-transformer, physicsnemo mesh utils,
# weatherbench2 metrics, graphtransformer reference) are git submodules pinned
# in .gitmodules; e3nn and pytorch_geometric are pip-installed from the forks.
# Alternatively point GPC_REPOS_ROOT at a directory with side-by-side clones.
set -euo pipefail
cd "$(dirname "$0")/.."
git submodule update --init --depth 1 third_party/LieConv third_party/lie-transformer \
  third_party/weatherbench2 third_party/graphtransformer
# physicsnemo is large (~400 MB); a shallow, blob-filtered checkout is enough for the mesh utilities
git submodule update --init --depth 1 --filter=blob:none third_party/physicsnemo
pip install -e ".[data,test]"
