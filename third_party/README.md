# Connected repositories (pinned forks)

| Repository (fork) | Pinned commit | How gpclimate uses it |
|---|---|---|
| `nayehyeon61-glitch/physicsnemo` | `097f4b2` | `models/graphcast/utils/icosahedral_mesh.py` (icosahedral hierarchy, `faces_to_edges`) and `graph_utils.py` (`latlon2xyz`, `xyz2latlon`), executed from the fork source without importing the heavy package `__init__` |
| `nayehyeon61-glitch/LieConv` | `93affb1` | `lie_conv.lieGroups.SO2` (`log` of the per-edge transport matrix, `lifted_elems` for the edge direction), `lie_conv.lieConv.Swish` |
| `nayehyeon61-glitch/lie-transformer` | `81a0aa3` | `eqv_transformer.multihead_neural.MultiheadLinear` for the location kernel of the physics cross-attention (LieTransformer "mlp" location featurisation, "sum" combination) |
| `nayehyeon61-glitch/graphtransformer` | `c9cd493` | Reference for the attention layer structure (Q/K/V/E, edge-modulated scores, clamp, O, residual+norm+FFN). The original is DGL-based and is **not** imported; the structure is re-expressed with PyG |
| `nayehyeon61-glitch/weatherbench2` | `1f1ff39` | `weatherbench2.metrics` (RMSE, CRPS, ensemble-mean RMSE, spread) and the public ERA5 64x32 zarr used in the smoke test |
| `nayehyeon61-glitch/e3nn` | `2aa7f58` | pip dependency: `e3nn.math.soft_one_hot_linspace` (radial basis), `e3nn.nn.FullyConnectedNet` (radial weights) |
| `nayehyeon61-glitch/pytorch_geometric` | `ea2c5f6` | pip dependency: `MessagePassing`, `MLP`, `utils.softmax`, `utils.scatter` |
| `nayehyeon61-glitch/neuralgcm`, `nayehyeon61-glitch/dinosaur` | - | Not used yet: both are JAX libraries; candidates for spectral physics diagnostics / residuals in a later step |

Resolution order at import time is documented in `gpclimate/external/repos.py`
(installed package -> `$GPC_REPOS_ROOT/<repo>` -> `third_party/<repo>` -> sibling checkout).
