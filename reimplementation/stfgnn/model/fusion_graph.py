"""4-step fusion adjacency from ``reference/STFGNN/utils_4n0_3layer_12T_res.py``.

Official ``construct_adj_fusion`` comment (4N_1 mode)::

    [T, 1, 1, T
     1, S, 1, 1
     1, 1, S, 1
     T, 1, 1, T]

S = spatial connectivity, T = DTW temporal adjacency.
``1`` on off-diagonal time blocks is same-node identity copied from the
(t0, t1) adjacent-time block, except (t0, t3)/(t3, t0) which are ``A_dtw``.
"""

from __future__ import annotations

import numpy as np
import torch

FUSION_STEPS = 4


def construct_adj_fusion(
    spatial_adj: np.ndarray,
    temporal_adj: np.ndarray,
    steps: int = FUSION_STEPS,
) -> np.ndarray:
    """Line-by-line port of original ``construct_adj_fusion(A, A_dtw, steps)``."""
    if steps != FUSION_STEPS:
        raise ValueError(f"official 4n0 fusion uses steps=4, got {steps}")
    spatial = np.array(spatial_adj, dtype=np.float32, copy=True)
    temporal = np.array(temporal_adj, dtype=np.float32, copy=True)
    if spatial.shape != temporal.shape or spatial.ndim != 2 or spatial.shape[0] != spatial.shape[1]:
        raise ValueError(
            f"spatial {spatial.shape} and temporal {temporal.shape} must be square and equal"
        )
    n_nodes = int(spatial.shape[0])
    localized = np.zeros((n_nodes * steps, n_nodes * steps), dtype=np.float32)
    for index in range(steps):
        start = index * n_nodes
        end = (index + 1) * n_nodes
        if index in (1, 2):
            localized[start:end, start:end] = spatial
        else:
            localized[start:end, start:end] = temporal
    for node in range(n_nodes):
        for lag in range(steps - 1):
            src = lag * n_nodes + node
            dst = (lag + 1) * n_nodes + node
            localized[src, dst] = 1.0
            localized[dst, src] = 1.0
    localized[3 * n_nodes : 4 * n_nodes, 0:n_nodes] = temporal
    localized[0:n_nodes, 3 * n_nodes : 4 * n_nodes] = temporal
    adjacent_block = localized[0:n_nodes, n_nodes : 2 * n_nodes]
    localized[2 * n_nodes : 3 * n_nodes, 0:n_nodes] = adjacent_block
    localized[0:n_nodes, 2 * n_nodes : 3 * n_nodes] = adjacent_block
    localized[n_nodes : 2 * n_nodes, 3 * n_nodes : 4 * n_nodes] = adjacent_block
    localized[3 * n_nodes : 4 * n_nodes, n_nodes : 2 * n_nodes] = adjacent_block
    for index in range(localized.shape[0]):
        localized[index, index] = 1.0
    return localized


def mask_init_from_fusion(localized_adj: np.ndarray) -> np.ndarray:
    return (np.asarray(localized_adj) != 0).astype(np.float32)


def flatten_time_major_window(window: torch.Tensor) -> torch.Tensor:
    """``(B, 4, N, C) -> (4N, B, C)``."""
    if window.ndim != 4:
        raise ValueError(f"window must be (B, 4, N, C), got {tuple(window.shape)}")
    batch, steps, n_nodes, channels = window.shape
    time_major = window.reshape(batch, steps * n_nodes, channels)
    return time_major.transpose(0, 1).contiguous()


def second_timestep_slice(localized: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """Official STFGCM crop: nodes ``[N:2N]``, the second step of the 4-step window."""
    return localized[int(n_nodes) : 2 * int(n_nodes)]


def time_length_after_stfgcl(seq_len: int, n_layers: int) -> int:
    """Each fusion layer shortens time by 3 (4-step window, stride 1, no pad)."""
    return int(seq_len) - 3 * int(n_layers)


def make_test_only_temporal_adjacency(n_nodes: int, extra_pairs: tuple[tuple[int, int], ...] = ()) -> np.ndarray:
    """Deterministic 0/1 matrix for unit/smoke tests. Not an official DTW graph."""
    adj = np.eye(n_nodes, dtype=np.float32)
    for src, dst in extra_pairs:
        adj[src, dst] = 1.0
        adj[dst, src] = 1.0
    return adj
