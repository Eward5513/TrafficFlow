"""Localized 3-step spatial-temporal adjacency matching original STSGCN ``construct_adj``.

Original: ``reference/STSGCN/utils.py::construct_adj`` and
``get_adjacency_matrix(..., type_='connectivity')``.
"""

from __future__ import annotations

import numpy as np
import torch

LOCAL_STEPS = 3


def construct_localized_adjacency(spatial_adj: np.ndarray, steps: int = LOCAL_STEPS) -> np.ndarray:
    """Build the ``(steps * N, steps * N)`` local ST graph.

    Node order is time-major::

        [t=0's N nodes, t=1's N nodes, ..., t=steps-1's N nodes]

    Spatial blocks on the diagonal copy ``A``. Adjacent time steps of the
    *same* spatial node are linked bidirectionally. There is no skip from
    t-1 to t+1. Self-loops are then set on every localized node. The matrix
    is **not** degree-normalized.

    The input ``spatial_adj`` is never modified.
    """
    if spatial_adj.ndim != 2 or spatial_adj.shape[0] != spatial_adj.shape[1]:
        raise ValueError(f"spatial adjacency must be square, got {spatial_adj.shape}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    spatial = np.array(spatial_adj, dtype=np.float32, copy=True)
    n_nodes = int(spatial.shape[0])
    localized = np.zeros((n_nodes * steps, n_nodes * steps), dtype=np.float32)
    for index in range(steps):
        start = index * n_nodes
        end = (index + 1) * n_nodes
        localized[start:end, start:end] = spatial
    for node in range(n_nodes):
        for lag in range(steps - 1):
            src = lag * n_nodes + node
            dst = (lag + 1) * n_nodes + node
            localized[src, dst] = 1.0
            localized[dst, src] = 1.0
    for index in range(localized.shape[0]):
        localized[index, index] = 1.0
    return localized


def mask_init_from_localized_adjacency(localized_adj: np.ndarray) -> np.ndarray:
    """Original ``(adj_mx != 0).astype(float32)`` Constant initializer."""
    return (np.asarray(localized_adj) != 0).astype(np.float32)


def flatten_time_major_window(window: torch.Tensor) -> torch.Tensor:
    """``(B, 3, N, C) -> (3N, B, C)`` matching original reshape then transpose.

    Original::

        t = reshape(t, (-1, 3 * N, C))   # (B, 3N, C), time-major
        t = transpose(t, (1, 0, 2))      # (3N, B, C)
    """
    if window.ndim != 4:
        raise ValueError(f"window must be (B, 3, N, C), got {tuple(window.shape)}")
    batch, steps, n_nodes, channels = window.shape
    time_major = window.reshape(batch, steps * n_nodes, channels)
    return time_major.transpose(0, 1).contiguous()


def center_slice(localized_nodes: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """Crop the middle time step from ``(3N, B, C)``: nodes ``[N:2N]``."""
    return localized_nodes[n_nodes : 2 * n_nodes]


def time_length_after_stsgcl(seq_len: int, n_layers: int) -> int:
    """Each STSGCL shortens the time axis by 2 with no padding."""
    return int(seq_len) - 2 * int(n_layers)
