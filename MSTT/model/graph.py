"""Edge lists for sparse spatial attention. Pure tensors, no file IO.

Direction convention (matches ``r_graph.npz``): ``adjacency[u, v] = 1`` means
vehicles leave road ``u`` and enter road ``v`` (``u`` is upstream of ``v``).

An attention edge ``(src, dst)`` sends ``k/v`` from ``src`` to the query at
``dst``; softmax normalizes over all edges that share one ``dst``.

- ``forward``: ``dst`` attends to its upstream roads (``adjacency[src, dst]``).
- ``backward``: ``dst`` attends to its downstream roads (``adjacency[dst, src]``).
- ``bidirectional``: union of both; ``edge_type`` keeps which relation holds.

Every node gets one self-loop. Edges are sorted by ``(dst, src)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

EDGE_SELF = 0
EDGE_UPSTREAM = 1
EDGE_DOWNSTREAM = 2
EDGE_BOTH = 3
NUM_EDGE_TYPES = 4
DIRECTIONS = ("forward", "backward", "bidirectional")


@dataclass(frozen=True)
class SpatialGraph:
    src: torch.Tensor
    dst: torch.Tensor
    edge_type: torch.Tensor
    num_nodes: int
    direction: str
    num_edge_types: int = NUM_EDGE_TYPES

    @property
    def num_edges(self) -> int:
        return int(self.src.numel())

    def in_degree(self) -> torch.Tensor:
        return torch.bincount(self.dst, minlength=self.num_nodes)


def build_spatial_graph(adjacency: np.ndarray | torch.Tensor, direction: str = "bidirectional") -> SpatialGraph:
    if direction not in DIRECTIONS:
        raise ValueError(f"direction {direction!r} not in {DIRECTIONS}")
    matrix = adjacency.detach().cpu().numpy() if torch.is_tensor(adjacency) else np.asarray(adjacency)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"adjacency must be square, got {matrix.shape}")
    if not np.isin(np.unique(matrix), (0, 1)).all():
        raise ValueError("adjacency must be 0/1 connectivity")
    link = matrix.astype(bool)
    np.fill_diagonal(link, False)
    upstream = link
    downstream = link.T
    if direction == "forward":
        keep = upstream
    elif direction == "backward":
        keep = downstream
    else:
        keep = upstream | downstream
    # Row = dst, column = src for the relation matrices below.
    keep_t = keep.T
    up_t = upstream.T & keep_t
    down_t = downstream.T & keep_t
    dst, src = np.nonzero(keep_t)
    kind = np.where(up_t[dst, src] & down_t[dst, src], EDGE_BOTH, np.where(up_t[dst, src], EDGE_UPSTREAM, EDGE_DOWNSTREAM))
    nodes = np.arange(matrix.shape[0])
    dst = np.concatenate((dst, nodes))
    src = np.concatenate((src, nodes))
    kind = np.concatenate((kind, np.full(nodes.shape, EDGE_SELF)))
    order = np.lexsort((src, dst))
    return SpatialGraph(
        src=torch.as_tensor(src[order], dtype=torch.long),
        dst=torch.as_tensor(dst[order], dtype=torch.long),
        edge_type=torch.as_tensor(kind[order], dtype=torch.long),
        num_nodes=int(matrix.shape[0]),
        direction=direction,
    )
