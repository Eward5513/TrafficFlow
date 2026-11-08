"""Node2vec on the static directed R graph.

The reference reads a precomputed file and does not state the dimension.
There is no Xuancheng node2vec asset and no header to copy, so this uses 128
dimensions. Walks follow outgoing R edges. ``p=q=1`` makes every neighbor
weight 1, which is a uniform out-walk. Nodes with no outgoing edge stay put
so they still receive an embedding. The embedding is frozen when ASTTN trains.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

NODE2VEC_SETTINGS = {
    "dimensions": 128,
    "walk_length": 80,
    "num_walks": 10,
    "window": 10,
    "p": 1.0,
    "q": 1.0,
    "negative": 5,
    "epochs": 2,
    "grad_clip_norm": 5.0,
    "loss_reduction": "sum",
    "learning_rate": 0.025,
    "min_learning_rate": 0.0001,
    "seed": 42,
    "graph": "directed R adjacency, walk along outgoing edges",
    "subsampling": "none",
    "min_count": 0,
    "embedding_table": "skip-gram input vectors",
}


def out_neighbors(adjacency: np.ndarray) -> list[np.ndarray]:
    matrix = np.asarray(adjacency)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency must be square")
    neighbors = []
    for node in range(int(matrix.shape[0])):
        neighbors.append(np.flatnonzero(matrix[node] > 0).astype(np.int32, copy=False))
    return neighbors


def random_walks(neighbors: list[np.ndarray], *, num_walks: int, walk_length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    nodes = len(neighbors)
    walks = np.empty((int(num_walks) * nodes, int(walk_length)), dtype=np.int32)
    row = 0
    for _ in range(int(num_walks)):
        for start in rng.permutation(nodes):
            current = int(start)
            walks[row, 0] = current
            for step in range(1, int(walk_length)):
                pool = neighbors[current]
                current = current if pool.size == 0 else int(pool[int(rng.integers(pool.size))])
                walks[row, step] = current
            row += 1
    return walks


def context_pairs(walks: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    centers = []
    contexts = []
    for offset in range(1, int(window) + 1):
        if offset >= walks.shape[1]:
            break
        left = walks[:, :-offset].reshape(-1)
        right = walks[:, offset:].reshape(-1)
        centers.append(left)
        contexts.append(right)
        centers.append(right)
        contexts.append(left)
    return np.concatenate(centers), np.concatenate(contexts)


def train_skipgram(walks: np.ndarray, *, settings: dict | None = None) -> np.ndarray:
    cfg = dict(NODE2VEC_SETTINGS if settings is None else settings)
    nodes = int(walks.max()) + 1
    if int(walks.min()) < 0 or int(np.unique(walks).size) != nodes:
        present = np.unique(walks)
        if int(present.size) != nodes or int(present[0]) != 0 or int(present[-1]) != nodes - 1:
            raise ValueError("walks must cover every node index exactly in 0..N-1")
    centers, contexts = context_pairs(walks, int(cfg["window"]))
    counts = np.bincount(walks.ravel(), minlength=nodes).astype(np.float64)
    if np.any(counts <= 0):
        raise ValueError("a node never appears in the node2vec walks")
    probabilities = np.power(counts, 0.75)
    probabilities /= probabilities.sum()
    generator = torch.Generator()
    generator.manual_seed(int(cfg["seed"]))
    dim = int(cfg["dimensions"])
    center = torch.empty(nodes, dim)
    context = torch.empty(nodes, dim)
    bound = 0.5 / float(dim)
    center.uniform_(-bound, bound, generator=generator)
    context.uniform_(-bound, bound, generator=generator)
    center.requires_grad_(True)
    context.requires_grad_(True)
    noise = torch.as_tensor(probabilities, dtype=torch.float32)
    order = torch.randperm(int(centers.shape[0]), generator=generator).numpy()
    batch_size = 8192
    negative = int(cfg["negative"])
    steps = int(np.ceil(order.size / batch_size)) * int(cfg["epochs"])
    step_id = 0
    start_lr = float(cfg["learning_rate"])
    end_lr = float(cfg["min_learning_rate"])
    for _ in range(int(cfg["epochs"])):
        for start in range(0, int(order.size), batch_size):
            chosen = order[start : start + batch_size]
            center_ids = torch.as_tensor(centers[chosen], dtype=torch.long)
            context_ids = torch.as_tensor(contexts[chosen], dtype=torch.long)
            negatives = torch.multinomial(noise, int(center_ids.shape[0]) * negative, replacement=True, generator=generator)
            negatives = negatives.view(int(center_ids.shape[0]), negative)
            center_vec = center.index_select(0, center_ids)
            context_vec = context.index_select(0, context_ids)
            negative_vec = context.index_select(0, negatives.reshape(-1)).view(int(center_ids.shape[0]), negative, dim)
            positive = (center_vec * context_vec).sum(dim=-1)
            negative_score = torch.einsum("bd,bnd->bn", center_vec, negative_vec)
            # Sum, not mean: word2vec applies the learning rate to each pair.
            pair_loss = -(torch.nn.functional.logsigmoid(positive) + F.logsigmoid(-negative_score).sum(dim=-1))
            loss = pair_loss.sum()
            loss.backward()
            progress = step_id / max(steps, 1)
            learning_rate = start_lr + (end_lr - start_lr) * progress
            torch.nn.utils.clip_grad_norm_((center, context), float(cfg["grad_clip_norm"]))
            with torch.no_grad():
                center.data.add_(center.grad, alpha=-learning_rate)
                context.data.add_(context.grad, alpha=-learning_rate)
            center.grad = None
            context.grad = None
            step_id += 1
            if step_id == 1 or step_id % 500 == 0 or step_id == steps:
                print(
                    f"[node2vec] step {step_id}/{steps} lr={learning_rate:.6f} mean_pair_loss={float(pair_loss.mean().detach()):.4f}",
                    flush=True,
                )
    embedding = center.detach().to(dtype=torch.float32).contiguous().numpy()
    if not np.isfinite(embedding).all():
        raise ValueError("node2vec produced a non-finite embedding")
    return embedding
