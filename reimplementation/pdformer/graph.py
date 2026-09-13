"""PDFormer geographic hops, DTW semantic relations, and Laplacian PE.

Hop construction follows ``PDFormerDataset._load_rel`` with
``type_short_path='hop'``. Geographic / semantic masks follow
``PDFormer.__init__``. Laplacian PE follows ``PDFormerExecutor._cal_lape``.

FastDTW is a NumPy port of ``fastdtw`` 0.3.4 (radius window, coarsen-and-expand).
It is not Euclidean distance and not exact DTW.

True in geo/semantic masks means **mask out** (``masked_fill_(-inf)``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from reimplementation.common.errors import ReimplementationError
from reimplementation.common.utils.hashing import sha256_file

UNREACHABLE_HOP = 511
FASTDTW_RADIUS = 6
GEO_MASK_TRUE_MEANS = "mask_out"
SEM_MASK_TRUE_MEANS = "mask_out"
TEST_ONLY_RELATIONS = "test_only_pdformer_relations"


def _as_square(matrix: np.ndarray, *, name: str) -> np.ndarray:
    array = np.array(matrix, copy=True)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ReimplementationError(f"{name} must be square, got {array.shape}")
    return array


def assert_binary_topology(matrix: np.ndarray, *, name: str = "topology") -> np.ndarray:
    """Refuse Gaussian / distance weights as hop inputs."""
    topology = _as_square(np.asarray(matrix, dtype=np.float64), name=name)
    if not np.isfinite(topology).all():
        raise ReimplementationError(f"{name} contains NaN or inf")
    unique = {float(value) for value in np.unique(topology)}
    if not unique.issubset({0.0, 1.0}):
        raise ReimplementationError(
            f"{name} is not 0/1 topology; PDFormer hop count cannot use Gaussian "
            "or metre-scale weights as shortest-path distances"
        )
    return topology.astype(np.float64)


def hop_shortest_path(
    binary_adj: np.ndarray,
    *,
    unreachable: int = UNREACHABLE_HOP,
    bidir: bool = True,
) -> np.ndarray:
    """Floyd–Warshall hop matrix matching original ``PDFormerDataset._load_rel``.

    Original::

        sh_mx[sh_mx > 0] = 1
        sh_mx[sh_mx == 0] = 511
        diag = 0
        sh[i, j] = min(sh[i, j], sh[i, k] + sh[k, j], 511)

    ``bidir=True`` (PeMS04/08) symmetrizes the 0/1 topology first. The input is
    copied and never modified.
    """
    original = np.array(binary_adj, copy=True)
    topology = assert_binary_topology(binary_adj, name="hop adjacency")
    if bool(bidir):
        topology = np.maximum(topology, topology.T)
    node_count = int(topology.shape[0])
    hops = np.full((node_count, node_count), float(unreachable), dtype=np.float64)
    hops[topology > 0] = 1.0
    np.fill_diagonal(hops, 0.0)
    for k in range(node_count):
        hops = np.minimum(hops, hops[:, k : k + 1] + hops[k : k + 1, :])
        hops = np.minimum(hops, float(unreachable))
    if not np.array_equal(original, binary_adj):
        raise ReimplementationError("hop_shortest_path modified the input adjacency")
    return hops


def geographic_mask(
    hop_matrix: np.ndarray,
    *,
    far_mask_delta: int,
    transpose: bool = True,
) -> np.ndarray:
    """Boolean geo mask. ``True`` means mask out.

    Original::

        sh_mx = sh_mx.T
        geo_mask[sh_mx >= far_mask_delta] = 1
        geo_mask = geo_mask.bool()

    The inequality is ``>=``, so hop == ``far_mask_delta`` is masked.
    Unreachable 511 is therefore masked when ``far_mask_delta`` is finite.
    """
    hops = _as_square(np.asarray(hop_matrix, dtype=np.float64), name="hop_matrix")
    if bool(transpose):
        hops = hops.T
    mask = hops >= float(far_mask_delta)
    allowed = (~mask).sum(axis=1)
    if np.any(allowed <= 0):
        blocked = np.where(allowed <= 0)[0].tolist()
        raise ReimplementationError(
            f"geographic mask fully blocks rows {blocked}; original softmax would be NaN. "
            "Self-hop is 0, so this usually means far_mask_delta <= 0"
        )
    return mask


def _difference(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.abs(left - right))


def _euclidean(left: np.ndarray, right: np.ndarray) -> float:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.linalg.norm(np.atleast_1d(delta), ord=2))


def _prep_series(series: np.ndarray) -> np.ndarray:
    array = np.asarray(series, dtype=np.float64)
    if array.ndim == 0:
        raise ReimplementationError("DTW series must have a time axis")
    return array


def _dtw_windowed(
    x: np.ndarray,
    y: np.ndarray,
    window: list[tuple[int, int]] | None,
    dist,
) -> tuple[float, list[tuple[int, int]]]:
    len_x = int(x.shape[0])
    len_y = int(y.shape[0])
    if window is None:
        pairs = [(i, j) for i in range(len_x) for j in range(len_y)]
    else:
        pairs = window
    costs = {(0, 0): 0.0}
    prev: dict[tuple[int, int], tuple[int, int]] = {}
    inf = float("inf")
    for i, j in pairs:
        dt = dist(x[i], y[j])
        candidates = (
            (costs.get((i, j + 1), inf), (i, j + 1)),
            (costs.get((i + 1, j), inf), (i + 1, j)),
            (costs.get((i, j), inf), (i, j)),
        )
        best_cost, best_prev = min(candidates, key=lambda item: item[0])
        costs[(i + 1, j + 1)] = best_cost + dt
        prev[(i + 1, j + 1)] = best_prev
    end = (len_x, len_y)
    if end not in costs:
        raise ReimplementationError("DTW window does not cover the series end")
    path: list[tuple[int, int]] = []
    cursor = end
    while cursor != (0, 0):
        path.append((cursor[0] - 1, cursor[1] - 1))
        cursor = prev[cursor]
    path.reverse()
    return float(costs[end]), path


def _reduce_by_half(series: np.ndarray) -> np.ndarray:
    length = int(series.shape[0]) - int(series.shape[0]) % 2
    if length < 2:
        return series[:0]
    left = series[:length:2]
    right = series[1:length:2]
    return (left + right) / 2.0


def _expand_window(
    path: Sequence[tuple[int, int]],
    len_x: int,
    len_y: int,
    radius: int,
) -> list[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set(path)
    for i, j in path:
        for da in range(-radius, radius + 1):
            for db in range(-radius, radius + 1):
                expanded.add((i + da, j + db))
    window_cells: set[tuple[int, int]] = set()
    for i, j in expanded:
        for a, b in (
            (i * 2, j * 2),
            (i * 2, j * 2 + 1),
            (i * 2 + 1, j * 2),
            (i * 2 + 1, j * 2 + 1),
        ):
            window_cells.add((a, b))
    window: list[tuple[int, int]] = []
    start_j = 0
    for a in range(len_x):
        new_start: int | None = None
        for b in range(start_j, len_y):
            if (a, b) in window_cells:
                window.append((a, b))
                if new_start is None:
                    new_start = b
            elif new_start is not None:
                break
        start_j = 0 if new_start is None else new_start
    return window


def fastdtw(
    x: np.ndarray,
    y: np.ndarray,
    *,
    radius: int = FASTDTW_RADIUS,
    dist=None,
) -> tuple[float, list[tuple[int, int]]]:
    """Approximate DTW matching ``fastdtw.fastdtw`` 0.3.4."""
    left = _prep_series(x)
    right = _prep_series(y)
    if left.ndim != right.ndim:
        raise ReimplementationError("DTW series ranks must match")
    if left.ndim > 1 and left.shape[1:] != right.shape[1:]:
        raise ReimplementationError("DTW feature dimensions must match")
    if dist is None:
        metric = _difference if left.ndim == 1 else _euclidean
    else:
        metric = dist
    return _fastdtw(left, right, int(radius), metric)


def _fastdtw(
    x: np.ndarray,
    y: np.ndarray,
    radius: int,
    dist,
) -> tuple[float, list[tuple[int, int]]]:
    min_time_size = radius + 2
    if int(x.shape[0]) < min_time_size or int(y.shape[0]) < min_time_size:
        return _dtw_windowed(x, y, None, dist)
    x_small = _reduce_by_half(x)
    y_small = _reduce_by_half(y)
    _, path = _fastdtw(x_small, y_small, radius, dist)
    window = _expand_window(path, int(x.shape[0]), int(y.shape[0]), radius)
    return _dtw_windowed(x, y, window, dist)


def pairwise_fastdtw_distance(
    series: np.ndarray,
    *,
    radius: int = FASTDTW_RADIUS,
) -> np.ndarray:
    """Pairwise FastDTW on ``[T, N, C]`` daily-mean curves. Upper triangle then mirror."""
    data = np.asarray(series, dtype=np.float64)
    if data.ndim != 3:
        raise ReimplementationError(f"DTW input must be [T, N, C], got {data.shape}")
    if not np.isfinite(data).all():
        raise ReimplementationError("DTW input contains NaN or inf")
    node_count = int(data.shape[1])
    distances = np.zeros((node_count, node_count), dtype=np.float64)
    for i in range(node_count):
        for j in range(i, node_count):
            distance, _ = fastdtw(data[:, i, :], data[:, j, :], radius=radius)
            distances[i, j] = distance
            distances[j, i] = distance
    if not np.isfinite(distances).all():
        raise ReimplementationError("DTW matrix contains NaN or inf")
    return distances


def semantic_mask(
    dtw_matrix: np.ndarray,
    *,
    dtw_delta: int,
    include_self: bool = True,
) -> np.ndarray:
    """Per-row top-``dtw_delta`` neighbors are allowed (mask False).

    Original::

        sem_mask = ones  # True = mask out
        sem_mask[i, argsort(dtw, axis=1)[i, :dtw_delta]] = 0

    ``dtw_delta`` is a count, not a distance threshold. Ties break by
    ``(distance, node_index)`` (deterministic; original ``argsort`` is unstable).
    The mask is **not** symmetrized.
    """
    distances = _as_square(np.asarray(dtw_matrix, dtype=np.float64), name="dtw_matrix")
    node_count = int(distances.shape[0])
    k = int(dtw_delta)
    if k <= 0:
        raise ReimplementationError(f"dtw_delta must be a positive count, got {k}")
    if k > node_count:
        raise ReimplementationError(f"dtw_delta={k} exceeds node count {node_count}")
    mask = np.ones((node_count, node_count), dtype=bool)
    for i in range(node_count):
        order = np.lexsort((np.arange(node_count), distances[i]))
        chosen = order[:k]
        mask[i, chosen] = False
        if include_self and mask[i, i]:
            raise ReimplementationError(
                f"node {i} does not keep itself among the {k} nearest DTW neighbors"
            )
    allowed = (~mask).sum(axis=1)
    if np.any(allowed <= 0):
        raise ReimplementationError("semantic mask fully blocks at least one row")
    return mask


def normalized_laplacian(adj: np.ndarray) -> tuple[np.ndarray, int]:
    """``I - D^{-1/2} A^T D^{-1/2}`` matching ``PDFormerExecutor._calculate_normalized_laplacian``.

    Isolated nodes (degree 0) get ``D^{-1/2}=0``. For undirected ``A=A^T`` this
    equals the usual symmetric normalized Laplacian.
    """
    adjacency = _as_square(np.asarray(adj, dtype=np.float64), name="laplacian adjacency")
    if not np.isfinite(adjacency).all():
        raise ReimplementationError("Laplacian adjacency contains NaN or inf")
    degree = adjacency.sum(axis=1)
    isolated = int(np.sum(degree == 0.0))
    inv_sqrt = np.power(degree, -0.5)
    inv_sqrt[~np.isfinite(inv_sqrt)] = 0.0
    scale = np.diag(inv_sqrt)
    laplacian = np.eye(adjacency.shape[0], dtype=np.float64) - scale @ adjacency.T @ scale
    return laplacian, isolated


def laplacian_positional_encoding(
    adj: np.ndarray,
    lape_dim: int,
    *,
    sign_convention: str = "max_abs_positive",
) -> dict[str, Any]:
    """Eigenvectors of the normalized Laplacian, original slice rule.

    Original takes ``EigVec[:, isolated+1 : lape_dim+isolated+1]`` after sorting
    eigenvalues ascending. The first non-isolated eigenvector (typically the
    constant mode) is skipped.

    ``sign_convention='max_abs_positive'`` is a reproducibility adapter: original
    training later applies ``random_flip`` to eigenvector signs. It does not
    change the span of the PE subspace.
    """
    original = np.array(adj, copy=True)
    laplacian, isolated = normalized_laplacian(adj)
    eigval, eigvec = np.linalg.eig(laplacian)
    order = np.argsort(eigval)
    eigval = eigval[order]
    eigvec = np.real(eigvec[:, order])
    start = int(isolated) + 1
    stop = int(lape_dim) + int(isolated) + 1
    if int(lape_dim) <= 0:
        raise ReimplementationError(f"lape_dim must be positive, got {lape_dim}")
    if stop > eigvec.shape[1]:
        raise ReimplementationError(
            f"not enough eigenvectors: need columns [{start}:{stop}] from {eigvec.shape[1]}"
        )
    pe = np.array(eigvec[:, start:stop], dtype=np.float64, copy=True)
    if pe.ndim == 1:
        pe = pe[:, np.newaxis]
    pe32 = pe.astype(np.float32, copy=True)
    if sign_convention == "max_abs_positive":
        for column in range(pe32.shape[1]):
            vec = pe32[:, column]
            pivot = int(np.argmax(np.abs(vec)))
            if vec[pivot] < 0.0:
                vec *= -1.0
            pivot = int(np.argmax(np.abs(vec)))
            if vec[pivot] < 0.0:
                vec *= -1.0
    elif sign_convention not in {"none", "original_unsigned"}:
        raise ReimplementationError(f"unknown Laplacian sign convention {sign_convention}")
    if not np.isfinite(pe32).all():
        raise ReimplementationError("Laplacian PE contains NaN or inf")
    if not np.array_equal(original, adj):
        raise ReimplementationError("laplacian_positional_encoding modified the input adjacency")
    return {
        "laplacian_pe": pe32,
        "eigenvalues": np.real(eigval).astype(np.float64),
        "isolated_point_num": isolated,
        "sign_convention": sign_convention,
        "slice": [start, stop],
    }


def assert_not_validation_or_test_path(path: Path) -> Path:
    resolved = Path(path)
    name = resolved.name.lower()
    parts = {part.lower() for part in resolved.parts}
    if name in {"validation.npz", "test.npz"} or "validation" in parts or "test" in parts:
        if "train" not in name and name not in {"split_manifest.json", "dataset_metadata.json"}:
            if name in {"validation.npz", "test.npz"} or any(
                part in {"validation", "test"} for part in resolved.parts
            ):
                if name in {"validation.npz", "test.npz"}:
                    raise ReimplementationError(
                        f"refusing validation/test file as DTW/relation input: {resolved}"
                    )
    if name in {"validation.npz", "test.npz"}:
        raise ReimplementationError(f"refusing validation/test NPZ: {resolved}")
    return resolved


def reject_heldout_relation_input(path: Path) -> Path:
    resolved = Path(path)
    lowered = resolved.as_posix().lower()
    if lowered.endswith("validation.npz") or lowered.endswith("test.npz"):
        raise ReimplementationError(
            f"PDFormer relations cannot be fit on validation/test: {resolved}"
        )
    return resolved


def assert_training_days_only(
    days: Sequence[str],
    manifest: Mapping[str, Any],
) -> list[str]:
    training = [str(item) for item in manifest["training_days"]]
    heldout = {
        str(item)
        for item in list(manifest.get("validation_days", [])) + list(manifest.get("test_days", []))
    }
    overlap = [day for day in training if day in heldout]
    if overlap:
        raise ReimplementationError(f"training_days overlap held-out days: {overlap}")
    requested = [str(item) for item in days]
    bad = [day for day in requested if day in heldout]
    if bad:
        raise ReimplementationError(
            f"PDFormer DTW cannot use validation/test days: {bad}. "
            "Use split_manifest training_days only."
        )
    unknown = [day for day in requested if day not in training]
    if unknown:
        raise ReimplementationError(f"days are not in training_days: {unknown}")
    return requested


def daily_mean_from_days(daily_flow: np.ndarray) -> np.ndarray:
    """Mean over days of ``[D, T, N]`` full flow -> ``[T, N, 1]``."""
    array = np.asarray(daily_flow, dtype=np.float64)
    if array.ndim != 3:
        raise ReimplementationError(f"daily flow must be [D, T, N], got {array.shape}")
    if array.shape[0] == 0:
        raise ReimplementationError("daily flow has no training days")
    if not np.isfinite(array).all():
        raise ReimplementationError("daily flow contains NaN or inf")
    mean = np.mean(array, axis=0, dtype=np.float64)
    return mean[..., np.newaxis]


def load_day_flow_csv(
    path: Path,
    *,
    node_ids: Sequence[str],
    steps_per_day: int,
    expected_sha256: str | None = None,
) -> np.ndarray:
    import pandas as pd

    if not path.is_file():
        raise ReimplementationError(f"full-flow CSV not found: {path}")
    reject_heldout_relation_input(path)
    if expected_sha256 is not None:
        digest = sha256_file(path)
        if digest != expected_sha256:
            raise ReimplementationError(
                f"{path} sha256 {digest} != dataset_metadata {expected_sha256}"
            )
    frame = pd.read_csv(path, dtype={"edge_id": str})
    required = {"window_index", "edge_id", "vehicle_count"}
    missing = required.difference(frame.columns)
    if missing:
        raise ReimplementationError(f"{path} missing columns {sorted(missing)}")
    table = frame.pivot(index="window_index", columns="edge_id", values="vehicle_count")
    table = table.reindex(index=list(range(int(steps_per_day))), columns=list(node_ids))
    if table.shape != (int(steps_per_day), len(node_ids)):
        raise ReimplementationError(
            f"{path} pivoted shape {table.shape} != ({steps_per_day}, {len(node_ids)})"
        )
    if table.isna().any().any():
        raise ReimplementationError(
            f"{path} has missing (day, slot, node) cells; refusing to fill"
        )
    values = np.asarray(table.to_numpy(), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ReimplementationError(f"{path} contains NaN or inf after pivot")
    return values


def test_only_dtw_matrix(node_count: int) -> np.ndarray:
    """Deterministic in-memory DTW stand-in. Never an official semantic graph."""
    index = np.arange(int(node_count), dtype=np.float64)
    distances = np.abs(index[:, None] - index[None, :])
    np.fill_diagonal(distances, 0.0)
    return distances


def test_only_pattern_keys(
    *,
    n_cluster: int,
    s_attn_size: int,
    output_dim: int,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.RandomState(int(seed))
    keys = rng.normal(size=(int(n_cluster), int(s_attn_size), int(output_dim))).astype(np.float32)
    return keys


def load_binary_topology(path: Path, *, expected_nodes: int | None = None) -> np.ndarray:
    if not path.is_file():
        raise ReimplementationError(f"topology not found: {path}")
    loaded = np.array(np.load(path, allow_pickle=False), dtype=np.float64, copy=True)
    topology = assert_binary_topology(loaded, name=str(path))
    if expected_nodes is not None and int(topology.shape[0]) != int(expected_nodes):
        raise ReimplementationError(
            f"{path} has {topology.shape[0]} nodes, expected {expected_nodes}"
        )
    return topology.astype(np.float32)
