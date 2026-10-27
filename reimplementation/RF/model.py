"""K-Means, univariate LAD regression, and a random-forest soft vote.

The cluster-specific line minimises absolute error in the original FCD-flow
units. K-Means and the forest see the seven standardised features. Soft voting
uses ``predict_proba`` columns indexed by ``classes_``, not by column position.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

N_CLUSTERS = 50
N_ESTIMATORS = 50
FEATURE_COUNT = 7


class ModelError(RuntimeError):
    """RF model fit or prediction failed."""


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Lower weighted median. Any point between the two central values is optimal."""
    if values.size == 0 or float(weights.sum()) <= 0:
        raise ModelError("weighted median requires positive weight")
    order = np.argsort(values, kind="mergesort")
    ordered = np.asarray(values, dtype=np.float64)[order]
    cumulative = np.cumsum(np.asarray(weights, dtype=np.float64)[order])
    index = int(np.searchsorted(cumulative, cumulative[-1] * 0.5, side="left"))
    return float(ordered[min(index, ordered.size - 1)])


def _aggregate(flow: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flow = np.asarray(flow, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if flow.shape != target.shape or flow.ndim != 1:
        raise ModelError("LAD inputs must be paired 1-D arrays")
    if not np.isfinite(flow).all() or not np.isfinite(target).all():
        raise ModelError("LAD inputs contain NaN or Inf")
    rounded_flow = np.rint(flow)
    rounded_target = np.rint(target)
    if float(np.max(np.abs(flow - rounded_flow))) > 1e-4 or float(np.max(np.abs(target - rounded_target))) > 1e-4:
        keys = np.stack([flow, target], axis=1)
    else:
        keys = np.stack([rounded_flow, rounded_target], axis=1)
    pairs, counts = np.unique(keys, axis=0, return_counts=True)
    return pairs[:, 0], pairs[:, 1], counts.astype(np.float64)


def fit_lad(flow: np.ndarray, target: np.ndarray) -> dict[str, float | str]:
    """Univariate least-absolute-deviations line target ≈ alpha + beta * flow."""
    if len(flow) == 0:
        return {"alpha": 0.0, "beta": 0.0, "mae": float("nan"), "status": "empty", "samples": 0}
    flow_u, target_u, weights = _aggregate(flow, target)
    samples = int(round(float(weights.sum())))

    def total_absolute_error(beta: float) -> tuple[float, float]:
        residual = target_u - float(beta) * flow_u
        alpha = weighted_median(residual, weights)
        loss = float(np.dot(weights, np.abs(residual - alpha)))
        return loss, alpha

    if np.unique(flow_u).size == 1:
        loss, alpha = total_absolute_error(0.0)
        return {
            "alpha": alpha,
            "beta": 0.0,
            "mae": loss / samples,
            "status": "duplicate_flow",
            "samples": samples,
        }

    positive = np.abs(flow_u) > 0
    slopes = np.abs(target_u[positive] / flow_u[positive])
    width = float(np.percentile(slopes, 99) * 10.0 + 10.0) if slopes.size else 10.0
    status = "converged"
    beta = 0.0
    loss = float("inf")
    for _attempt in range(3):
        result = minimize_scalar(
            lambda value: total_absolute_error(value)[0],
            bounds=(-width, width),
            method="bounded",
            options={"xatol": 1e-10, "maxiter": 500},
        )
        if not result.success or not np.isfinite(result.fun):
            raise ModelError(f"LAD scalar search failed: {result.message}")
        beta = float(result.x)
        loss = float(result.fun)
        if abs(beta) < 0.95 * width:
            break
        width *= 20.0
        status = "expanded_bounds"
    else:
        status = "boundary"
    _loss_check, alpha = total_absolute_error(beta)
    if not np.isfinite(alpha) or not np.isfinite(beta) or not np.isfinite(loss):
        raise ModelError("LAD returned a non-finite coefficient")
    return {
        "alpha": float(alpha),
        "beta": beta,
        "mae": loss / samples,
        "status": status,
        "samples": samples,
    }


def align_probabilities(proba: np.ndarray, classes: np.ndarray, n_clusters: int) -> np.ndarray:
    """Place ``predict_proba`` columns onto cluster ids 0..n_clusters-1."""
    aligned = np.zeros((proba.shape[0], n_clusters), dtype=np.float64)
    for column, cluster_id in enumerate(classes):
        cluster_index = int(cluster_id)
        if cluster_index < 0 or cluster_index >= n_clusters:
            raise ModelError(f"forest class {cluster_index} is outside 0..{n_clusters - 1}")
        aligned[:, cluster_index] = proba[:, column]
    return aligned


class RFEstimator:
    def __init__(
        self,
        *,
        n_clusters: int = N_CLUSTERS,
        n_estimators: int = N_ESTIMATORS,
        seed: int = 42,
        n_jobs: int = 1,
        kmeans_n_init: int = 10,
    ) -> None:
        self.n_clusters = int(n_clusters)
        self.n_estimators = int(n_estimators)
        self.seed = int(seed)
        self.n_jobs = int(n_jobs)
        self.kmeans_n_init = int(kmeans_n_init)
        self.scaler = StandardScaler()
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            init="k-means++",
            n_init=self.kmeans_n_init,
            max_iter=300,
            tol=1e-4,
            random_state=self.seed,
            algorithm="lloyd",
        )
        self.forest = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion="gini",
            random_state=self.seed,
            n_jobs=self.n_jobs,
            bootstrap=True,
        )
        self.alpha = np.zeros(self.n_clusters, dtype=np.float64)
        self.beta = np.zeros(self.n_clusters, dtype=np.float64)
        self.cluster_samples = np.zeros(self.n_clusters, dtype=np.int64)
        self.cluster_mae = np.full(self.n_clusters, np.nan, dtype=np.float64)
        self.cluster_status = np.array(["unfit"] * self.n_clusters, dtype="U32")
        self.kmeans_iterations = -1
        self.kmeans_inertia = float("nan")

    def fit(self, features: np.ndarray, target: np.ndarray) -> "RFEstimator":
        features = np.asarray(features)
        target = np.asarray(target, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != FEATURE_COUNT:
            raise ModelError(f"expected [N, 7] features, got {features.shape}")
        if target.shape != (features.shape[0],):
            raise ModelError("target length does not match the feature rows")
        if features.shape[0] < self.n_clusters:
            raise ModelError("fewer samples than clusters")
        print(f"scaler+kmeans start rows={features.shape[0]}", flush=True)
        scaled = self.scaler.fit_transform(features)
        with threadpool_limits(limits=max(self.n_jobs, 1)):
            labels = self.kmeans.fit_predict(scaled)
        self.kmeans_iterations = int(getattr(self.kmeans, "n_iter_", -1))
        self.kmeans_inertia = float(self.kmeans.inertia_)
        print(
            f"kmeans done iterations={self.kmeans_iterations} inertia={self.kmeans_inertia:.6g}",
            flush=True,
        )
        flow = np.asarray(features[:, 0], dtype=np.float64)
        for cluster_id in range(self.n_clusters):
            selected = labels == cluster_id
            fitted = fit_lad(flow[selected], target[selected])
            self.alpha[cluster_id] = float(fitted["alpha"])
            self.beta[cluster_id] = float(fitted["beta"])
            self.cluster_samples[cluster_id] = int(fitted["samples"])
            self.cluster_mae[cluster_id] = float(fitted["mae"])
            self.cluster_status[cluster_id] = str(fitted["status"])
        print("lad done; random forest start", flush=True)
        self.forest.fit(scaled, labels)
        print("random forest done", flush=True)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features)
        if features.ndim != 2 or features.shape[1] != FEATURE_COUNT:
            raise ModelError(f"expected [N, 7] features, got {features.shape}")
        scaled = self.scaler.transform(features)
        probabilities = self.forest.predict_proba(scaled)
        aligned = align_probabilities(probabilities, self.forest.classes_, self.n_clusters)
        row_sums = aligned.sum(axis=1)
        if float(np.max(np.abs(row_sums - 1.0))) > 1e-5:
            raise ModelError("soft-vote probabilities are not normalised")
        flow = np.asarray(features[:, 0], dtype=np.float64)
        return aligned @ self.alpha + flow * (aligned @ self.beta)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, directory / "scaler.joblib")
        joblib.dump(self.kmeans, directory / "kmeans.joblib")
        joblib.dump(self.forest, directory / "random_forest.joblib")
        np.savez(
            directory / "regression_coefficients.npz",
            alpha=self.alpha,
            beta=self.beta,
            cluster_sample_counts=self.cluster_samples,
            cluster_training_mae=self.cluster_mae,
            optimizer_status=np.asarray(self.cluster_status, dtype="U32"),
            n_clusters=np.int32(self.n_clusters),
        )
        payload = {
            "n_clusters": self.n_clusters,
            "n_estimators": self.n_estimators,
            "seed": self.seed,
            "n_jobs": self.n_jobs,
            "kmeans_n_init": self.kmeans_n_init,
            "kmeans_iterations": self.kmeans_iterations,
            "kmeans_inertia": self.kmeans_inertia,
            "kmeans_params": self.kmeans.get_params(deep=False),
            "forest_params": self.forest.get_params(deep=False),
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "feature_names": [
                "fcd_flow",
                "fcd_average_speed",
                "normalized_speed",
                "num_lanes",
                "speed_limit",
                "weekday",
                "hour",
            ],
            "regression": "univariate LAD, alpha + beta * raw fcd_flow",
            "voting": "soft",
        }
        (directory / "model_config.json").write_text(
            json.dumps(payload, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> "RFEstimator":
        config = json.loads((directory / "model_config.json").read_text(encoding="utf-8"))
        estimator = cls(
            n_clusters=int(config["n_clusters"]),
            n_estimators=int(config["n_estimators"]),
            seed=int(config["seed"]),
            n_jobs=int(config["n_jobs"]),
            kmeans_n_init=int(config["kmeans_n_init"]),
        )
        estimator.scaler = joblib.load(directory / "scaler.joblib")
        estimator.kmeans = joblib.load(directory / "kmeans.joblib")
        estimator.forest = joblib.load(directory / "random_forest.joblib")
        with np.load(directory / "regression_coefficients.npz", allow_pickle=False) as payload:
            estimator.alpha = np.asarray(payload["alpha"], dtype=np.float64)
            estimator.beta = np.asarray(payload["beta"], dtype=np.float64)
            estimator.cluster_samples = np.asarray(payload["cluster_sample_counts"], dtype=np.int64)
            estimator.cluster_mae = np.asarray(payload["cluster_training_mae"], dtype=np.float64)
            estimator.cluster_status = np.asarray(payload["optimizer_status"]).astype(str)
        return estimator


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def self_test() -> None:
    probabilities = np.array([[0.2, 0.8], [1.0, 0.0]])
    aligned = align_probabilities(probabilities, np.array([1, 0]), 3)
    if aligned.shape != (2, 3) or abs(aligned[0, 0] - 0.8) > 1e-12 or abs(aligned[0, 1] - 0.2) > 1e-12:
        raise ModelError("class-column alignment failed")
    alpha = np.array([1.0, 10.0, 0.0])
    beta = np.array([1.0, 0.0, 0.0])
    flow = np.array([4.0, 4.0])
    voted = aligned @ alpha + flow * (aligned @ beta)
    # Class 0 has weight 0.8 and line 1+1*flow; class 1 has weight 0.2 and line 10.
    if abs(voted[0] - (0.8 * 5.0 + 0.2 * 10.0)) > 1e-10:
        raise ModelError(f"soft vote value {voted[0]}")

    rng = np.random.default_rng(42)
    flow = rng.integers(0, 8, size=80).astype(np.float64)
    target = 1.5 + 2.5 * flow + rng.normal(0.0, 0.05, size=80)
    fitted = fit_lad(flow, target)
    reference = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs", fit_intercept=True)
    reference.fit(flow.reshape(-1, 1), target)

    def absolute_error(alpha: float, beta: float) -> float:
        return float(np.abs(target - alpha - beta * flow).sum())

    ours = absolute_error(float(fitted["alpha"]), float(fitted["beta"]))
    other = absolute_error(float(reference.intercept_), float(reference.coef_[0]))
    if ours > other + 1e-5:
        raise ModelError(f"LAD loss {ours} is worse than QuantileRegressor {other}: {fitted}")
    exact_flow = np.array([0.0, 1.0, 2.0, 4.0, 7.0])
    exact = fit_lad(exact_flow, 1.5 + 2.5 * exact_flow)
    if abs(float(exact["alpha"]) - 1.5) > 1e-6 or abs(float(exact["beta"]) - 2.5) > 1e-6:
        raise ModelError(f"exact line was not recovered: {exact}")
    duplicate = fit_lad(np.array([3.0, 3.0, 3.0, 3.0]), np.array([1.0, 2.0, 9.0, 4.0]))
    if duplicate["status"] != "duplicate_flow" or abs(float(duplicate["beta"])) > 0 or abs(float(duplicate["alpha"]) - 2.0) > 1e-8:
        raise ModelError(f"duplicate-flow LAD failed: {duplicate}")

    synthetic_x = rng.normal(size=(120, FEATURE_COUNT))
    synthetic_x[:, 0] = rng.integers(0, 5, size=120)
    labels = np.where(synthetic_x[:, 0] < 2, 0, np.where(synthetic_x[:, 0] < 4, 1, 2))
    synthetic_y = np.choose(labels, [1.0, 3.0, 5.0]) + 0.5 * synthetic_x[:, 0]
    estimator = RFEstimator(n_clusters=3, n_estimators=8, seed=42, n_jobs=1, kmeans_n_init=3)
    estimator.fit(synthetic_x, synthetic_y)
    first = estimator.predict(synthetic_x[:6])
    from tempfile import TemporaryDirectory

    with TemporaryDirectory(prefix="rf_selftest_") as temporary:
        directory = Path(temporary)
        estimator.save(directory)
        restored = RFEstimator.load(directory)
        second = restored.predict(synthetic_x[:6])
    if not np.allclose(first, second, atol=1e-8):
        raise ModelError("reloaded model predictions differ")
    if estimator.forest.criterion != "gini" or estimator.forest.n_estimators != 8:
        raise ModelError("forest settings drifted")
    print("rf self-test PASS", flush=True)
