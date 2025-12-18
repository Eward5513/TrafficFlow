from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import polars as pl


@dataclass(frozen=True)
class QueryStats:
    rows: int
    vehicles: int
    truncated: bool
    sample_every: float
    max_points: int


class TrafficData:
    """
    In-memory store for SUMO FCD points.

    CSV schema (semicolon separated) is expected to include at least:
      - timestep_time (float seconds)
      - vehicle_id (string)
      - vehicle_x (lon, float)
      - vehicle_y (lat, float)
    """

    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path
        self.df = self._load(csv_path)

        # Precompute basic stats
        self.min_time: float = float(self.df["timestep_time"].min())
        self.max_time: float = float(self.df["timestep_time"].max())
        self.row_count: int = int(self.df.height)
        self.vehicle_count: int = int(self.df["vehicle_id"].n_unique())

    def _load(self, csv_path: str) -> pl.DataFrame:
        df = pl.read_csv(
            csv_path,
            separator=";",
            infer_schema_length=10_000,
            null_values=["", "NaN", "nan"],
            dtypes={
                "timestep_time": pl.Float64,
                "vehicle_id": pl.Utf8,
                "vehicle_x": pl.Float64,
                "vehicle_y": pl.Float64,
            },
        )

        # Keep only columns we may return frequently; preserve other useful columns too.
        # (If the CSV contains extra fields, we keep them available for future extensions.)
        if "timestep_time" not in df.columns or "vehicle_id" not in df.columns:
            raise ValueError("CSV missing required columns: timestep_time, vehicle_id")
        if "vehicle_x" not in df.columns or "vehicle_y" not in df.columns:
            raise ValueError("CSV missing required columns: vehicle_x, vehicle_y")

        # Sort for stable ordering and faster per-vehicle slicing.
        # Drop obviously invalid rows (no lon/lat) to avoid runtime errors during float conversion.
        df = df.filter(pl.col("vehicle_x").is_not_null() & pl.col("vehicle_y").is_not_null())

        df = df.sort(["vehicle_id", "timestep_time"])
        return df

    def meta(self) -> dict[str, Any]:
        return {
            "csvPath": self.csv_path,
            "minTime": self.min_time,
            "maxTime": self.max_time,
            "rowCount": self.row_count,
            "vehicleCount": self.vehicle_count,
        }

    @staticmethod
    def _apply_bbox(q: pl.DataFrame, bbox: Optional[Iterable[float]]) -> pl.DataFrame:
        if bbox is None:
            return q
        west, south, east, north = list(bbox)
        return q.filter(
            pl.col("vehicle_x").is_not_null()
            & pl.col("vehicle_y").is_not_null()
            & (pl.col("vehicle_x") >= west)
            & (pl.col("vehicle_x") <= east)
            & (pl.col("vehicle_y") >= south)
            & (pl.col("vehicle_y") <= north)
        )

    @staticmethod
    def _apply_time(q: pl.DataFrame, time_start: float, time_end: float) -> pl.DataFrame:
        return q.filter((pl.col("timestep_time") >= time_start) & (pl.col("timestep_time") <= time_end))

    @staticmethod
    def _apply_ids(q: pl.DataFrame, vehicle_ids: Optional[list[str]]) -> pl.DataFrame:
        if not vehicle_ids:
            return q
        return q.filter(pl.col("vehicle_id").is_in(vehicle_ids))

    @staticmethod
    def _sample(q: pl.DataFrame, sample_every: float) -> pl.DataFrame:
        """
        Sample to at most one point per vehicle per time bucket of width `sample_every` seconds.
        This is stable and works even when timestep_time is float (e.g. 0.5 increments).
        """
        if sample_every <= 0:
            return q
        q2 = q.with_columns(((pl.col("timestep_time") / sample_every).floor().cast(pl.Int64)).alias("_bucket"))
        # Keep first row per (vehicle,bucket) in sorted order.
        q2 = q2.group_by(["vehicle_id", "_bucket"], maintain_order=True).agg(pl.all().first())
        # Polars returns columns as lists for agg; unwrap via explode is unnecessary because we used first().
        # However, pl.all().first() yields scalar columns directly.
        return q2.drop("_bucket")

    def vehicles(
        self,
        time_start: float,
        time_end: float,
        bbox: Optional[list[float]] = None,
        limit: int = 5000,
    ) -> list[str]:
        q = self._apply_time(self.df, time_start, time_end)
        q = self._apply_bbox(q, bbox)
        ids = q.select(pl.col("vehicle_id").unique()).get_column("vehicle_id").to_list()
        ids = sorted([str(x) for x in ids])
        return ids[: max(0, int(limit))]

    def trajectory(
        self,
        vehicle_id: str,
        time_start: Optional[float] = None,
        time_end: Optional[float] = None,
        # 0 or negative means "no limit"
        max_points: int = 0,
    ) -> dict[str, Any]:
        q = self.df.filter(pl.col("vehicle_id") == vehicle_id)
        q = q.filter(pl.col("vehicle_x").is_not_null() & pl.col("vehicle_y").is_not_null())
        if time_start is not None and time_end is not None:
            q = self._apply_time(q, time_start, time_end)

        truncated = False
        if max_points and max_points > 0 and q.height > max_points:
            q = q.head(max_points)
            truncated = True

        pts = q.select(["timestep_time", "vehicle_x", "vehicle_y"]).to_dicts()
        coords = [[float(p["vehicle_x"]), float(p["vehicle_y"])] for p in pts]

        return {
            "vehicleId": vehicle_id,
            "truncated": truncated,
            "pointCount": len(coords),
            "geojson": {
                "type": "Feature",
                "properties": {"vehicle_id": vehicle_id},
                "geometry": {"type": "LineString", "coordinates": coords},
            },
            "points": [
                {"t": float(p["timestep_time"]), "lon": float(p["vehicle_x"]), "lat": float(p["vehicle_y"])}
                for p in pts
            ],
        }

    def query(
        self,
        time_start: float,
        time_end: float,
        bbox: Optional[list[float]] = None,
        vehicle_ids: Optional[list[str]] = None,
        # 0 or negative means "no limit"
        max_points: int = 0,
        # 0 or negative means "no sampling"
        sample_every: float = 0.0,
    ) -> tuple[list[dict[str, Any]], list[str], QueryStats]:
        q = self._apply_time(self.df, time_start, time_end)
        q = self._apply_bbox(q, bbox)
        q = self._apply_ids(q, vehicle_ids)
        # Ensure we never emit invalid lon/lat points
        q = q.filter(pl.col("vehicle_x").is_not_null() & pl.col("vehicle_y").is_not_null())
        q = self._sample(q, sample_every)

        vehicle_ids_hit = (
            q.select(pl.col("vehicle_id").unique()).get_column("vehicle_id").to_list()
            if q.height > 0
            else []
        )
        vehicle_ids_hit = sorted([str(x) for x in vehicle_ids_hit])

        truncated = False
        if max_points and max_points > 0 and q.height > max_points:
            q = q.head(max_points)
            truncated = True

        # Minimal payload for point cloud rendering
        pts = q.select(["timestep_time", "vehicle_id", "vehicle_x", "vehicle_y"]).to_dicts()
        points = [
            {
                "t": float(p["timestep_time"]),
                "vehicleId": str(p["vehicle_id"]),
                "lon": float(p["vehicle_x"]),
                "lat": float(p["vehicle_y"]),
            }
            for p in pts
        ]

        stats = QueryStats(
            rows=len(points),
            vehicles=len(vehicle_ids_hit),
            truncated=truncated,
            sample_every=float(sample_every),
            max_points=int(max_points),
        )
        return points, vehicle_ids_hit, stats




