from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Allow running with:
#   - `cd vis/backend && uvicorn app:app --reload`
#   - `cd vis/backend && python app.py`
# and also tolerate `python -m uvicorn app:app` where import paths vary.
try:
    from data import TrafficData  # type: ignore
except Exception:  # pragma: no cover
    from .data import TrafficData  # type: ignore


HERE = Path(__file__).resolve().parent
VIS_DIR = HERE.parent
REPO_ROOT = VIS_DIR.parent

DEFAULT_CSV = REPO_ROOT / "out" / "fcd_geo.csv"
CESIUM_BUILD = VIS_DIR / "frontend" / "Cesium-1.127" / "Build" / "Cesium"
FRONTEND_DIR = VIS_DIR / "frontend"
OUT_DIR = REPO_ROOT / "out"


def _parse_bbox(bbox: Optional[str]) -> Optional[list[float]]:
    if not bbox:
        return None
    parts = [p.strip() for p in bbox.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be west,south,east,north")
    return [float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])]


class QueryBody(BaseModel):
    timeStart: float = Field(..., description="seconds")
    timeEnd: float = Field(..., description="seconds")
    bbox: Optional[list[float]] = Field(None, description="[west,south,east,north] in degrees")
    vehicleIds: Optional[list[str]] = None
    # 0 or negative means "no limit"
    maxPoints: int = 0
    # 0 or negative means "no sampling"
    sampleEvery: float = 0.0


def create_app() -> FastAPI:
    app = FastAPI(title="TrafficFlow Vis API", version="0.1.0")

    # Dev-friendly CORS (front can be hosted separately if needed)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    csv_path = Path(os.environ.get("FCD_GEO_CSV", str(DEFAULT_CSV))).resolve()
    def _get_traffic() -> TrafficData:
        traffic = getattr(app.state, "traffic", None)
        if traffic is None:
            app.state.traffic = TrafficData(str(csv_path))
            traffic = app.state.traffic
        return traffic

    @app.on_event("startup")
    def _startup() -> None:
        app.state.csv_path = str(csv_path)
        # NOTE: Do NOT load large CSV at startup (user requested). Data will be loaded lazily
        # on the first call to /api/vehicles, /api/trajectory, or /api/query.
        # app.state.traffic = TrafficData(str(csv_path))

    @app.get("/api/health")
    def health() -> dict:
        traffic = getattr(app.state, "traffic", None)
        return {
            "ok": True,
            "preloaded": traffic is not None,
            "csvPath": str(csv_path),
            "meta": traffic.meta() if traffic is not None else None,
        }

    @app.get("/api/vehicles")
    def vehicles(
        timeStart: Annotated[float, Query(..., description="seconds")],
        timeEnd: Annotated[float, Query(..., description="seconds")],
        # NOTE: FastAPI requires the default to be set with `= None` when using `Annotated[...]`.
        bbox: Annotated[Optional[str], Query(description="west,south,east,north (degrees)")] = None,
        # Same rule: default should be set with `= 5000` (not in Query()) to keep Python signature valid.
        limit: Annotated[int, Query(ge=1, le=20000)] = 5000,
    ) -> dict:
        traffic = _get_traffic()
        bbox_list = _parse_bbox(bbox)
        ids = traffic.vehicles(timeStart, timeEnd, bbox=bbox_list, limit=limit)
        return {"vehicleIds": ids}

    @app.get("/api/trajectory/{vehicle_id}")
    def trajectory(
        vehicle_id: str,
        # Same rule as above: default must be set with `= None` (not inside Query()) when using Annotated.
        timeStart: Annotated[Optional[float], Query(description="seconds")] = None,
        timeEnd: Annotated[Optional[float], Query(description="seconds")] = None,
    ) -> JSONResponse:
        traffic = _get_traffic()
        data = traffic.trajectory(vehicle_id, time_start=timeStart, time_end=timeEnd)
        return JSONResponse(data)

    @app.post("/api/query")
    def query(body: QueryBody) -> dict:
        traffic = _get_traffic()
        points, ids, stats = traffic.query(
            time_start=body.timeStart,
            time_end=body.timeEnd,
            bbox=body.bbox,
            vehicle_ids=body.vehicleIds,
            max_points=body.maxPoints,
            sample_every=body.sampleEvery,
        )
        return {
            "vehicleIds": ids,
            "points": points,
            "stats": {
                "rows": stats.rows,
                "vehicles": stats.vehicles,
                "truncated": stats.truncated,
                "sampleEvery": stats.sample_every,
                "maxPoints": stats.max_points,
            },
        }

    # ---- Static hosting: Cesium build + frontend ----
    if CESIUM_BUILD.exists():
        app.mount("/cesium", StaticFiles(directory=str(CESIUM_BUILD)), name="cesium")

    if FRONTEND_DIR.exists():
        app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

    # Serve repo `out/` as static files so the frontend can load data directly (no API call).
    # Example: /data/net_lanes_internal.geojson
    if OUT_DIR.exists():
        app.mount("/data", StaticFiles(directory=str(OUT_DIR)), name="data")

    @app.get("/", response_class=HTMLResponse)
    def index() -> FileResponse:
        index_path = FRONTEND_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail=f"frontend index not found: {index_path}")
        return FileResponse(index_path)

    return app


app = create_app()


def _main() -> None:
    """
    Convenience entrypoint:
      python app.py
    """
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), reload=True)


if __name__ == "__main__":
    _main()




