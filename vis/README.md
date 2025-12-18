# TrafficFlow Cesium Visualization

This folder contains a lightweight Cesium frontend and a Python (FastAPI) backend for querying and visualizing SUMO FCD trajectories.

## Data files (expected locations)

By default the backend reads:
- `../out/fcd_geo.csv`
- `../out/net_lanes_internal.geojson`

You can override paths via environment variables:
- `FCD_GEO_CSV=/abs/path/to/fcd_geo.csv`
- `ROADNET_GEOJSON=/abs/path/to/net_lanes_internal.geojson`

## Run (recommended)

### 1) Install backend deps

```bash
cd /home/tzhang174/TrafficFlow/vis/backend
python3 -m pip install -r requirements.txt
```

### 2) Start backend (serves API + frontend + Cesium)

```bash
cd /home/tzhang174/TrafficFlow/vis/backend
uvicorn app:app --reload --host 0.0.0.0 --port 8080
```

### 3) Open in browser

- `http://localhost:8080/`

## What the UI does

- **Basemap**: OpenStreetMap tiles.
- **Road network**: loads `/api/roadnet` (GeoJSON).
- **Vehicle query**:
  - `List vehicles (filters)` queries `/api/vehicles` with your time range + optional bbox.
  - Selecting an id and `Load trajectory` fetches `/api/trajectory/{vehicle_id}` and shows the full polyline + a moving point driven by Cesium Timeline.
- **Time range**:
  - Inputs are in **seconds** (`timestep_time`).
  - `Sync to Timeline` sets a small window around the current Timeline time.
- **Space range**:
  - `Use current view bbox` uses the current camera view rectangle as bbox.
- **Run query (points)**:
  - Queries `/api/query` and renders returned points as a point cloud (limited by `maxPoints` and `sampleEvery`).

## Notes

- `vehicle_x` is treated as **longitude**, `vehicle_y` as **latitude**.
- This UI is configured to **default to full data** (no sampling, no maxPoints limit). If the dataset is large, the browser may become slow or unresponsive.
- To reduce load, set `sampleEvery` (e.g. 2–5 seconds) and/or `maxPoints` (e.g. 50000) before running queries.


