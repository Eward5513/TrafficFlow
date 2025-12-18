# TrafficFlow Cesium Visualization

This folder contains:
- a lightweight **Cesium frontend**
- a **Python FastAPI backend** (served by `uvicorn`) for querying SUMO FCD trajectories.

## Data files (expected locations)

By default the backend reads:
- `../out/fcd_geo.csv`

You can override paths via environment variables:
- `FCD_GEO_CSV=/abs/path/to/fcd_geo.csv`

## Run

### 1) Install backend deps

```bash
cd vis/backend
python3 -m pip install -r requirements.txt
```

### 2) Start backend (serves API + frontend + Cesium)

```bash
cd vis/backend
uvicorn app:app --reload --host 0.0.0.0 --port 8080
```

### 3) Open in browser

- `http://localhost:8080/`

## What the UI does

- **Basemap**: OpenStreetMap tiles.
- (Backend API endpoints are available at `/api/...` on the same origin)

## Notes

- `vehicle_x` is treated as **longitude**, `vehicle_y` as **latitude**.
- If your browser shows a **pure blue globe**, it usually means imagery tiles failed to load, or Cesium couldn't find its static assets.
  In this setup, Cesium static files are mounted at `/cesium/` by the backend.


