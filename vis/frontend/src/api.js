// In uvicorn/FastAPI mode, frontend and backend are served from the same origin.
// Keep requests relative so it works on any host/port.
const API_BASE = "";

export async function apiHealth() {
  const r = await fetch(`${API_BASE}/api/health`);
  if (!r.ok) throw new Error(`health failed: ${r.status}`);
  return await r.json();
}

export async function apiVehicles({ timeStart, timeEnd, bbox, limit = 5000 }) {
  const params = new URLSearchParams();
  params.set("timeStart", String(timeStart));
  params.set("timeEnd", String(timeEnd));
  params.set("limit", String(limit));
  if (bbox) params.set("bbox", bbox.join(","));

  const r = await fetch(`${API_BASE}/api/vehicles?${params.toString()}`);
  if (!r.ok) throw new Error(`vehicles failed: ${r.status}`);
  return await r.json();
}

export async function apiTrajectory({ vehicleId, timeStart, timeEnd }) {
  const params = new URLSearchParams();
  if (timeStart != null) params.set("timeStart", String(timeStart));
  if (timeEnd != null) params.set("timeEnd", String(timeEnd));
  const r = await fetch(`${API_BASE}/api/trajectory/${encodeURIComponent(vehicleId)}?${params.toString()}`);
  if (!r.ok) throw new Error(`trajectory failed: ${r.status}`);
  return await r.json();
}

export async function apiQuery({ vehicleIds, timeStart, timeEnd, bbox, maxPoints, sampleEvery }) {
  const body = {
    vehicleIds: vehicleIds && vehicleIds.length ? vehicleIds : null,
    timeStart,
    timeEnd,
    bbox: bbox || null,
    maxPoints,
    sampleEvery,
  };
  const r = await fetch(`${API_BASE}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`query failed: ${r.status}`);
  return await r.json();
}


