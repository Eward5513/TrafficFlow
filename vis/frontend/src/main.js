// Minimal Cesium frontend: OSM basemap + sample datasource

const statusEl = document.getElementById("status");

function setStatus(msg) {
  if (statusEl) statusEl.textContent = msg;
}

function createViewer() {
  Cesium.Ion.defaultAccessToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiI1YjFhYTRjZS0zYzZlLTRmN2ItOTE5NC1mMzEwYjFiZjE3NTUiLCJpZCI6MzEzNjA3LCJpYXQiOjE3NTAzMDY1MjF9.k7exedEe-OwSQ2qgC5NNIMec5tXhTiCEp6of6vdYv0o';

  // BaseLayerPicker 模式下，Viewer 不会使用 imageryProvider 作为默认底图；
  // 需要通过 ViewModel 指定可选项 + 默认选中项。
  const osmViewModel = new Cesium.ProviderViewModel({
    name: "OpenStreetMap",
    tooltip: "OpenStreetMap",
    iconUrl: Cesium.buildModuleUrl("Widgets/Images/ImageryProviders/openStreetMap.png"),
    creationFunction: () =>
      new Cesium.OpenStreetMapImageryProvider({
        url: "https://tile.openstreetmap.org/",
      }),
  });

  // 既保留 Cesium 默认的可选底图（ion 等），又把 OSM 放在最前并默认选中。
  const imageryProviderViewModels = [osmViewModel, ...Cesium.createDefaultImageryProviderViewModels()];

  return new Cesium.Viewer("cesiumContainer", {
    imageryProviderViewModels,
    selectedImageryProviderViewModel: osmViewModel,
    terrainProvider: new Cesium.EllipsoidTerrainProvider(),
    timeline: true,
    animation: true,
    shouldAnimate: false,
    sceneModePicker: false,
    baseLayerPicker: true,
    geocoder: false,
    homeButton: false,
    navigationHelpButton: false,
    navigationInstructionsInitiallyVisible: false,
  });
}

function addSampleDataSource(viewer) {
  // Simple demo track near Shanghai.
  const ds = new Cesium.CustomDataSource("demo");
  const positions = Cesium.Cartesian3.fromDegreesArray([
    121.2505, 31.2952,
    121.2520, 31.2965,
    121.2538, 31.2978,
    121.2555, 31.2988,
    121.2570, 31.2994,
  ]);

  ds.entities.add({
    name: "Demo polyline",
    polyline: {
      positions,
      width: 4,
      material: Cesium.Color.ORANGE.withAlpha(0.9),
      clampToGround: false,
    },
  });

  ds.entities.add({
    name: "Demo point",
    position: Cesium.Cartesian3.fromDegrees(121.2555, 31.2988, 0),
    point: {
      pixelSize: 10,
      color: Cesium.Color.RED,
      outlineColor: Cesium.Color.WHITE,
      outlineWidth: 2,
    },
    label: {
      text: "示例数据源",
      font: "14px sans-serif",
      fillColor: Cesium.Color.WHITE,
      outlineColor: Cesium.Color.BLACK,
      outlineWidth: 2,
      style: Cesium.LabelStyle.FILL_AND_OUTLINE,
      verticalOrigin: Cesium.VerticalOrigin.TOP,
      pixelOffset: new Cesium.Cartesian2(0, -18),
    },
  });

  viewer.dataSources.add(ds);
  return ds;
}

async function loadRoadNetwork(viewer) {
  // Backend mounts repo `out/` as `/data` (see vis/backend/app.py).
  // Road network GeoJSON:
  //   /data/roadnet.geojson
  const url = "/data/roadnet.geojson";
  setStatus(`加载路网: ${url} ...`);

  const ds = await Cesium.GeoJsonDataSource.load(url, {
    clampToGround: true,
  });

  // Style: thin cyan lines, subtle fill if polygons exist.
  for (const e of ds.entities.values) {
    if (e.polyline) {
      e.polyline.width = 2;
      e.polyline.material = Cesium.Color.CYAN.withAlpha(0.85);
      e.polyline.clampToGround = true;
    }
    if (e.polygon) {
      e.polygon.material = Cesium.Color.CYAN.withAlpha(0.08);
      e.polygon.outline = true;
      e.polygon.outlineColor = Cesium.Color.CYAN.withAlpha(0.6);
    }
  }

  ds.name = "roadnet";
  viewer.dataSources.add(ds);

  // Optional: zoom to the road network for a good first view.
  try {
    await viewer.zoomTo(ds);
  } catch (_) {
    // ignore
  }

  viewer.scene.requestRender();
  setStatus("路网已加载。");
  return ds;
}

async function loadFirstTrajectoriesFromCsv(viewer, opts = {}) {
  const url = opts.url || "/data/fcd_geo.csv";
  const maxVehicles = opts.maxVehicles || 10;
  const maxTotalPoints = opts.maxTotalPoints || 20000;

  setStatus(`加载 CSV 轨迹(前${maxVehicles}条): ${url} ...`);

  const r = await fetch(url);
  if (!r.ok) throw new Error(`CSV 请求失败: ${r.status} ${r.statusText}`);
  if (!r.body) throw new Error("浏览器不支持流式读取 response.body");

  const reader = r.body.getReader();
  const decoder = new TextDecoder("utf-8");

  let buffer = "";
  let headerParsed = false;
  let idxId = -1, idxLon = -1, idxLat = -1;

  const vehicleOrder = [];
  const pointsByVehicle = new Map(); // id -> [lon,lat,lon,lat,...]
  let totalPoints = 0;

  const pushPoint = (vid, lon, lat) => {
    if (!pointsByVehicle.has(vid)) pointsByVehicle.set(vid, []);
    pointsByVehicle.get(vid).push(lon, lat);
    totalPoints += 1;
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop() || "";

      for (const line of lines) {
        const s = line.trim();
        if (!s) continue;

        if (!headerParsed) {
          // fcd_geo.csv is expected to be semicolon-separated.
          const cols = s.split(";");
          idxId = cols.indexOf("vehicle_id");
          idxLon = cols.indexOf("vehicle_x");
          idxLat = cols.indexOf("vehicle_y");
          if (idxId < 0 || idxLon < 0 || idxLat < 0) {
            throw new Error(`CSV 头缺少必要列: vehicle_id/vehicle_x/vehicle_y (got: ${cols.join(",")})`);
          }
          headerParsed = true;
          continue;
        }

        const cols = s.split(";");
        const vid = cols[idxId];
        if (!vid) continue;

        if (!pointsByVehicle.has(vid)) {
          if (vehicleOrder.length >= maxVehicles) {
            // Stop once we already got enough vehicles and we are seeing a new one.
            await reader.cancel();
            throw new Error("__STOP__");
          }
          vehicleOrder.push(vid);
        }

        const lon = Number(cols[idxLon]);
        const lat = Number(cols[idxLat]);
        if (!Number.isFinite(lon) || !Number.isFinite(lat)) continue;
        pushPoint(vid, lon, lat);

        if (totalPoints >= maxTotalPoints) {
          await reader.cancel();
          throw new Error("__STOP__");
        }
      }
    }
  } catch (e) {
    if (e && e.message === "__STOP__") {
      // normal early stop
    } else {
      throw e;
    }
  } finally {
    try {
      reader.releaseLock();
    } catch (_) {
      // ignore
    }
  }

  const ds = new Cesium.CustomDataSource("csvTrajectories");
  const palette = [
    Cesium.Color.ORANGE,
    Cesium.Color.LIME,
    Cesium.Color.CYAN,
    Cesium.Color.MAGENTA,
    Cesium.Color.YELLOW,
    Cesium.Color.DEEPSKYBLUE,
    Cesium.Color.CHARTREUSE,
    Cesium.Color.SALMON,
    Cesium.Color.VIOLET,
    Cesium.Color.AQUA,
  ];

  let added = 0;
  for (let i = 0; i < vehicleOrder.length; i++) {
    const vid = vehicleOrder[i];
    const arr = pointsByVehicle.get(vid) || [];
    if (arr.length < 4) continue; // need at least 2 points

    const positions = Cesium.Cartesian3.fromDegreesArray(arr);
    const color = palette[i % palette.length].withAlpha(0.9);
    ds.entities.add({
      name: `traj:${vid}`,
      polyline: {
        positions,
        width: 3,
        material: color,
        clampToGround: true,
      },
    });
    added += 1;
  }

  viewer.dataSources.add(ds);
  try {
    await viewer.zoomTo(ds);
  } catch (_) {
    // ignore
  }
  viewer.scene.requestRender();

  setStatus(`CSV轨迹已加载: 车辆 ${added}/${maxVehicles}, 点数 ${totalPoints}（预览）`);
  return ds;
}

async function main() {
  setStatus("初始化 Cesium...");

  const viewer = createViewer();

  // Surface imagery load errors in the UI (very helpful when tiles/paths are wrong).
  try {
    const baseLayer = viewer.imageryLayers.get(0);
    const provider = baseLayer && baseLayer.imageryProvider;
    if (provider && provider.errorEvent && typeof provider.errorEvent.addEventListener === "function") {
      provider.errorEvent.addEventListener((err) => {
        console.error("ImageryProvider error:", err);
        setStatus(`底图加载失败: ${err?.message || err}`);
      });
    }
  } catch (e) {
    console.warn("Failed to attach imagery error handler:", e);
  }

  // Auto-load road network once the page is ready.
  try {
    await loadRoadNetwork(viewer);
  } catch (e) {
    console.error(e);
    setStatus(`路网加载失败: ${e.message || e}`);
  }

  // Load first 10 trajectories from CSV for a quick visual check.
  try {
    await loadFirstTrajectoriesFromCsv(viewer, { url: "/data/fcd_geo.csv", maxVehicles: 10 });
  } catch (e) {
    console.error(e);
    setStatus(`CSV轨迹加载失败: ${e.message || e}`);
  }

  // Fixed view; timeline/animation UI stays visible at the bottom.
  viewer.camera.setView({
    destination: Cesium.Cartesian3.fromDegrees(121.2555, 31.2988, 1500),
  });

  // Disable inertia to avoid drift.
  const c = viewer.scene.screenSpaceCameraController;
  c.inertiaSpin = 0;
  c.inertiaTranslate = 0;
  c.inertiaZoom = 0;

  setStatus("OSM 底图已加载，点击按钮加载示例数据源。");

  const btn = document.getElementById("btnLoadData");
  if (btn) {
    btn.addEventListener("click", () => {
      addSampleDataSource(viewer);
      setStatus("示例数据源已加载。");
      viewer.scene.requestRender();
    });
  }
}

main().catch((e) => {
  console.error(e);
  setStatus(`加载失败: ${e.message || e}`);
});
