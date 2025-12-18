// Minimal Cesium frontend: OSM basemap + sample datasource

const statusEl = document.getElementById("status");

function setStatus(msg) {
  if (statusEl) statusEl.textContent = msg;
}

function createViewer() {
  Cesium.Ion.defaultAccessToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiI1YjFhYTRjZS0zYzZlLTRmN2ItOTE5NC1mMzEwYjFiZjE3NTUiLCJpZCI6MzEzNjA3LCJpYXQiOjE3NTAzMDY1MjF9.k7exedEe-OwSQ2qgC5NNIMec5tXhTiCEp6of6vdYv0o';

  return new Cesium.Viewer("cesiumContainer", {
    imageryProvider: new Cesium.OpenStreetMapImageryProvider({
      url: "https://a.tile.openstreetmap.org/",
    }),
    terrainProvider: new Cesium.EllipsoidTerrainProvider(),
    timeline: true,
    animation: true,
    shouldAnimate: false,
    sceneModePicker: false,
    baseLayerPicker: false,
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

async function main() {
  setStatus("初始化 Cesium...");

  const viewer = createViewer();

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
