"use strict";

(function bootstrap() {
  const healthEl = document.getElementById("health");
  const dataEl = document.getElementById("data");
  const vinInputEl = document.getElementById("vinInput");
  const drawVinBtnEl = document.getElementById("drawVinBtn");
  const nextVinRouteBtnEl = document.getElementById("nextVinRouteBtn");
  const clearVinBtnEl = document.getElementById("clearVinBtn");
  const vinMergeToggleEl = document.getElementById("vinMergeToggle");
  const vinStatusEl = document.getElementById("vinStatus");
  const vinRouteDataListEl = document.getElementById("vinRouteDataList");
  const statsMetaEl = document.getElementById("statsMeta");
  const statsSummaryEl = document.getElementById("statsSummary");
  const statsListEl = document.getElementById("statsList");
  const statsFileNameInputEl = document.getElementById("statsFileNameInput");
  const applyStatsBtnEl = document.getElementById("applyStatsBtn");
  const highlightToggleEl = document.getElementById("highlightToggle");
  const basemapToggleEl = document.getElementById("basemapToggle");
  const statsPanelEl = document.getElementById("statsPanel");
  const statsPanelHeaderEl = statsPanelEl ? statsPanelEl.querySelector(".panel-header") : null;
  const togglePanelBtnEl = document.getElementById("togglePanelBtn");
  const toggleListBtnEl = document.getElementById("toggleListBtn");
  const startEndMetaEl = document.getElementById("startEndMeta");
  const startEndSummaryEl = document.getElementById("startEndSummary");
  const startEndListEl = document.getElementById("startEndList");
  const startEndFileNameInputEl = document.getElementById("startEndFileNameInput");
  const applyStartEndBtnEl = document.getElementById("applyStartEndBtn");
  const startEndHighlightToggleEl = document.getElementById("startEndHighlightToggle");
  const startEndTypeSelectEl = document.getElementById("startEndTypeSelect");
  const startEndPanelEl = document.getElementById("startEndPanel");
  const startEndPanelHeaderEl = startEndPanelEl ? startEndPanelEl.querySelector(".panel-header") : null;
  const toggleStartEndPanelBtnEl = document.getElementById("toggleStartEndPanelBtn");
  const toggleStartEndListBtnEl = document.getElementById("toggleStartEndListBtn");
  const vinPanelEl = document.getElementById("vinPanel");
  const vinPanelHeaderEl = vinPanelEl ? vinPanelEl.querySelector(".panel-header") : null;
  const toggleVinPanelBtnEl = document.getElementById("toggleVinPanelBtn");
  const START_END_HIGHLIGHT_COLOR = "rgb(139, 0, 0)";

  const state = {
    basemapLayer: null,
    basemapGeojson: null,
    basemapFeatureCount: 0,
    basemapVisible: true,
    highlightLayer: null,
    highlightFeatureCount: 0,
    highlightVisible: true,
    statsSourceUrl: "-",
    rawTopEdges: [],
    edgeCountByBaseId: new Map(),
    minHighlightCount: 0,
    maxHighlightCount: 1,
    panelCollapsed: false,
    listCollapsed: false,
    vinPanelCollapsed: false,
    startEndPanelCollapsed: false,
    startEndListCollapsed: false,
    startEndStatsPayload: null,
    startEndStatsSourceUrl: "-",
    startEndCurrentType: "start",
    startEndRawTopEdges: [],
    startEndEdgeCountByBaseId: new Map(),
    startEndHighlightLayer: null,
    startEndHighlightFeatureCount: 0,
    startEndHighlightVisible: true,
    vinRouteLayer: null,
    vinStartEndLayer: null,
    vinQuery: "",
    vinRoutes: [],
    vinRouteCursor: 0,
    vinSourceFile: "-",
    vinMergeEnabled: true,
    panelZSeed: 1100,
  };

  function setText(el, text) {
    if (el) {
      el.textContent = text;
    }
  }

  function bringLayerToFront(layer) {
    if (!layer) {
      return;
    }
    if (typeof layer.bringToFront === "function") {
      layer.bringToFront();
      return;
    }
    if (typeof layer.eachLayer === "function") {
      layer.eachLayer((child) => {
        bringLayerToFront(child);
      });
    }
  }

  function bringPanelToFront(panelEl) {
    if (!panelEl) {
      return;
    }
    state.panelZSeed += 1;
    panelEl.style.zIndex = String(state.panelZSeed);
  }

  function setPanelPosition(panelEl, left, top) {
    if (!panelEl) {
      return;
    }
    panelEl.style.left = `${Math.round(left)}px`;
    panelEl.style.top = `${Math.round(top)}px`;
    panelEl.style.right = "auto";
  }

  function makePanelDraggable(panelEl, headerEl) {
    if (!panelEl || !headerEl) {
      return;
    }

    let dragging = false;
    let pointerOffsetX = 0;
    let pointerOffsetY = 0;

    const onMouseMove = (event) => {
      if (!dragging) {
        return;
      }
      const panelRect = panelEl.getBoundingClientRect();
      const maxLeft = Math.max(0, window.innerWidth - panelRect.width);
      const maxTop = Math.max(0, window.innerHeight - panelRect.height);

      const nextLeft = Math.max(0, Math.min(maxLeft, event.clientX - pointerOffsetX));
      const nextTop = Math.max(0, Math.min(maxTop, event.clientY - pointerOffsetY));
      setPanelPosition(panelEl, nextLeft, nextTop);
    };

    const stopDrag = () => {
      dragging = false;
      document.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mouseup", stopDrag);
    };

    headerEl.addEventListener("mousedown", (event) => {
      // Allow regular click behaviors for buttons in header.
      const target = event.target;
      if (target instanceof Element && target.closest("button,input,select,label,a")) {
        return;
      }

      const panelRect = panelEl.getBoundingClientRect();
      dragging = true;
      pointerOffsetX = event.clientX - panelRect.left;
      pointerOffsetY = event.clientY - panelRect.top;
      bringPanelToFront(panelEl);
      document.addEventListener("mousemove", onMouseMove);
      document.addEventListener("mouseup", stopDrag);
      event.preventDefault();
    });
  }

  function applyDefaultPanelLayout() {
    if (!statsPanelEl || !startEndPanelEl || !vinPanelEl) {
      return;
    }

    const gap = 12;
    const margin = 12;
    const statsRect = statsPanelEl.getBoundingClientRect();

    const statsX = Math.max(margin, window.innerWidth - statsRect.width - margin);
    const statsY = margin;
    setPanelPosition(statsPanelEl, statsX, statsY);

    // Keep the second panel in a vertical stack by default.
    const startEndX = statsX;
    const startEndY = statsY + statsRect.height + gap;
    setPanelPosition(startEndPanelEl, startEndX, startEndY);
    const startEndRect = startEndPanelEl.getBoundingClientRect();
    const vinX = statsX;
    const vinY = startEndY + startEndRect.height + gap;
    setPanelPosition(vinPanelEl, vinX, vinY);
    bringPanelToFront(vinPanelEl);
    bringPanelToFront(startEndPanelEl);
    bringPanelToFront(statsPanelEl);
  }

  function getBaseFeatureStyle(feature) {
    const osmType = feature && feature.properties ? feature.properties.osm_type : null;
    if (osmType === "relation") {
      return { color: "#f59e0b", weight: 1, opacity: 0.8 };
    }
    return { color: "#0ea5e9", weight: 1.2, opacity: 0.9 };
  }

  function normalizeEdgeId(edgeId) {
    if (typeof edgeId !== "string") {
      return null;
    }
    let token = edgeId.trim();
    if (!token) {
      return null;
    }
    if (token.startsWith("-")) {
      token = token.slice(1);
    }
    token = token.split("#", 1)[0];
    return token || null;
  }

  function clamp01(value) {
    if (!Number.isFinite(value)) {
      return 0;
    }
    return Math.max(0, Math.min(1, value));
  }

  function toRgbString(rgb) {
    return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
  }

  function interpolateRgb(startRgb, endRgb, t) {
    const ratio = clamp01(t);
    return [
      Math.round(startRgb[0] + (endRgb[0] - startRgb[0]) * ratio),
      Math.round(startRgb[1] + (endRgb[1] - startRgb[1]) * ratio),
      Math.round(startRgb[2] + (endRgb[2] - startRgb[2]) * ratio),
    ];
  }

  function getCountRatioByRange(count, minValue, maxValue) {
    const c = Math.max(0, Number(count) || 0);
    const minC = Math.max(0, Number(minValue) || 0);
    const maxC = Math.max(minC, Number(maxValue) || 1);
    const denom = Math.log1p(maxC) - Math.log1p(minC);
    if (denom <= 1e-9) {
      return 1;
    }
    return clamp01((Math.log1p(c) - Math.log1p(minC)) / denom);
  }

  function getCountColorByRange(count, minValue, maxValue) {
    // Low count -> vivid purple; high count -> deep purple.
    // Keep the whole range highlighted while still encoding count differences.
    const light = [170, 82, 255];
    const dark = [70, 16, 130];
    return toRgbString(interpolateRgb(light, dark, getCountRatioByRange(count, minValue, maxValue)));
  }

  function getFeatureBaseId(feature) {
    if (!feature || !feature.properties || feature.properties.osm_id == null) {
      return null;
    }
    return String(feature.properties.osm_id).replace(/^-/, "");
  }

  function hasMatchedEdge(feature) {
    const baseId = getFeatureBaseId(feature);
    return !!(baseId && state.edgeCountByBaseId.has(baseId));
  }

  function getHighlightFeatureStyle(feature) {
    const baseId = getFeatureBaseId(feature);
    if (!baseId) {
      return { color: "rgba(0,0,0,0)", weight: 0, opacity: 0 };
    }
    const match = state.edgeCountByBaseId.get(baseId);
    if (!match) {
      return { color: "rgba(0,0,0,0)", weight: 0, opacity: 0 };
    }

    const ratio = getCountRatioByRange(match.count, state.minHighlightCount, state.maxHighlightCount);
    return {
      color: getCountColorByRange(match.count, state.minHighlightCount, state.maxHighlightCount),
      weight: 2 + ratio * 4,
      opacity: 1.0,
    };
  }

  function bindEdgePopup(layer, feature) {
    const baseId = getFeatureBaseId(feature);
    if (!baseId) {
      if (typeof layer.unbindPopup === "function") {
        layer.unbindPopup();
      }
      return;
    }

    const match = state.edgeCountByBaseId.get(baseId);
    if (!match) {
      if (typeof layer.unbindPopup === "function") {
        layer.unbindPopup();
      }
      return;
    }

    const rawPreview = match.rawIds.slice(0, 5).join(", ");
    layer.bindPopup(
      `<div><b>edge id</b>: ${baseId}</div><div><b>count</b>: ${match.count}</div><div><b>raw ids</b>: ${rawPreview || "-"}</div>`
    );
  }

  function countMatchedFeatures() {
    return state.highlightFeatureCount;
  }

  function updateBasemapStatus() {
    if (!state.basemapGeojson) {
      return;
    }
    const visibleText = state.basemapVisible ? "显示" : "隐藏";
    setText(dataEl, `底图状态：已加载 (${state.basemapFeatureCount} features, 当前${visibleText})`);
  }

  function setBasemapVisible(visible) {
    state.basemapVisible = !!visible;
    if (!state.basemapLayer) {
      return;
    }

    const onMap = map.hasLayer(state.basemapLayer);
    if (state.basemapVisible && !onMap) {
      state.basemapLayer.addTo(map);
    } else if (!state.basemapVisible && onMap) {
      map.removeLayer(state.basemapLayer);
    }

    // Keep highlights above basemap when both are visible.
    if (state.highlightVisible && state.highlightLayer && map.hasLayer(state.highlightLayer)) {
      state.highlightLayer.bringToFront();
    }
    if (
      state.startEndHighlightVisible &&
      state.startEndHighlightLayer &&
      map.hasLayer(state.startEndHighlightLayer)
    ) {
      state.startEndHighlightLayer.bringToFront();
    }
    updateBasemapStatus();
  }

  function setHighlightVisible(visible) {
    state.highlightVisible = !!visible;
    if (!state.highlightLayer) {
      return;
    }

    const onMap = map.hasLayer(state.highlightLayer);
    if (state.highlightVisible && !onMap) {
      state.highlightLayer.addTo(map);
      state.highlightLayer.bringToFront();
    } else if (!state.highlightVisible && onMap) {
      map.removeLayer(state.highlightLayer);
    }
  }

  function rebuildHighlightLayer() {
    if (!state.basemapGeojson || !Array.isArray(state.basemapGeojson.features)) {
      state.highlightFeatureCount = 0;
      return;
    }

    const features = state.basemapGeojson.features.filter((feature) => hasMatchedEdge(feature));
    state.highlightFeatureCount = features.length;

    if (state.highlightLayer && map.hasLayer(state.highlightLayer)) {
      map.removeLayer(state.highlightLayer);
    }

    state.highlightLayer = L.geoJSON(
      { type: "FeatureCollection", features: features },
      {
        style: function styleHighlight(feature) {
          return getHighlightFeatureStyle(feature);
        },
        onEachFeature: function onEachHighlightFeature(feature, featureLayer) {
          bindEdgePopup(featureLayer, feature);
        },
      }
    );

    if (state.highlightVisible) {
      state.highlightLayer.addTo(map);
      state.highlightLayer.bringToFront();
    }
  }

  function setPanelCollapsed(collapsed) {
    state.panelCollapsed = !!collapsed;
    if (statsPanelEl) {
      statsPanelEl.classList.toggle("is-collapsed", state.panelCollapsed);
    }
    if (togglePanelBtnEl) {
      togglePanelBtnEl.textContent = state.panelCollapsed ? "展开" : "收起";
    }
  }

  function setListCollapsed(collapsed) {
    state.listCollapsed = !!collapsed;
    if (statsPanelEl) {
      statsPanelEl.classList.toggle("list-collapsed", state.listCollapsed);
    }
    if (toggleListBtnEl) {
      toggleListBtnEl.textContent = state.listCollapsed ? "展开列表" : "收起";
    }
  }

  function setVinPanelCollapsed(collapsed) {
    state.vinPanelCollapsed = !!collapsed;
    if (vinPanelEl) {
      vinPanelEl.classList.toggle("is-collapsed", state.vinPanelCollapsed);
    }
    if (toggleVinPanelBtnEl) {
      toggleVinPanelBtnEl.textContent = state.vinPanelCollapsed ? "展开" : "收起";
    }
  }

  function setStartEndPanelCollapsed(collapsed) {
    state.startEndPanelCollapsed = !!collapsed;
    if (startEndPanelEl) {
      startEndPanelEl.classList.toggle("is-collapsed", state.startEndPanelCollapsed);
    }
    if (toggleStartEndPanelBtnEl) {
      toggleStartEndPanelBtnEl.textContent = state.startEndPanelCollapsed ? "展开" : "收起";
    }
  }

  function setStartEndListCollapsed(collapsed) {
    state.startEndListCollapsed = !!collapsed;
    if (startEndPanelEl) {
      startEndPanelEl.classList.toggle("list-collapsed", state.startEndListCollapsed);
    }
    if (toggleStartEndListBtnEl) {
      toggleStartEndListBtnEl.textContent = state.startEndListCollapsed ? "展开列表" : "收起";
    }
  }

  function getStartEndTypeLabel(typeKey) {
    if (typeKey === "start") return "起点";
    if (typeKey === "end") return "终点";
    return "起点或终点";
  }

  function getStartEndGroup(typeKey) {
    if (!state.startEndStatsPayload || typeof state.startEndStatsPayload !== "object") {
      return null;
    }
    const group = state.startEndStatsPayload[typeKey];
    if (!group || typeof group !== "object") {
      return null;
    }
    return group;
  }

  function hasStartEndMatchedEdge(feature) {
    const baseId = getFeatureBaseId(feature);
    return !!(baseId && state.startEndEdgeCountByBaseId.has(baseId));
  }

  function getStartEndHighlightFeatureStyle(feature) {
    const baseId = getFeatureBaseId(feature);
    if (!baseId) {
      return { color: "rgba(0,0,0,0)", weight: 0, opacity: 0 };
    }
    const match = state.startEndEdgeCountByBaseId.get(baseId);
    if (!match) {
      return { color: "rgba(0,0,0,0)", weight: 0, opacity: 0 };
    }

    return {
      color: START_END_HIGHLIGHT_COLOR,
      weight: 4,
      opacity: 1.0,
    };
  }

  function bindStartEndPopup(layer, feature) {
    const baseId = getFeatureBaseId(feature);
    if (!baseId) {
      if (typeof layer.unbindPopup === "function") {
        layer.unbindPopup();
      }
      return;
    }

    const match = state.startEndEdgeCountByBaseId.get(baseId);
    if (!match) {
      if (typeof layer.unbindPopup === "function") {
        layer.unbindPopup();
      }
      return;
    }

    layer.bindPopup(
      `<div><b>ranking_type</b>: ${getStartEndTypeLabel(
        state.startEndCurrentType
      )}</div><div><b>edge id</b>: ${baseId}</div><div><b>count</b>: ${match.count}</div>`
    );
  }

  function rebuildStartEndHighlightLayer() {
    if (!state.basemapGeojson || !Array.isArray(state.basemapGeojson.features)) {
      state.startEndHighlightFeatureCount = 0;
      return;
    }

    const features = state.basemapGeojson.features.filter((feature) => hasStartEndMatchedEdge(feature));
    state.startEndHighlightFeatureCount = features.length;

    if (state.startEndHighlightLayer && map.hasLayer(state.startEndHighlightLayer)) {
      map.removeLayer(state.startEndHighlightLayer);
    }

    state.startEndHighlightLayer = L.geoJSON(
      { type: "FeatureCollection", features },
      {
        style: function styleStartEnd(feature) {
          return getStartEndHighlightFeatureStyle(feature);
        },
        onEachFeature: function onEachStartEnd(feature, featureLayer) {
          bindStartEndPopup(featureLayer, feature);
        },
      }
    );

    if (state.startEndHighlightVisible) {
      state.startEndHighlightLayer.addTo(map);
      state.startEndHighlightLayer.bringToFront();
    }
  }

  function setStartEndHighlightVisible(visible) {
    state.startEndHighlightVisible = !!visible;
    if (!state.startEndHighlightLayer) {
      return;
    }

    const onMap = map.hasLayer(state.startEndHighlightLayer);
    if (state.startEndHighlightVisible && !onMap) {
      state.startEndHighlightLayer.addTo(map);
      state.startEndHighlightLayer.bringToFront();
    } else if (!state.startEndHighlightVisible && onMap) {
      map.removeLayer(state.startEndHighlightLayer);
    }
  }

  function refreshStartEndSummary() {
    const group = getStartEndGroup(state.startEndCurrentType);
    const groupResultCount = group && Number.isFinite(group.result_edge_count)
      ? group.result_edge_count
      : state.startEndRawTopEdges.length;
    setText(
      startEndSummaryEl,
      `文件: ${state.startEndStatsSourceUrl} | 类型: ${getStartEndTypeLabel(
        state.startEndCurrentType
      )} | top_edges: ${state.startEndRawTopEdges.length} | 结果边: ${groupResultCount} | 可映射边: ${
        state.startEndEdgeCountByBaseId.size
      } | 地图匹配边: ${state.startEndHighlightFeatureCount} | 高亮: ${
        state.startEndHighlightVisible ? "开" : "关"
      }`
    );
  }

  function applyStartEndType(typeKey) {
    const nextType = typeKey === "start" || typeKey === "end" || typeKey === "start_or_end"
      ? typeKey
      : "start";
    state.startEndCurrentType = nextType;

    const group = getStartEndGroup(nextType);
    const topEdges = group && Array.isArray(group.top_edges) ? group.top_edges : [];
    state.startEndRawTopEdges = topEdges;

    const { mapping } = buildEdgeMapping(topEdges);
    state.startEndEdgeCountByBaseId = mapping;

    renderStartEndRows(topEdges, 200);
    rebuildStartEndHighlightLayer();
    setStartEndHighlightVisible(state.startEndHighlightVisible);
    refreshStartEndSummary();
  }

  const map = L.map("map", {
    preferCanvas: true,
    zoomControl: true,
  }).setView([31.285, 121.245], 12);

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  }).addTo(map);

  const vinRouteController = createVinRouteController({
    map,
    state,
    vinMergeToggleEl,
    vinStatusEl,
    vinRouteDataListEl,
    nextVinRouteBtnEl,
    setText,
    bringLayerToFront,
  });

  async function checkHealth() {
    try {
      const r = await fetch("/api/health");
      if (!r.ok) {
        throw new Error(`HTTP ${r.status}`);
      }
      const payload = await r.json();
      setText(healthEl, `后端状态：正常 (ok=${payload.ok})`);
    } catch (err) {
      setText(healthEl, `后端状态：失败 (${err.message || err})`);
    }
  }

  async function loadBasemap() {
    setText(dataEl, "底图状态：加载 basemap.geojson 中...");
    try {
      const r = await fetch("/data/basemap.geojson");
      if (!r.ok) {
        throw new Error(`HTTP ${r.status}`);
      }
      const geojson = await r.json();
      state.basemapGeojson = geojson;
      const layer = L.geoJSON(geojson, {
        style: function styleByData(feature) {
          return getBaseFeatureStyle(feature);
        },
      }).addTo(map);
      state.basemapLayer = layer;
      state.basemapFeatureCount = Array.isArray(geojson.features) ? geojson.features.length : 0;

      const bounds = layer.getBounds();
      if (bounds.isValid()) {
        map.fitBounds(bounds, { padding: [24, 24] });
      }
      state.basemapVisible = !!(basemapToggleEl && basemapToggleEl.checked);
      setBasemapVisible(state.basemapVisible);
      rebuildHighlightLayer();
      rebuildStartEndHighlightLayer();
    } catch (err) {
      setText(dataEl, `底图状态：加载失败 (${err.message || err})`);
    }
  }

  function renderStatsRows(listEl, topEdges, limit, minCount, maxCount) {
    if (!listEl) {
      return;
    }
    listEl.innerHTML = "";

    const rows = topEdges.slice(0, limit);
    if (rows.length === 0) {
      const empty = document.createElement("div");
      empty.className = "stats-empty";
      empty.textContent = "没有可显示的 edge。";
      listEl.appendChild(empty);
      return;
    }

    rows.forEach((item) => {
      const row = document.createElement("div");
      row.className = "stats-row";
      const countValue = Number(item.count);
      const ratio = getCountRatioByRange(countValue, minCount, maxCount);
      const color = getCountColorByRange(countValue, minCount, maxCount);
      row.style.borderLeft = `4px solid ${color}`;
      row.style.paddingLeft = "6px";
      row.style.background = `rgba(255,255,255,${0.03 + ratio * 0.12})`;

      const edge = document.createElement("span");
      edge.className = "stats-edge";
      edge.textContent = item.edge_id;

      const count = document.createElement("span");
      count.className = "stats-count";
      count.textContent = String(item.count);
      count.style.color = color;
      count.style.fontWeight = "600";

      row.appendChild(edge);
      row.appendChild(count);
      listEl.appendChild(row);
    });
  }

  function renderStartEndRows(topEdges, limit) {
    if (!startEndListEl) {
      return;
    }
    startEndListEl.innerHTML = "";

    const rows = topEdges.slice(0, limit);
    if (rows.length === 0) {
      const empty = document.createElement("div");
      empty.className = "stats-empty";
      empty.textContent = "没有可显示的 edge。";
      startEndListEl.appendChild(empty);
      return;
    }

    rows.forEach((item) => {
      const row = document.createElement("div");
      row.className = "stats-row";
      row.style.borderLeft = `4px solid ${START_END_HIGHLIGHT_COLOR}`;
      row.style.paddingLeft = "6px";
      row.style.background = "rgba(139, 0, 0, 0.12)";

      const edge = document.createElement("span");
      edge.className = "stats-edge";
      edge.textContent = item.edge_id;

      const count = document.createElement("span");
      count.className = "stats-count";
      count.textContent = String(item.count);
      count.style.color = START_END_HIGHLIGHT_COLOR;
      count.style.fontWeight = "600";

      row.appendChild(edge);
      row.appendChild(count);
      startEndListEl.appendChild(row);
    });
  }

  function buildEdgeMapping(topEdges) {
    const mapping = new Map();
    let maxCount = 0;
    let minCount = Number.POSITIVE_INFINITY;

    topEdges.forEach((item) => {
      const rawEdgeId = String(item.edge_id || "").trim();
      const countValue = Number(item.count);
      if (!rawEdgeId || !Number.isFinite(countValue)) {
        return;
      }

      const baseId = normalizeEdgeId(rawEdgeId);
      if (!baseId) {
        return;
      }

      const existing = mapping.get(baseId);
      if (existing) {
        existing.count += countValue;
        if (existing.rawIds.length < 8) {
          existing.rawIds.push(rawEdgeId);
        }
      } else {
        mapping.set(baseId, { count: countValue, rawIds: [rawEdgeId] });
      }
    });

    mapping.forEach((value) => {
      if (value.count > maxCount) {
        maxCount = value.count;
      }
      if (value.count < minCount) {
        minCount = value.count;
      }
    });

    if (!Number.isFinite(minCount)) {
      minCount = 0;
    }
    if (maxCount <= 0) {
      maxCount = 1;
    }

    return { mapping, maxCount, minCount };
  }

  function resolveStatsUrl(fileName) {
    const text = String(fileName || "").trim();
    if (!text) {
      return "";
    }
    if (text.startsWith("http://") || text.startsWith("https://") || text.startsWith("/")) {
      return text;
    }
    return `/analysis/${text}`;
  }

  async function loadEdgeFrequencyStatsByFileName(fileName) {
    const url = resolveStatsUrl(fileName);
    if (!url) {
      setText(statsMetaEl, "统计状态：请输入文件名。");
      return;
    }

    setText(statsMetaEl, `统计状态：加载 ${url} 中...`);
    try {
      const r = await fetch(url);
      if (!r.ok) {
        throw new Error(`HTTP ${r.status}`);
      }

      const payload = await r.json();
      const topEdges = Array.isArray(payload.top_edges) ? payload.top_edges : [];
      state.rawTopEdges = topEdges;

      const { mapping, maxCount, minCount } = buildEdgeMapping(topEdges);
      state.edgeCountByBaseId = mapping;
      state.minHighlightCount = minCount;
      state.maxHighlightCount = maxCount;
      state.highlightVisible = !!(highlightToggleEl && highlightToggleEl.checked);
      state.statsSourceUrl = url;

      rebuildHighlightLayer();
      setHighlightVisible(state.highlightVisible);
      const matchedFeatures = countMatchedFeatures();

      setText(statsMetaEl, "统计状态：已加载");
      setText(
        statsSummaryEl,
        `文件: ${url} | top_edges: ${topEdges.length} | 可映射边: ${mapping.size} | count范围: ${minCount}~${maxCount} | 地图匹配边: ${matchedFeatures} | 高亮: ${state.highlightVisible ? "开" : "关"}`
      );

      renderStatsRows(statsListEl, topEdges, 200, minCount, maxCount);
    } catch (err) {
      setText(statsMetaEl, `统计状态：加载失败 (${err.message || err})`);
      setText(statsSummaryEl, "地图高亮：未应用");
      renderStatsRows(statsListEl, [], 0, 0, 1);
    }
  }

  async function loadStartEndStatsByFileName(fileName) {
    const url = resolveStatsUrl(fileName);
    if (!url) {
      setText(startEndMetaEl, "统计状态：请输入文件名。");
      return;
    }

    setText(startEndMetaEl, `统计状态：加载 ${url} 中...`);
    try {
      const r = await fetch(url);
      if (!r.ok) {
        throw new Error(`HTTP ${r.status}`);
      }

      const payload = await r.json();
      if (!payload || typeof payload !== "object") {
        throw new Error("JSON 格式无效");
      }

      state.startEndStatsPayload = payload;
      state.startEndStatsSourceUrl = url;
      state.startEndHighlightVisible = !!(startEndHighlightToggleEl && startEndHighlightToggleEl.checked);

      const selectedType = startEndTypeSelectEl ? startEndTypeSelectEl.value : "start";
      applyStartEndType(selectedType);

      setText(startEndMetaEl, "统计状态：已加载");
    } catch (err) {
      setText(startEndMetaEl, `统计状态：加载失败 (${err.message || err})`);
      setText(startEndSummaryEl, "地图高亮：未应用");
      renderStartEndRows([], 0);
    }
  }

  if (applyStatsBtnEl) {
    applyStatsBtnEl.addEventListener("click", () => {
      const fileName = statsFileNameInputEl ? statsFileNameInputEl.value : "";
      loadEdgeFrequencyStatsByFileName(fileName);
    });
  }

  if (highlightToggleEl) {
    highlightToggleEl.addEventListener("change", () => {
      setHighlightVisible(!!highlightToggleEl.checked);
      const matchedFeatures = countMatchedFeatures();
      setText(
        statsSummaryEl,
        `文件: ${state.statsSourceUrl} | top_edges: ${state.rawTopEdges.length} | 可映射边: ${state.edgeCountByBaseId.size} | 地图匹配边: ${matchedFeatures} | 高亮: ${state.highlightVisible ? "开" : "关"}`
      );
    });
  }

  if (applyStartEndBtnEl) {
    applyStartEndBtnEl.addEventListener("click", () => {
      const fileName = startEndFileNameInputEl ? startEndFileNameInputEl.value : "";
      loadStartEndStatsByFileName(fileName);
    });
  }

  if (startEndHighlightToggleEl) {
    startEndHighlightToggleEl.addEventListener("change", () => {
      setStartEndHighlightVisible(!!startEndHighlightToggleEl.checked);
      refreshStartEndSummary();
    });
  }

  if (startEndTypeSelectEl) {
    startEndTypeSelectEl.addEventListener("change", () => {
      applyStartEndType(startEndTypeSelectEl.value);
    });
  }

  if (basemapToggleEl) {
    basemapToggleEl.addEventListener("change", () => {
      setBasemapVisible(!!basemapToggleEl.checked);
    });
  }

  if (drawVinBtnEl) {
    drawVinBtnEl.addEventListener("click", () => {
      const vin = vinInputEl ? vinInputEl.value : "";
      vinRouteController.draw(vin);
    });
  }

  if (nextVinRouteBtnEl) {
    nextVinRouteBtnEl.addEventListener("click", () => {
      vinRouteController.next();
    });
  }

  if (vinInputEl) {
    vinInputEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        vinRouteController.draw(vinInputEl.value);
      }
    });
  }

  if (clearVinBtnEl) {
    clearVinBtnEl.addEventListener("click", () => {
      vinRouteController.clear();
    });
  }

  if (vinMergeToggleEl) {
    vinMergeToggleEl.addEventListener("change", () => {
      vinRouteController.redrawCurrent();
    });
  }

  if (togglePanelBtnEl) {
    togglePanelBtnEl.addEventListener("click", () => {
      setPanelCollapsed(!state.panelCollapsed);
    });
  }

  if (toggleListBtnEl) {
    toggleListBtnEl.addEventListener("click", () => {
      setListCollapsed(!state.listCollapsed);
    });
  }

  if (toggleStartEndPanelBtnEl) {
    toggleStartEndPanelBtnEl.addEventListener("click", () => {
      setStartEndPanelCollapsed(!state.startEndPanelCollapsed);
    });
  }

  if (toggleStartEndListBtnEl) {
    toggleStartEndListBtnEl.addEventListener("click", () => {
      setStartEndListCollapsed(!state.startEndListCollapsed);
    });
  }

  if (toggleVinPanelBtnEl) {
    toggleVinPanelBtnEl.addEventListener("click", () => {
      setVinPanelCollapsed(!state.vinPanelCollapsed);
    });
  }

  makePanelDraggable(statsPanelEl, statsPanelHeaderEl);
  makePanelDraggable(startEndPanelEl, startEndPanelHeaderEl);
  makePanelDraggable(vinPanelEl, vinPanelHeaderEl);

  setPanelCollapsed(true);
  setListCollapsed(false);
  setStartEndPanelCollapsed(true);
  setStartEndListCollapsed(false);
  setVinPanelCollapsed(true);
  applyDefaultPanelLayout();

  checkHealth();
  loadBasemap();
})();
