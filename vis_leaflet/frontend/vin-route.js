"use strict";

function createVinRouteController(params) {
  const map = params.map;
  const state = params.state;
  const vinMergeToggleEl = params.vinMergeToggleEl;
  const vinStatusEl = params.vinStatusEl;
  const vinRouteDataListEl = params.vinRouteDataListEl;
  const nextVinRouteBtnEl = params.nextVinRouteBtnEl;
  const setText = params.setText;
  const bringLayerToFront = params.bringLayerToFront;

  function renderVinRouteData(routes, activeRouteIndex) {
    if (!vinRouteDataListEl) {
      return;
    }
    vinRouteDataListEl.innerHTML = "";
    const list = Array.isArray(routes) ? routes : [];
    if (list.length === 0) {
      const empty = document.createElement("div");
      empty.className = "stats-empty";
      empty.textContent = "暂无轨迹明细。";
      vinRouteDataListEl.appendChild(empty);
      return;
    }

    const safeIndex = ((Number(activeRouteIndex) || 0) % list.length + list.length) % list.length;
    const route = list[safeIndex];
    const block = document.createElement("div");
    block.className = "vin-route-block";

    const title = document.createElement("div");
    title.className = "vin-route-title";
    title.textContent = `轨迹 #${safeIndex + 1} / ${list.length}`;
    block.appendChild(title);

    const edgeIds = Array.isArray(route && route.edge_ids) ? route.edge_ids : [];
    const times = Array.isArray(route && route.times) ? route.times : [];
    const rowCount = Math.max(edgeIds.length, times.length);
    if (rowCount === 0) {
      const empty = document.createElement("div");
      empty.className = "stats-empty";
      empty.textContent = "该轨迹没有可显示数据。";
      block.appendChild(empty);
    } else {
      for (let i = 0; i < rowCount; i += 1) {
        const row = document.createElement("div");
        row.className = "vin-route-row";

        const timeEl = document.createElement("span");
        timeEl.className = "vin-route-time";
        timeEl.textContent = String(times[i] || "-");

        const edgeEl = document.createElement("span");
        edgeEl.className = "vin-route-edge";
        edgeEl.textContent = String(edgeIds[i] || "-");

        row.appendChild(timeEl);
        row.appendChild(edgeEl);
        block.appendChild(row);
      }
    }

    vinRouteDataListEl.appendChild(block);
  }

  function setNextVinRouteEnabled(enabled) {
    if (!nextVinRouteBtnEl) {
      return;
    }
    nextVinRouteBtnEl.disabled = !enabled;
    nextVinRouteBtnEl.style.opacity = enabled ? "1" : "0.55";
  }

  function clear() {
    if (state.vinRouteLayer && map.hasLayer(state.vinRouteLayer)) {
      map.removeLayer(state.vinRouteLayer);
    }
    if (state.vinStartEndLayer && map.hasLayer(state.vinStartEndLayer)) {
      map.removeLayer(state.vinStartEndLayer);
    }
    state.vinRouteLayer = null;
    state.vinStartEndLayer = null;
    state.vinQuery = "";
    state.vinRoutes = [];
    state.vinRouteCursor = 0;
    state.vinSourceFile = "-";
    state.vinMergeEnabled = !!(vinMergeToggleEl && vinMergeToggleEl.checked);
    setText(vinStatusEl, "VIN 轨迹：未绘制");
    renderVinRouteData([], 0);
    setNextVinRouteEnabled(false);
  }

  function renderActiveRoute(fitBounds) {
    if (state.vinRouteLayer && map.hasLayer(state.vinRouteLayer)) {
      map.removeLayer(state.vinRouteLayer);
    }
    if (state.vinStartEndLayer && map.hasLayer(state.vinStartEndLayer)) {
      map.removeLayer(state.vinStartEndLayer);
    }
    state.vinRouteLayer = null;
    state.vinStartEndLayer = null;

    const routes = Array.isArray(state.vinRoutes) ? state.vinRoutes : [];
    if (routes.length === 0) {
      renderVinRouteData([], 0);
      setNextVinRouteEnabled(false);
      setText(vinStatusEl, `VIN 轨迹：VIN=${state.vinQuery} 未找到可用轨迹`);
      return;
    }

    state.vinRouteCursor = ((state.vinRouteCursor % routes.length) + routes.length) % routes.length;
    const route = routes[state.vinRouteCursor];
    const routeNo = state.vinRouteCursor + 1;

    const featuresByExactId = new Map();
    state.basemapGeojson.features.forEach((feature) => {
      const osmId = feature && feature.properties && feature.properties.osm_id != null
        ? String(feature.properties.osm_id).trim()
        : "";
      if (osmId && !featuresByExactId.has(osmId)) {
        featuresByExactId.set(osmId, feature);
      }
    });

    const rawEdgeIds = Array.isArray(route && route.edge_ids) ? route.edge_ids : [];
    const rawTimes = Array.isArray(route && route.times) ? route.times : [];
    const exactEdgeIds = rawEdgeIds.map((edgeId) => String(edgeId || "").trim()).filter((edgeId) => !!edgeId);
    const inputEdgeTotal = exactEdgeIds.length;
    const timestampTotal = rawTimes.length;

    const edgeRoleMap = new Map();
    function addEdgeRole(edgeId, role, timeText) {
      if (!edgeId) {
        return;
      }
      if (!edgeRoleMap.has(edgeId)) {
        edgeRoleMap.set(edgeId, []);
      }
      edgeRoleMap.get(edgeId).push({
        routeNo,
        role,
        time: timeText || "-",
      });
    }

    if (exactEdgeIds.length > 0) {
      addEdgeRole(exactEdgeIds[0], "起点", String(rawTimes[0] || "").trim());
      addEdgeRole(
        exactEdgeIds[exactEdgeIds.length - 1],
        "终点",
        String(rawTimes[exactEdgeIds.length - 1] || "").trim()
      );
    }

    const orderedFeatures = [];
    const lastIndex = exactEdgeIds.length - 1;
    exactEdgeIds.forEach((edgeId, index) => {
      const feature = featuresByExactId.get(edgeId);
      if (!feature) {
        return;
      }
      const timeText = String(rawTimes[index] || "").trim();
      const isStart = index === 0;
      const isEnd = index === lastIndex;
      orderedFeatures.push({
        type: "Feature",
        geometry: feature.geometry,
        properties: {
          ...(feature.properties || {}),
          route_no: routeNo,
          route_index: index + 1,
          route_edge_id: edgeId,
          route_time: timeText,
          route_is_start: isStart,
          route_is_end: isEnd,
        },
      });
    });

    const multiRoleBorderLayers = [];
    const multiRoleEdgeAdded = new Set();
    const routeLayer = L.geoJSON(
      { type: "FeatureCollection", features: orderedFeatures },
      {
        style: function styleVinRoute(feature) {
          const props = feature && feature.properties ? feature.properties : {};
          const edgeId = props.route_edge_id != null ? String(props.route_edge_id) : "";
          const isStart = !!props.route_is_start;
          const isEnd = !!props.route_is_end;

          if (!isStart && !isEnd) {
            return { color: "#1e3a8a", weight: 5, opacity: 0.95 };
          }

          const roles = edgeId && edgeRoleMap.has(edgeId) ? edgeRoleMap.get(edgeId) : [];
          const roleCount = roles.length;
          if (roleCount > 1) {
            return { color: "#22c55e", weight: 8, opacity: 0.98 };
          }
          if (isStart) {
            return { color: "#22c55e", weight: 8, opacity: 0.98 };
          }
          return { color: "#ef4444", weight: 8, opacity: 0.98 };
        },
        onEachFeature: function onEachVinRouteFeature(feature, layer) {
          const props = feature && feature.properties ? feature.properties : {};
          const routeNoText = props.route_no != null ? String(props.route_no) : "-";
          const routeIndex = props.route_index != null ? String(props.route_index) : "-";
          const edgeId = props.route_edge_id != null ? String(props.route_edge_id) : "-";
          const timeText = props.route_time ? String(props.route_time) : "-";
          const isStart = !!props.route_is_start;
          const isEnd = !!props.route_is_end;
          const roles = edgeId && edgeRoleMap.has(edgeId) ? edgeRoleMap.get(edgeId) : [];
          const roleCount = roles.length;
          const hasStartRole = roles.some((item) => item.role === "起点");
          const hasEndRole = roles.some((item) => item.role === "终点");
          const roleRows = roles
            .map((item) => `<div>轨迹#${item.routeNo} ${item.role} @ ${item.time || "-"}</div>`)
            .join("");

          if (roleCount > 1 && (isStart || isEnd) && !multiRoleEdgeAdded.has(edgeId)) {
            const borderLayer = L.geoJSON(feature, {
              style: { color: "#ef4444", weight: 12, opacity: 0.95 },
              interactive: false,
            });
            multiRoleBorderLayers.push(borderLayer);
            multiRoleEdgeAdded.add(edgeId);
          }

          layer.bindPopup(
            `<div><b>VIN</b>: ${state.vinQuery}</div><div><b>轨迹</b>: #${routeNoText}</div><div><b>序号</b>: ${routeIndex}</div><div><b>edge_id</b>: ${edgeId}</div><div><b>时间</b>: ${timeText}</div><div><b>当前角色</b>: ${
              isStart && isEnd ? "起点+终点" : isStart ? "起点" : isEnd ? "终点" : "普通段"
            }</div><div><b>该边角色</b>: ${
              roleCount > 1 ? "多角色(红框绿芯)" : hasStartRole ? "起点" : hasEndRole ? "终点" : "普通"
            }</div><div><b>该边所有起终点记录</b>:</div>${roleRows || "<div>-</div>"}`
          );
        },
      }
    );

    state.vinStartEndLayer =
      multiRoleBorderLayers.length > 0 ? L.layerGroup(multiRoleBorderLayers).addTo(map) : null;
    state.vinRouteLayer = routeLayer.addTo(map);
    bringLayerToFront(state.vinStartEndLayer);
    bringLayerToFront(state.vinRouteLayer);

    if (fitBounds) {
      const bounds = routeLayer.getBounds();
      if (bounds && bounds.isValid()) {
        map.fitBounds(bounds, { padding: [30, 30], maxZoom: 17 });
      }
    }

    renderVinRouteData(routes, state.vinRouteCursor);
    setNextVinRouteEnabled(routes.length > 1);
    setText(
      vinStatusEl,
      `VIN 轨迹：VIN=${state.vinQuery} | 当前=${routeNo}/${routes.length} | 模式=${
        state.vinMergeEnabled ? "合并" : "不合并"
      } | 来源=${state.vinSourceFile} | 当前轨迹输入边数=${inputEdgeTotal} | 匹配边数=${
        orderedFeatures.length
      } | 时间戳数=${timestampTotal}`
    );
  }

  async function draw(vin) {
    const queryVin = String(vin || "").trim();
    const mergeEnabled = !!(vinMergeToggleEl && vinMergeToggleEl.checked);
    if (!queryVin) {
      setText(vinStatusEl, "VIN 轨迹：请输入 VIN");
      return;
    }
    if (!state.basemapGeojson || !Array.isArray(state.basemapGeojson.features)) {
      setText(vinStatusEl, "VIN 轨迹：底图尚未加载完成");
      return;
    }

    setText(vinStatusEl, `VIN 轨迹：正在查询 VIN=${queryVin} ...`);
    try {
      const r = await fetch(
        `/api/route_by_vin?vin=${encodeURIComponent(queryVin)}&merge=${mergeEnabled ? "1" : "0"}`
      );
      const payload = await r.json();
      if (!r.ok || !payload.ok) {
        throw new Error(payload && payload.error ? payload.error : `HTTP ${r.status}`);
      }
      const routes = Array.isArray(payload.routes)
        ? payload.routes
        : [
            {
              edge_ids: Array.isArray(payload.edge_ids) ? payload.edge_ids : [],
              times: Array.isArray(payload.times) ? payload.times : [],
            },
          ];
      state.vinQuery = queryVin;
      state.vinRoutes = routes;
      state.vinRouteCursor = 0;
      state.vinSourceFile = payload.source_file || "-";
      state.vinMergeEnabled = mergeEnabled;
      renderActiveRoute(true);
    } catch (err) {
      setText(vinStatusEl, `VIN 轨迹：绘制失败 (${err.message || err})`);
      renderVinRouteData([], 0);
      setNextVinRouteEnabled(false);
    }
  }

  function next() {
    const routes = Array.isArray(state.vinRoutes) ? state.vinRoutes : [];
    if (routes.length === 0) {
      setText(vinStatusEl, "VIN 轨迹：请先查询 VIN");
      return;
    }
    state.vinRouteCursor = (state.vinRouteCursor + 1) % routes.length;
    renderActiveRoute(true);
  }

  setNextVinRouteEnabled(false);

  return {
    clear,
    draw,
    next,
    redrawCurrent: function redrawCurrent() {
      if (state.vinQuery) {
        draw(state.vinQuery);
      }
    },
  };
}
