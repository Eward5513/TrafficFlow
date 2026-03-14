# vis_leaflet 最小可视化框架

一个零构建的前后端可视化示例：
- 前端：原生 `HTML/CSS/JS` + `Leaflet`（CDN）
- 后端：Python 标准库 `http.server`，仅提供一个心跳接口

## 目录结构

- `backend/server.py`：后端入口
- `frontend/index.html`：页面结构与 Leaflet 引入
- `frontend/main.js`：地图初始化、心跳请求、GeoJSON 加载
- `frontend/styles.css`：页面样式

## 运行方式

在仓库根目录执行：

```bash
python vis_leaflet/backend/server.py
```

默认监听：`http://localhost:8080`

可选环境变量：
- `HOST`（默认 `0.0.0.0`）
- `PORT`（默认 `8080`）

## 可用接口与资源

- `GET /api/health`：心跳测试
- `GET /`：前端页面
- `GET /frontend/*`：前端静态资源
- `GET /data/*`：仓库 `data` 目录静态文件

当前前端默认读取：
- `/data/basemap.geojson`

## 验证

1. 打开 `http://localhost:8080/`
2. 页面状态栏显示：
   - 后端状态正常
   - 底图加载成功
3. 直接访问 `http://localhost:8080/data/basemap.geojson`，应返回 GeoJSON 内容

说明：该示例默认按手动方式验证，不会自动启动或自动运行服务。
