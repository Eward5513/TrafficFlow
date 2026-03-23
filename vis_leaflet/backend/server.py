#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import mimetypes
import os
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

HERE = Path(__file__).resolve().parent
VIS_LEAFLET_DIR = HERE.parent
REPO_ROOT = VIS_LEAFLET_DIR.parent
FRONTEND_DIR = VIS_LEAFLET_DIR / "frontend"
DATA_DIR = REPO_ROOT / "data"
ANALYSIS_DIR = REPO_ROOT / "analysis"
ROUTE_BY_EDGE_PATH = ANALYSIS_DIR / "ocr_output" / "route_by_edge.txt"
ROUTE_BY_EDGE_NO_MERGE_PATH = ANALYSIS_DIR / "ocr_output" / "route_by_edge_no_merge.txt"


def _parse_route_tokens(tokens: list[str]) -> tuple[list[str], list[str]]:
    edge_ids: list[str] = []
    times: list[str] = []
    idx = 0
    while idx < len(tokens):
        current = tokens[idx]
        if ":" in current:
            if idx + 1 < len(tokens):
                times.append(current)
                edge_ids.append(tokens[idx + 1])
            idx += 2
        else:
            edge_ids.append(current)
            idx += 1
    return edge_ids, times


def _to_bool(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _load_routes_by_vin(vin: str, merged: bool) -> dict:
    source_path = ROUTE_BY_EDGE_PATH if merged else ROUTE_BY_EDGE_NO_MERGE_PATH
    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(f"file not found: {source_path}")

    query_vin = str(vin).strip()
    if not query_vin:
        return {
            "vin": "",
            "merged": merged,
            "source_file": source_path.name,
            "route_count": 0,
            "routes": [],
        }

    routes: list[dict] = []
    with source_path.open("r", encoding="utf-8") as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line:
                continue
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] != query_vin:
                continue
            edge_ids, times = _parse_route_tokens(tokens[1:])
            routes.append(
                {
                    "edge_ids": edge_ids,
                    "times": times,
                    "point_count": len(edge_ids),
                }
            )
            # merged file should have exactly one trajectory per VIN, no need to scan further
            if merged:
                break

    return {
        "vin": query_vin,
        "merged": merged,
        "source_file": source_path.name,
        "route_count": len(routes),
        "routes": routes,
    }


def _safe_join(base: Path, rel_path: str) -> Path | None:
    candidate = (base / rel_path).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError:
        return None
    return candidate


class AppHandler(BaseHTTPRequestHandler):
    server_version = "TrafficFlowLeaflet/0.1"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = unquote(parsed.path)
        query = parse_qs(parsed.query)

        if route == "/api/health":
            self._handle_health()
            return
        if route == "/api/route_by_vin":
            self._handle_route_by_vin(query)
            return

        if route == "/":
            self._serve_file(FRONTEND_DIR / "index.html")
            return

        if route.startswith("/frontend/"):
            rel = route.removeprefix("/frontend/")
            target = _safe_join(FRONTEND_DIR, rel)
            if target is None:
                self._write_json({"ok": False, "error": "invalid path"}, status=HTTPStatus.BAD_REQUEST)
                return
            self._serve_file(target)
            return

        if route.startswith("/data/"):
            rel = route.removeprefix("/data/")
            target = _safe_join(DATA_DIR, rel)
            if target is None:
                self._write_json({"ok": False, "error": "invalid path"}, status=HTTPStatus.BAD_REQUEST)
                return
            self._serve_file(target)
            return

        if route.startswith("/analysis/"):
            rel = route.removeprefix("/analysis/")
            target = _safe_join(ANALYSIS_DIR, rel)
            if target is None:
                self._write_json({"ok": False, "error": "invalid path"}, status=HTTPStatus.BAD_REQUEST)
                return
            self._serve_file(target)
            return

        self._write_json({"ok": False, "error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: object) -> None:
        # Keep concise request logs.
        print(f"[{self.log_date_time_string()}] {self.address_string()} - {format % args}")

    def _handle_health(self) -> None:
        payload = {
            "ok": True,
            "service": "vis_leaflet_backend",
            "time": datetime.now(timezone.utc).isoformat(),
        }
        self._write_json(payload, status=HTTPStatus.OK)

    def _handle_route_by_vin(self, query: dict[str, list[str]]) -> None:
        vin = (query.get("vin") or [""])[0].strip()
        merged = _to_bool((query.get("merge") or ["true"])[0], default=True)
        if not vin:
            self._write_json(
                {"ok": False, "error": "vin is required", "vin": "", "routes": [], "route_count": 0, "merged": merged},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        try:
            payload = _load_routes_by_vin(vin, merged=merged)
        except FileNotFoundError as err:
            self._write_json({"ok": False, "error": str(err)}, status=HTTPStatus.NOT_FOUND)
            return
        except Exception as err:  # noqa: BLE001
            self._write_json({"ok": False, "error": f"failed to parse route file: {err}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if payload["route_count"] <= 0:
            self._write_json({"ok": False, "error": f"vin not found: {vin}", **payload}, status=HTTPStatus.NOT_FOUND)
            return

        self._write_json({"ok": True, **payload}, status=HTTPStatus.OK)

    def _serve_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self._write_json({"ok": False, "error": f"file not found: {path.name}"}, status=HTTPStatus.NOT_FOUND)
            return

        content_type, _ = mimetypes.guess_type(str(path))
        if content_type is None:
            content_type = "application/octet-stream"

        raw = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _write_json(self, payload: dict, status: HTTPStatus) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))

    server = ThreadingHTTPServer((host, port), AppHandler)
    print(f"Serving vis_leaflet on http://{host}:{port}")
    print(f"Frontend dir: {FRONTEND_DIR}")
    print(f"Data dir: {DATA_DIR}")
    print(f"Analysis dir: {ANALYSIS_DIR}")
    server.serve_forever()


if __name__ == "__main__":
    main()
