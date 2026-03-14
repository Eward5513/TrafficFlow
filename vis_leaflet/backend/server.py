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
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent
VIS_LEAFLET_DIR = HERE.parent
REPO_ROOT = VIS_LEAFLET_DIR.parent
FRONTEND_DIR = VIS_LEAFLET_DIR / "frontend"
DATA_DIR = REPO_ROOT / "data"
ANALYSIS_DIR = REPO_ROOT / "analysis"


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

        if route == "/api/health":
            self._handle_health()
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
