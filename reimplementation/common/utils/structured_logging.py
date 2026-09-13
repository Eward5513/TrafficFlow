"""JSONL training logs. Each write is flushed immediately."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from reimplementation.common.utils.atomic_io import atomic_write_text


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def log(self, record: Mapping[str, Any]) -> None:
        payload = {"timestamp": utc_now(), **dict(record)}
        self._handle.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return str(value)


def rewrite_jsonl(path: Path, records: list[Mapping[str, Any]]) -> None:
    text = "".join(
        json.dumps({"timestamp": utc_now(), **dict(record)}, ensure_ascii=False, default=_json_default)
        + "\n"
        for record in records
    )
    atomic_write_text(path, text)
