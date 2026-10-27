"""Atomic UTF-8 / bytes writers for JSON, text, and binary artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from reimplementation.common.errors import ReimplementationError


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if os.path.getsize(tmp_name) <= 0 and len(data) > 0:
            raise ReimplementationError(f"temporary file is empty before replace: {tmp_name}")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    if not path.is_file():
        raise ReimplementationError(f"atomic replace did not create {path}")


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def atomic_write_npz(path: Path, **arrays: Any) -> None:
    """Write an NPZ without leaving a 0-byte destination.

    ``numpy.savez(filename)`` appends ``.npz`` when the path does not already
    end with that suffix. Passing a file object avoids that rewrite, so the
    mkstemp path is the archive that ``os.replace`` publishes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".npz",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        size = os.path.getsize(tmp_name)
        if size <= 0:
            raise ReimplementationError(f"temporary NPZ is empty: {tmp_name}")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        extra = Path(str(tmp_name) + ".npz")
        if extra.is_file() and extra != Path(tmp_name):
            try:
                extra.unlink()
            except OSError:
                pass
        raise
    if not path.is_file() or path.stat().st_size <= 0:
        raise ReimplementationError(f"atomic NPZ replace left an unreadable file: {path}")
