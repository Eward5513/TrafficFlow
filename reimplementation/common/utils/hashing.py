"""SHA256 helpers for files and arrays."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

HASH_CHUNK = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_numpy(array: np.ndarray) -> str:
    payload = np.ascontiguousarray(array)
    return hashlib.sha256(payload.tobytes()).hexdigest()
