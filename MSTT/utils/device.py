"""Device selection. ``auto`` picks the visible GPU with the most free memory.

Free memory is read from ``nvidia-smi`` so probing does not open a CUDA
context on every GPU. Indexes returned are logical (after
``CUDA_VISIBLE_DEVICES``); the physical index is reported alongside.
"""

from __future__ import annotations

import os
import subprocess

import torch


def _visible_physical() -> list[int] | None:
    text = os.environ.get("CUDA_VISIBLE_DEVICES")
    if text is None:
        return None
    items = [item.strip() for item in text.split(",") if item.strip()]
    try:
        return [int(item) for item in items]
    except ValueError:
        return None


def gpu_free_memory_mb() -> dict[int, float]:
    """Physical GPU index -> free MiB."""
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    free: dict[int, float] = {}
    for line in output.strip().splitlines():
        index, value = (part.strip() for part in line.split(","))
        free[int(index)] = float(value)
    return free


def resolve_device(name: str) -> tuple[torch.device, dict[str, object]]:
    text = str(name).strip().lower()
    if text == "cpu" or not torch.cuda.is_available():
        if text not in ("cpu", "auto"):
            raise RuntimeError(f"device {name!r} requested but CUDA is unavailable")
        return torch.device("cpu"), {"physical_gpu_index": None, "selection": "cpu"}
    visible = _visible_physical()
    if text == "auto":
        free = gpu_free_memory_mb()
        logical_to_physical = {i: (visible[i] if visible is not None else i) for i in range(torch.cuda.device_count())}
        ranked = sorted(logical_to_physical, key=lambda i: -free.get(logical_to_physical[i], 0.0))
        logical = ranked[0]
        return torch.device(f"cuda:{logical}"), {
            "physical_gpu_index": logical_to_physical[logical],
            "selection": "auto_max_free_memory",
            "free_memory_mb_at_selection": free.get(logical_to_physical[logical]),
        }
    if text.isdigit():
        text = f"cuda:{text}"
    device = torch.device(text)
    logical = device.index if device.index is not None else 0
    physical = visible[logical] if visible is not None and logical < len(visible) else logical
    return torch.device(f"cuda:{logical}"), {"physical_gpu_index": physical, "selection": "explicit"}
