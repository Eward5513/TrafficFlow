"""Device selection.

``auto`` uses an idle visible GPU: any GPU with a compute process is
excluded, one of the rest is chosen at random, and training does not start
when none are idle. Memory is not consulted. The draw uses
``random.SystemRandom`` so it does not consume the seeded training RNG.
Indexes returned are logical (after ``CUDA_VISIBLE_DEVICES``); the physical
index is reported alongside.
"""

from __future__ import annotations

import os
import random
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


def _nvidia_smi(query: str) -> list[str]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"nvidia-smi failed: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or f"exit {completed.returncode}"
        raise RuntimeError(f"nvidia-smi failed: {detail}")
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def busy_physical_gpus() -> set[int]:
    """Physical indexes that currently have at least one compute process."""
    uuid_to_index: dict[str, int] = {}
    for line in _nvidia_smi("gpu=index,uuid"):
        index_text, uuid = (part.strip() for part in line.split(",", 1))
        uuid_to_index[uuid] = int(index_text)
    busy: set[int] = set()
    for line in _nvidia_smi("compute-apps=gpu_uuid,pid"):
        uuid, _pid = (part.strip() for part in line.split(",", 1))
        index = uuid_to_index.get(uuid)
        if index is not None:
            busy.add(index)
    return busy


def idle_logical_gpus(logical_to_physical: dict[int, int], busy: set[int]) -> list[int]:
    return sorted(logical for logical, physical in logical_to_physical.items() if physical not in busy)


def resolve_device(name: str, *, rng: random.Random | None = None) -> tuple[torch.device, dict[str, object]]:
    text = str(name).strip().lower()
    if text == "cpu":
        return torch.device("cpu"), {"physical_gpu_index": None, "selection": "cpu"}
    if not torch.cuda.is_available():
        raise RuntimeError(f"device {name!r} requested but CUDA is unavailable")
    visible = _visible_physical()
    logical_to_physical = {i: (visible[i] if visible is not None else i) for i in range(torch.cuda.device_count())}
    if text == "auto":
        if visible is None and os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise RuntimeError("device auto cannot parse CUDA_VISIBLE_DEVICES as GPU indexes")
        busy = busy_physical_gpus()
        idle = idle_logical_gpus(logical_to_physical, busy)
        if not idle:
            visible_physical = [logical_to_physical[i] for i in sorted(logical_to_physical)]
            raise RuntimeError(
                "no idle GPU; refusing to start. "
                f"visible={visible_physical} busy={sorted(busy)}"
            )
        logical = (rng or random.SystemRandom()).choice(idle)
        physical = logical_to_physical[logical]
        print(
            f"[MSTT] device auto selected physical GPU {physical} (cuda:{logical}); "
            f"idle={[logical_to_physical[i] for i in idle]}",
            flush=True,
        )
        return torch.device(f"cuda:{logical}"), {
            "physical_gpu_index": physical,
            "selection": "auto_idle",
            "idle_physical_gpus": [logical_to_physical[i] for i in idle],
        }
    if text.isdigit():
        text = f"cuda:{text}"
    device = torch.device(text)
    logical = device.index if device.index is not None else 0
    if logical not in logical_to_physical:
        raise RuntimeError(f"device {name!r} is outside the visible GPUs {sorted(logical_to_physical)}")
    physical = logical_to_physical[logical]
    return torch.device(f"cuda:{logical}"), {"physical_gpu_index": physical, "selection": "explicit"}
