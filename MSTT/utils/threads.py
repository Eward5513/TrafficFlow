"""Cap one MSTT process before PyTorch opens its default intra-op pool.

On this 128-core host an unset ``OMP_NUM_THREADS`` becomes 64 OpenMP workers.
Callers that set ``MSTT_NUM_THREADS`` get that pool, and never more than 15.
"""

from __future__ import annotations

import os

MAX_THREADS = 15
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def parse_thread_cap(raw: str | None, *, default: int | None = None) -> int | None:
    """Return a pool size in ``1..15``. An empty value uses ``default``."""
    if raw is None or str(raw).strip() == "":
        return default
    try:
        n = int(str(raw).strip())
    except ValueError as exc:
        raise SystemExit(f"MSTT_NUM_THREADS must be an integer from 1 to {MAX_THREADS}, got {raw!r}") from exc
    if n < 1 or n > MAX_THREADS:
        raise SystemExit(f"MSTT_NUM_THREADS must be an integer from 1 to {MAX_THREADS}, got {n}")
    return n


def apply_thread_cap() -> int | None:
    """Apply ``MSTT_NUM_THREADS`` before any parallel work. Unset leaves PyTorch alone."""
    n = parse_thread_cap(os.environ.get("MSTT_NUM_THREADS"))
    if n is None:
        return None
    text = str(n)
    for key in THREAD_ENV_KEYS:
        os.environ[key] = text
    import torch

    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return n
