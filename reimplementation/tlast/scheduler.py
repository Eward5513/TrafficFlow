"""Epoch cosine schedule from ``reference/TLAST/run.py``.

``t_in_epochs=True``, so the per-batch ``step_update`` in the reference
trainer does not change the learning rate. After epoch ``i`` (1-based) the
reference calls ``scheduler.step(i)``. Warmup starts at ``1e-6``.
``decay_rate=0.1`` applies only on a later cycle; a 100-epoch run stays in
the first cycle while the weights are updated.
"""

from __future__ import annotations

import math

import torch


def author_cosine_lr(
    step: int,
    *,
    base_lr: float,
    t_initial: int,
    lr_min: float,
    decay_rate: float,
    warmup_t: int,
    warmup_lr_init: float,
) -> float:
    t = int(step)
    horizon = max(int(t_initial), 1)
    warmup = int(warmup_t)
    if warmup > 0 and t < warmup:
        return float(warmup_lr_init) + float(t) * (float(base_lr) - float(warmup_lr_init)) / float(warmup)
    cycle = t // horizon
    t_curr = t - horizon * cycle
    gamma = float(decay_rate) ** cycle
    floor = float(lr_min) * gamma
    peak = float(base_lr) * gamma
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * float(t_curr) / float(horizon)))


class AuthorCosineLRScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        t_initial: int,
        lr_min: float,
        decay_rate: float,
        warmup_t: int,
        warmup_lr_init: float,
    ) -> None:
        self.optimizer = optimizer
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.t_initial = max(int(t_initial), 1)
        self.lr_min = float(lr_min)
        self.decay_rate = float(decay_rate)
        self.warmup_t = int(warmup_t)
        self.warmup_lr_init = float(warmup_lr_init)
        self.last_epoch = 0
        if self.warmup_t > 0:
            self._set_lr(self.warmup_lr_init)

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)

    def learning_rate(self, step: int) -> float:
        return author_cosine_lr(
            step,
            base_lr=self.base_lrs[0],
            t_initial=self.t_initial,
            lr_min=self.lr_min,
            decay_rate=self.decay_rate,
            warmup_t=self.warmup_t,
            warmup_lr_init=self.warmup_lr_init,
        )

    def step(self, epoch: int) -> None:
        """``epoch`` is 1-based, matching ``for i in range(1, epochs + 1)``."""
        self.last_epoch = int(epoch)
        self._set_lr(self.learning_rate(self.last_epoch))

    def state_dict(self) -> dict[str, int | float]:
        return {
            "last_epoch": self.last_epoch,
            "t_initial": self.t_initial,
            "lr_min": self.lr_min,
            "decay_rate": self.decay_rate,
            "warmup_t": self.warmup_t,
            "warmup_lr_init": self.warmup_lr_init,
        }

    def load_state_dict(self, payload: dict[str, int | float]) -> None:
        self.last_epoch = int(payload["last_epoch"])
        if self.last_epoch > 0:
            self._set_lr(self.learning_rate(self.last_epoch))
