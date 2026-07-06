from __future__ import annotations

from typing import TypedDict


class TrainMetrics(TypedDict):
    l_time: float
    l_mr_stft: float
    l_mel: float
    l_highband: float
    total: float


class CheckpointState(TypedDict, total=False):
    epoch: int | str
    model: dict
    optimizer: dict
    scheduler: dict
    ema: dict | None
