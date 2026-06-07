"""Exponential Moving Average of model weights."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMA:
    """Maintains an EMA copy of model parameters.

    Usage::

        ema = EMA(model, decay=0.999)
        # after each optimizer.step():
        ema.update(model)
        # for validation / inference:
        with ema.average_parameters():
            output = model(x)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for s_param, param in zip(self.shadow.parameters(), model.parameters(), strict=True):
            s_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def average_parameters(self) -> EMAContext:
        return EMAContext(self)


class EMAContext:
    def __init__(self, ema: EMA) -> None:
        self.ema = ema
        self._backup: dict[str, torch.Tensor] = {}

    def __enter__(self) -> None:
        model = self.ema.shadow
        self._backup = {k: v.clone() for k, v in model.state_dict().items()}

    def __exit__(self, *args: object) -> None:
        pass  # shadow is already the averaged copy; nothing to restore
