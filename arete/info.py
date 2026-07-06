from __future__ import annotations

import argparse

from arete import settings


def cmd_info(args: argparse.Namespace) -> None:
    from arete.models import STFTUNet, WaveformUNet

    for name, model_cls, kwargs in [
        (
            "WaveformUNet",
            WaveformUNet,
            dict(
                in_channels=settings.AUDIO["channels"],
                base_channels=settings.MODEL["base_channels"],
                depth=settings.MODEL["depth"],
                kernel_size=settings.MODEL["kernel_size"],
            ),
        ),
        (
            "STFTUNet",
            STFTUNet,
            dict(
                sample_rate=settings.AUDIO["sample_rate"],
                base_ch=settings.MODEL["base_channels"],
                depth=max(1, settings.MODEL["depth"] - 1),
            ),
        ),
    ]:
        model = model_cls(**kwargs)
        n = sum(p.numel() for p in model.parameters())
        print(f"{name}: {n:,} parameters")
