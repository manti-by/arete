from __future__ import annotations

import argparse
from pathlib import Path

from arete import settings


def cmd_enhance(args: argparse.Namespace) -> None:
    from arete.models import STFTUNet, WaveformUNet
    from arete.services import Enhancer

    if args.model_type == "waveform":
        model = WaveformUNet(
            in_channels=settings.AUDIO["channels"],
            base_channels=settings.MODEL["base_channels"],
            depth=settings.MODEL["depth"],
            kernel_size=settings.MODEL["kernel_size"],
        )
    else:
        model = STFTUNet(sample_rate=settings.AUDIO["sample_rate"], base_ch=settings.MODEL["base_channels"])

    if args.output is None:
        p = Path(args.input)
        output = str(p.parent / f"{p.stem}_enhanced{p.suffix}")
    else:
        output = args.output

    enhancer = Enhancer.from_checkpoint(
        checkpoint_path=args.checkpoint,
        model=model,
        sample_rate=settings.AUDIO["sample_rate"],
        chunk_seconds=settings.AUDIO["chunk_seconds"],
        hop_seconds=settings.AUDIO["hop_seconds"],
        device=args.device,
    )
    enhancer.enhance_file(args.input, output)
