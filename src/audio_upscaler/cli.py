"""Command-line interface for audio-upscaler."""

from __future__ import annotations

import logging
from pathlib import Path

import click
import yaml


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


@click.group()
def main() -> None:
    """Audio Upscaler — DSEE-like audio restoration for music."""


@main.command()
@click.option(
    "--config", default="configs/default.yaml", show_default=True, help="Path to YAML config"
)
@click.option("--data-dir", required=True, help="Directory with clean lossless audio files")
@click.option("--device", default="cuda", show_default=True, help="torch device (cuda / cpu / mps)")
@click.option(
    "--model-type", default="waveform", type=click.Choice(["waveform", "stft"]), show_default=True
)
def train(config: str, data_dir: str, device: str, model_type: str) -> None:
    """Train the audio restoration model."""
    from torch.utils.data import DataLoader

    from .data import Degrader
    from .data.dataset import make_train_val_datasets
    from .models import STFTUNet, WaveformUNet
    from .training import Trainer

    with open(config) as f:
        cfg = yaml.safe_load(f)

    ac = cfg["audio"]
    tc = cfg["training"]
    mc = cfg["model"]
    dc = cfg["degradations"]

    degrader = Degrader(
        sample_rate=ac["sample_rate"],
        mp3_bitrates=dc["mp3_bitrates"],
        aac_bitrates=dc["aac_bitrates"],
        opus_bitrates=dc["opus_bitrates"],
        resample_rates=dc["resample_rates"],
        prob_lowpass=dc["prob_lowpass"],
        lowpass_cutoff_hz=dc["lowpass_cutoff_hz"],
        prob_stereo_collapse=dc["prob_stereo_collapse"],
        prob_clipping=dc["prob_clipping"],
        clipping_threshold=dc["clipping_threshold"],
    )

    train_ds, val_ds = make_train_val_datasets(
        root=data_dir,
        sample_rate=ac["sample_rate"],
        chunk_seconds=ac["chunk_seconds"],
        mono=(ac["channels"] == 1),
        train_split=cfg["data"]["train_split"],
        seed=cfg["data"]["seed"],
        degrader=degrader,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=tc["batch_size"],
        shuffle=True,
        num_workers=tc["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tc["batch_size"],
        shuffle=False,
        num_workers=tc["num_workers"],
        pin_memory=True,
    )

    if model_type == "waveform":
        model = WaveformUNet(
            in_channels=ac["channels"],
            base_channels=mc["base_channels"],
            depth=mc["depth"],
            kernel_size=mc["kernel_size"],
        )
    else:
        model = STFTUNet(
            sample_rate=ac["sample_rate"],
            base_ch=mc["base_channels"],
            depth=mc["depth"],
        )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    click.echo(f"Model: {model_type} | Parameters: {n_params:,}")

    trainer = Trainer(model, train_loader, val_loader, cfg, device=device)
    trainer.fit()
    click.echo("Training complete.")


@main.command()
@click.option("--checkpoint", required=True, help="Path to checkpoint .pt file")
@click.option("--input", "input_path", required=True, help="Input audio file")
@click.option(
    "--output", "output_path", default=None, help="Output file (default: <input>_enhanced.<ext>)"
)
@click.option("--config", default="configs/default.yaml", show_default=True)
@click.option("--device", default="cpu", show_default=True, help="torch device")
@click.option(
    "--model-type", default="waveform", type=click.Choice(["waveform", "stft"]), show_default=True
)
def enhance(
    checkpoint: str,
    input_path: str,
    output_path: str | None,
    config: str,
    device: str,
    model_type: str,
) -> None:
    """Enhance a single audio file using a trained checkpoint."""
    import yaml

    from .inference import Enhancer
    from .models import STFTUNet, WaveformUNet

    with open(config) as f:
        cfg = yaml.safe_load(f)

    ac = cfg["audio"]
    mc = cfg["model"]

    if model_type == "waveform":
        model = WaveformUNet(
            in_channels=ac["channels"],
            base_channels=mc["base_channels"],
            depth=mc["depth"],
            kernel_size=mc["kernel_size"],
        )
    else:
        model = STFTUNet(sample_rate=ac["sample_rate"], base_ch=mc["base_channels"])

    if output_path is None:
        p = Path(input_path)
        output_path = str(p.parent / f"{p.stem}_enhanced{p.suffix}")

    enhancer = Enhancer.from_checkpoint(
        checkpoint_path=checkpoint,
        model=model,
        sample_rate=ac["sample_rate"],
        chunk_seconds=ac["chunk_seconds"],
        hop_seconds=ac["hop_seconds"],
        device=device,
    )
    enhancer.enhance_file(input_path, output_path)


@main.command()
@click.option("--config", default="configs/default.yaml", show_default=True)
def info(config: str) -> None:
    """Print model parameter count and config summary."""
    import yaml

    from .models import STFTUNet, WaveformUNet

    with open(config) as f:
        cfg = yaml.safe_load(f)

    ac = cfg["audio"]
    mc = cfg["model"]

    for name, model_cls, kwargs in [
        (
            "WaveformUNet",
            WaveformUNet,
            dict(
                in_channels=ac["channels"],
                base_channels=mc["base_channels"],
                depth=mc["depth"],
                kernel_size=mc["kernel_size"],
            ),
        ),
        (
            "STFTUNet",
            STFTUNet,
            dict(
                sample_rate=ac["sample_rate"],
                base_ch=mc["base_channels"],
                depth=max(1, mc["depth"] - 1),
            ),
        ),
    ]:
        model = model_cls(**kwargs)
        n = sum(p.numel() for p in model.parameters())
        click.echo(f"{name}: {n:,} parameters")
