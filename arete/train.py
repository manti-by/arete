from __future__ import annotations

import argparse

from arete import settings


def cmd_train(args: argparse.Namespace) -> None:
    from torch.utils.data import DataLoader

    from arete.data.dataset import make_train_val_datasets
    from arete.models import STFTUNet, WaveformUNet
    from arete.services import Degrader, Trainer

    degrader = Degrader(
        sample_rate=settings.AUDIO["sample_rate"],
        mp3_bitrates=settings.DEGRADATIONS["mp3_bitrates"],
        aac_bitrates=settings.DEGRADATIONS["aac_bitrates"],
        opus_bitrates=settings.DEGRADATIONS["opus_bitrates"],
        resample_rates=settings.DEGRADATIONS["resample_rates"],
        prob_lowpass=settings.DEGRADATIONS["prob_lowpass"],
        lowpass_cutoff_hz=settings.DEGRADATIONS["lowpass_cutoff_hz"],
        prob_stereo_collapse=settings.DEGRADATIONS["prob_stereo_collapse"],
        prob_clipping=settings.DEGRADATIONS["prob_clipping"],
        clipping_threshold=settings.DEGRADATIONS["clipping_threshold"],
    )

    train_ds, val_ds = make_train_val_datasets(
        root=args.data_dir,
        sample_rate=settings.AUDIO["sample_rate"],
        chunk_seconds=settings.AUDIO["chunk_seconds"],
        mono=(settings.AUDIO["channels"] == 1),
        train_split=settings.DATA["train_split"],
        seed=settings.DATA["seed"],
        degrader=degrader,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=settings.TRAINING["batch_size"],
        shuffle=True,
        num_workers=settings.TRAINING["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=settings.TRAINING["batch_size"],
        shuffle=False,
        num_workers=settings.TRAINING["num_workers"],
        pin_memory=True,
    )

    if args.model_type == "waveform":
        model = WaveformUNet(
            in_channels=settings.AUDIO["channels"],
            base_channels=settings.MODEL["base_channels"],
            depth=settings.MODEL["depth"],
            kernel_size=settings.MODEL["kernel_size"],
        )
    else:
        model = STFTUNet(
            sample_rate=settings.AUDIO["sample_rate"],
            base_ch=settings.MODEL["base_channels"],
            depth=settings.MODEL["depth"],
        )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model_type} | Parameters: {n_params:,}")

    cfg = {
        "audio": dict(settings.AUDIO),
        "training": dict(settings.TRAINING),
        "loss": dict(settings.LOSS),
        "model": dict(settings.MODEL),
    }

    trainer = Trainer(model, train_loader, val_loader, cfg, device=args.device)
    trainer.fit()
    print("Training complete.")
