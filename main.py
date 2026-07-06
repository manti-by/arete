#!/usr/bin/env python3
import argparse
import logging.config

from arete import settings
from arete.enhance import cmd_enhance
from arete.info import cmd_info
from arete.train import cmd_train
from arete.validate import cmd_validate


logging.config.dictConfig(settings.LOGGING)

parser = argparse.ArgumentParser(
    prog="arete",
    description="Audio restoration / bandwidth extension model",
)
sub = parser.add_subparsers(dest="command", required=True)

train_p = sub.add_parser("train", help="Train the audio restoration model")
train_p.add_argument("--data-dir", required=True, help="Directory with clean audio files")
train_p.add_argument("--device", default="cuda", help="torch device (cuda / cpu / mps)")
train_p.add_argument("--model-type", default="waveform", choices=["waveform", "stft"], help="Model architecture")
train_p.set_defaults(func=cmd_train)

enh_p = sub.add_parser("enhance", help="Enhance an audio file using a trained checkpoint")
enh_p.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
enh_p.add_argument("--input", required=True, help="Input audio file")
enh_p.add_argument("--output", default=None, help="Output file")
enh_p.add_argument("--device", default="cpu", help="torch device")
enh_p.add_argument("--model-type", default="waveform", choices=["waveform", "stft"], help="Model architecture")
enh_p.set_defaults(func=cmd_enhance)

validate_p = sub.add_parser("validate", help="Validate and summarise audio dataset")
validate_p.add_argument("--data-dir", default="data/raw", help="Directory with audio files")
validate_p.set_defaults(func=cmd_validate)

info_p = sub.add_parser("info", help="Print model info")
info_p.set_defaults(func=cmd_info)


def main() -> None:
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
