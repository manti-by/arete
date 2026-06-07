#!/usr/bin/env python3
"""Utility script: validate audio files in data/raw/ and print dataset stats.

Usage:
    uv run python scripts/prepare_dataset.py --data-dir data/raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torchaudio


SUPPORTED = {".wav", ".flac", ".aiff", ".aif"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarise audio dataset")
    parser.add_argument("--data-dir", default="data/raw", help="Directory with audio files")
    args = parser.parse_args()

    root = Path(args.data_dir)
    files = [p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED]
    files.sort()

    if not files:
        print(f"No supported audio files found in {root}")
        return

    total_duration = 0.0
    errors = []

    for path in files:
        try:
            info = torchaudio.info(str(path))
            duration = info.num_frames / info.sample_rate
            total_duration += duration
            print(
                f"  {path.name:50s}  {info.sample_rate} Hz  {info.num_channels}ch  {duration:.1f}s"
            )
        except (OSError, RuntimeError, ValueError) as e:
            errors.append((path, str(e)))
            print(f"  ERROR: {path.name}: {e}")

    print(f"\nTotal files  : {len(files)}")
    print(f"Total duration: {total_duration / 60:.1f} min ({total_duration / 3600:.2f} h)")
    print(f"Errors       : {len(errors)}")

    if errors:
        print("\nFiles with errors:")
        for p, msg in errors:
            print(f"  {p}: {msg}")


if __name__ == "__main__":
    main()
