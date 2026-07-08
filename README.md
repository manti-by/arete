# arete

**DSEE-like audio restoration for music** — supervised neural model that restores high-frequency detail and removes compression artifacts from lossy audio (MP3 / AAC / Opus).

Trained on pairs of `(degraded, clean)` audio generated on-the-fly from your lossless tracks.

## Motivation

Sony DSEE restores detail lost during compression. arete implements the same idea as a research/pet project:
- Input: compressed / artifact-ridden audio
- Output: restored waveform with recovered high-frequency content
- Training: synthetic degradation of your own lossless library

## Quick Start

Requires **Python ≥ 3.13**, [uv](https://github.com/astral-sh/uv), and **ffmpeg** in PATH.

```bash
curl -Lsf https://astral.sh/uv/install.sh | sh
cd arete && make install
```

Place lossless tracks in `data/raw/`, then:

```bash
# Train a model
uv run python main.py train --data-dir data/raw --model-type waveform --device cuda

# Enhance a file
uv run python main.py enhance --checkpoint checkpoints/checkpoint_epoch_final.pt --input track.mp3 --output enhanced.wav
```

## Development

```bash
make check       # lint + format + type check
make test        # run tests
make test-cov    # tests with coverage
```

## License

AGPL-3.0

