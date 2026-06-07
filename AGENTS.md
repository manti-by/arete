# arete — AGENTS.md

## Stack

PyTorch + torchaudio audio restoration project. Packaging: uv + hatchling. Lint/format: ruff. Tests: pytest. Config: YAML. Pre-commit: ruff, pyupgrade, bandit.

## Setup

- Requires Python >= 3.11, [uv](https://github.com/astral-sh/uv), and **ffmpeg in PATH**.
- `uv sync --extra dev` installs all dev deps (pytest, pytest-cov, ruff, pre-commit, bandit).
- `uv sync --extra demucs` for stem-aware enhancement.

## Commands

| Purpose | Command |
|---------|---------|
| Install | `make install` |
| Lint | `uv run ruff check src/ tests/ scripts/` |
| Format | `uv run ruff format src/ tests/ scripts/` |
| Pre-commit | `uv run pre-commit run --all-files` |
| Test | `uv run pytest` |
| Test with coverage | `uv run pytest --cov=audio_upscaler --cov-report=term-missing` |
| Train | `uv run audio-upscaler train --data-dir data/raw --model-type waveform` |
| Enhance | `uv run audio-upscaler enhance --checkpoint <ckpt> --input <file> --output <out>` |
| Model info | `uv run audio-upscaler info` |
| Validate dataset | `uv run python scripts/prepare_dataset.py --data-dir data/raw` |
| CI | `make ci` (install → lint → test) |

## Makefile targets

`install`, `lint`, `format`, `check` (pre-commit), `test`, `test-cov`, `train`, `info`, `validate`, `update`, `ci`.

## Ruff config (`pyproject.toml`)

- `line-length = 100`, `target-version = "py311"`, double quotes, space indentation.
- Rules: E, F, W, I, UP, B, SIM, N, S, BLE, A, INP, RUF (E501, S603 ignored).
- Source root: `src/`.
- Per-file: `__init__.py` ignores E402/F403/F405; `tests/*` ignores S/N802/N815.

## CI (`.github/workflows/checks.yml`)

Runs on push/PR: uv install → ruff check → pytest.

## Project structure

```
src/audio_upscaler/
  cli.py          — Click CLI (train / enhance / info)
  models/         — WaveformUNet, STFTUNet, EMA
  data/           — AudioPairDataset, Degrader (on-the-fly codec degradations)
  losses/         — CombinedAudioLoss (L1 + MultiResolutionSTFT + Mel + High-band)
  training/       — Trainer (AMP, EMA, cosine LR, TensorBoard)
  inference/      — Enhancer (overlap-add)
  baselines/      — DSP baselines for comparison
configs/default.yaml     — all hyperparameters
data/raw/                — place lossless tracks (FLAC/WAV/AIFF) here
data/degraded/           — optional pre-degraded cache
data/processed/          — optional pre-chunked cache
tests/                   — test_models.py, test_losses.py, test_dataset.py, test_baselines.py
```

## Key facts

- CLI entrypoint: `audio-upscaler` (defined in pyproject.toml `[project.scripts]`, calls `audio_upscaler.cli:main`).
- Hyperparameter source of truth: `configs/default.yaml`. All training/inference uses this YAML config.
- Training produces TensorBoard logs in `runs/` and checkpoints in `checkpoints/`.
- Default model is WaveformUNet (residual waveform U-Net). Pass `--model-type stft` for STFT-domain variant.
- Tests use dummy `torch.randn` tensors, no real audio files needed. No external services.
- Source files consistently use `from __future__ import annotations`.
