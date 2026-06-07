# audio_upscaler

**DSEE-like audio restoration for music** — supervised neural model that restores high-frequency detail and removes compression artifacts from lossy audio (MP3 / AAC / Opus).

Trained on pairs of `(degraded, clean)` audio generated on-the-fly from your lossless tracks.

---

## Motivation

Sony DSEE (Digital Sound Enhancement Engine) restores detail lost during compression. This project implements the same idea as a research/pet project:

- Input: compressed / artifact-ridden audio
- Output: restored waveform with recovered high-frequency content
- Training: synthetic degradation of your own lossless library

---

## Architecture

### Baseline A — Waveform Residual U-Net (recommended first model)

```
degraded waveform (B, C, T)
        │
   Encoder ×depth  (Conv1d + GroupNorm + LeakyReLU + Downsample)
        │
   Bottleneck
        │
   Decoder ×depth  (Upsample + skip concat + Conv1d)
        │
   Output projection
        │
   y = x + f_θ(x)   ← residual prediction
```

Parameters: ~8–20 M depending on `base_channels` and `depth`.

### Baseline B — STFT-domain 2-D U-Net

Operates on log-magnitude spectrograms. Predicted magnitude is combined with the input phase via ISTFT. Easier to interpret visually but requires phase recovery.

---

## Loss Function

```
L = λ1·L_time + λ2·L_MR-STFT + λ3·L_mel + λ4·L_highband
```

| Component | Default weight | Purpose |
|-----------|---------------|---------|
| L1 (waveform) | 0.3 | Time-domain alignment |
| MultiResolutionSTFT | 0.5 | Perceptual spectral fidelity |
| Mel-STFT | 0.2 | Perceptual loudness matching |
| High-band L1 | 0.1 | Prevent high-freq smearing |

---

## Project Structure

```
audio_upscaler/
├── configs/
│   └── default.yaml          # all hyperparameters
├── data/
│   ├── raw/                  # your lossless music (FLAC/WAV/AIFF)
│   ├── degraded/             # optional pre-degraded cache
│   └── processed/            # optional pre-chunked cache
├── src/
│   └── audio_upscaler/
│       ├── models/
│       │   ├── unet_waveform.py   # Waveform residual U-Net
│       │   ├── unet_stft.py       # STFT 2-D U-Net
│       │   └── ema.py             # Exponential Moving Average
│       ├── data/
│       │   ├── dataset.py         # AudioPairDataset + train/val split
│       │   └── degradations.py    # ffmpeg-based on-the-fly degradations
│       ├── losses/
│       │   └── combined.py        # CombinedAudioLoss (auraloss-based)
│       ├── training/
│       │   └── trainer.py         # Training loop + TensorBoard + EMA + AMP
│       ├── inference/
│       │   └── enhancer.py        # Overlap-add inference engine
│       ├── baselines/
│       │   └── dsp.py             # Classical DSP baselines for comparison
│       └── cli.py                 # Click CLI: train / enhance / info
├── tests/
│   ├── test_models.py
│   ├── test_losses.py
│   ├── test_dataset.py
│   └── test_baselines.py
├── pyproject.toml
└── README.md
```

---

## Installation

Requires **Python ≥ 3.11**, [uv](https://github.com/astral-sh/uv), and **ffmpeg** in PATH.

```bash
# Install uv (if not already installed)
curl -Lsf https://astral.sh/uv/install.sh | sh

# Clone / unzip the project
cd audio_upscaler

# Create virtual environment and install dependencies
uv sync

# Install with dev extras (ruff, pytest)
uv sync --extra dev

# Optional: demucs support for stem-aware enhancement
uv sync --extra demucs
```

---

## Usage

### Train

```bash
# Put your lossless tracks in data/raw/
uv run audio-upscaler train \
    --data-dir data/raw \
    --model-type waveform \
    --device cuda
```

Training logs → `runs/` (TensorBoard), checkpoints → `checkpoints/`.

```bash
# Monitor training
tensorboard --logdir runs/
```

### Enhance a file

```bash
uv run audio-upscaler enhance \
    --checkpoint checkpoints/checkpoint_epoch_final.pt \
    --input my_track_128kbps.mp3 \
    --output my_track_enhanced.wav \
    --device cpu
```

### Model info

```bash
uv run audio-upscaler info
```

---

## Configuration

All hyperparameters live in `configs/default.yaml`. Key sections:

```yaml
audio:
  sample_rate: 44100
  channels: 1          # 1 = mono, 2 = stereo
  chunk_seconds: 2.5

model:
  name: WaveformUNet
  base_channels: 32    # increase for more capacity
  depth: 5             # encoder/decoder levels

training:
  epochs: 100
  batch_size: 16
  learning_rate: 3.0e-4
  mixed_precision: true
```

---

## Development

```bash
# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Tests
uv run pytest

# Tests with coverage
uv run pytest --cov=audio_upscaler --cov-report=term-missing
```

---

## Step-by-Step Training Plan

| Step | Action |
|------|--------|
| 1 | Place lossless masters (FLAC/WAV) in `data/raw/` |
| 2 | Run `audio-upscaler train --model-type waveform` |
| 3 | Evaluate objective metrics: LSD, multi-resolution STFT loss on validation |
| 4 | Listen on 20–30 held-out tracks across genres (electronic, acoustic, vocal) |
| 5 | Switch to `--model-type stft` and compare |
| 6 | Tune loss weights in `configs/default.yaml` if high-freq is over- or under-corrected |
| 7 | Optional: add stem-aware pipeline via demucs |

---

## Evaluation Metrics

- **LSD** (Log-Spectral Distance) — primary objective metric for bandwidth extension
- **Multi-resolution STFT loss** on validation set
- **SI-SDR** — signal-to-distortion ratio (less meaningful for music, use as secondary)
- **MUSHRA-like blind listening** — play degraded / DSP baseline / model output in random order and rate perceptual quality

---

## DSP Baselines

Located in `src/audio_upscaler/baselines/dsp.py`. Compare your model against:

| Baseline | Description |
|----------|-------------|
| `upsample_baseline` | Downsample → upsample cycle (codec approximation) |
| `wiener_denoise` | Spectral subtraction noise reduction |
| `harmonic_bwe` | Copy lower harmonics into upper frequency band |

---

## Risks & Known Issues

- **Pleasant hallucination**: model may add synthetic "air" that sounds nice but doesn't match the original. Monitor high-band loss on validation.
- **Codec overfitting**: mix MP3 / AAC / Opus / resampling in each batch (already default).
- **Genre bias**: validate separately on electronic, acoustic, vocal, and dense mix tracks.

---

## Stack

| Library | Role |
|---------|------|
| `torch` + `torchaudio` | Model, STFT, resampling, training |
| `librosa` | Audio analysis and offline preprocessing |
| `soundfile` | File I/O in degradation pipeline |
| `auraloss` | MultiResolutionSTFTLoss, MelSTFTLoss |
| `ffmpeg` | On-the-fly codec degradations (MP3/AAC/Opus) |
| `uv` | Fast Python packaging and virtual environments |
| `ruff` | Linting and formatting |

---

## License

MIT
