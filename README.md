# arete

**DSEE-like audio restoration for music** — supervised neural model that restores high-frequency detail and removes compression artifacts from lossy audio (MP3 / AAC / Opus).

Trained on pairs of `(degraded, clean)` audio generated on-the-fly from your lossless tracks.

---

## Motivation

Sony DSEE (Digital Sound Enhancement Engine) restores detail lost during compression. arete implements the same idea as a research/pet project:

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
arete/
├── main.py                  # CLI entry point (argparse: train / enhance / info / validate)
├── configs/
│   └── default.yaml         # all hyperparameters
├── arete/
│   ├── settings.py          # config loading from YAML + environment
│   ├── library/             # pure data layer (dataclasses, TypedDicts, exceptions)
│   ├── models/              # neural architectures
│   │   ├── unet_waveform.py # Waveform residual U-Net
│   │   ├── unet_stft.py     # STFT 2-D U-Net
│   │   └── ema.py           # Exponential Moving Average
│   ├── services/            # external system integrations
│   │   ├── degradation.py   # ffmpeg-based on-the-fly degradations
│   │   ├── enhancer.py      # Overlap-add inference engine
│   │   ├── trainer.py       # Training loop + TensorBoard + EMA + AMP
│   │   └── validation.py    # Dataset validation and summarisation
│   ├── data/
│   │   └── dataset.py       # AudioPairDataset + train/val split
│   ├── losses/
│   │   └── combined.py      # CombinedAudioLoss (auraloss-based)
│   └── baselines/
│       └── dsp.py           # Classical DSP baselines for comparison
├── data/
│   ├── raw/                 # your lossless music (FLAC/WAV/AIFF)
│   ├── degraded/            # optional pre-degraded cache
│   └── processed/           # optional pre-chunked cache
├── tests/
│   ├── test_models.py
│   ├── test_losses.py
│   ├── test_dataset.py
│   ├── test_baselines.py
│   ├── test_enhancer.py
│   ├── test_trainer.py
│   ├── test_degradations.py
│   └── test_cli.py
├── pyproject.toml
├── Makefile
└── README.md
```

---

## Installation

Requires **Python ≥ 3.13**, [uv](https://github.com/astral-sh/uv), and **ffmpeg** in PATH.

```bash
# Install uv (if not already installed)
curl -Lsf https://astral.sh/uv/install.sh | sh

cd arete

# Create virtual environment and install dependencies
make install
```

---

## Usage

### Train

```bash
# Put your lossless tracks in data/raw/
uv run python main.py train \
    --data-dir data/raw \
    --model-type waveform \
    --device cuda
```

Training logs → `runs/` (TensorBoard), checkpoints → `checkpoints/`.

```bash
tensorboard --logdir runs/
```

### Enhance a file

```bash
uv run python main.py enhance \
    --checkpoint checkpoints/checkpoint_epoch_final.pt \
    --input my_track_128kbps.mp3 \
    --output my_track_enhanced.wav \
    --device cpu
```

### Validate dataset

```bash
uv run python main.py validate --data-dir data/raw
```

### Model info

```bash
uv run python main.py info
```

---

## Configuration

All hyperparameters live in `configs/default.yaml`. Key sections:

```yaml
audio:
  sample_rate: 44100
  channels: 1
  chunk_seconds: 2.5

model:
  name: WaveformUNet
  base_channels: 32
  depth: 5

training:
  epochs: 100
  batch_size: 16
  learning_rate: 3.0e-4
  mixed_precision: true
```

Override config path via `ARETE_CONFIG` environment variable.

---

## Development

```bash
# Install dev dependencies
make install

# Lint, format, and type check
make check

# Run tests
make test

# Tests with coverage
make test-cov
```

Individual tools:

```bash
uv run ruff check .          # lint
uv run ruff format .         # format
uv run ty check              # type check
uv run pytest tests/         # test
```

---

## Step-by-Step Training Plan

| Step | Action |
|------|--------|
| 1 | Place lossless masters (FLAC/WAV) in `data/raw/` |
| 2 | Run `uv run python main.py validate --data-dir data/raw` |
| 3 | Run `uv run python main.py train --model-type waveform` |
| 4 | Evaluate objective metrics: LSD, multi-resolution STFT loss on validation |
| 5 | Listen on 20–30 held-out tracks across genres (electronic, acoustic, vocal) |
| 6 | Switch to `--model-type stft` and compare |
| 7 | Tune loss weights in `configs/default.yaml` if high-freq is over- or under-corrected |
| 8 | Optional: add stem-aware pipeline via demucs |

---

## Evaluation Metrics

- **LSD** (Log-Spectral Distance) — primary objective metric for bandwidth extension
- **Multi-resolution STFT loss** on validation set
- **SI-SDR** — signal-to-distortion ratio (less meaningful for music, use as secondary)
- **MUSHRA-like blind listening** — play degraded / DSP baseline / model output in random order and rate perceptual quality

---

## DSP Baselines

Located in `arete/baselines/dsp.py`. Compare your model against:

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
| `ty` | Type checking |

---

## License

MIT
