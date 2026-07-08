# Arete Documentation

## Architecture

### Baseline A — Waveform Residual U-Net (recommended first model)

```
degraded waveform (B, C, T)
        |
   Encoder x depth  (Conv1d + GroupNorm + LeakyReLU + Downsample)
        |
   Bottleneck
        |
   Decoder x depth  (Upsample + skip concat + Conv1d)
        |
   Output projection
        |
   y = x + f_theta(x)   <- residual prediction
```

Parameters: ~8-20 M depending on `base_channels` and `depth`.

### Baseline B — STFT-domain 2-D U-Net

Operates on log-magnitude spectrograms. Predicted magnitude is combined with the input phase via ISTFT. Easier to interpret visually but requires phase recovery.

> **Note:** The `GriffinLim` transform is instantiated but not used; phase is taken from the input mixture.

## Loss Function

```
L = lambda1 * L_time + lambda2 * L_MR-STFT + lambda3 * L_mel + lambda4 * L_highband
```

| Component           | Default weight | Purpose                      |
| ------------------- | -------------- | ---------------------------- |
| L1 (waveform)       | 0.3            | Time-domain alignment        |
| MultiResolutionSTFT | 0.5            | Perceptual spectral fidelity |
| Mel-STFT            | 0.2            | Perceptual loudness matching |
| High-band L1        | 0.1            | Prevent high-freq smearing   |

## Project Structure

```
arete/
├── main.py                  # CLI entry point (train / enhance / validate / info)
├── arete/
│   ├── train.py             # cmd_train handler
│   ├── enhance.py           # cmd_enhance handler
│   ├── validate.py          # cmd_validate handler
│   ├── info.py              # cmd_info handler
│   ├── settings.py          # hyperparameter source of truth
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

## Usage

### Validate dataset

```bash
uv run python main.py validate --data-dir data/raw
```

> **Note:** Validation accepts `.wav`, `.flac`, `.aiff`, `.aif` only. Training (`dataset.py`) also accepts `.mp4` and `.m4a`.

### Model info

```bash
uv run python main.py info
```

Training logs go to `runs/` (TensorBoard), checkpoints to `checkpoints/`.

```bash
tensorboard --logdir runs/
```

## Configuration

All hyperparameters live in `arete/settings.py` as Python dicts. There is no YAML config file.

| Dict            | Key fields (defaults)                                                                     |
| --------------- | ----------------------------------------------------------------------------------------- |
| `AUDIO`         | `sample_rate: 44100`, `channels: 1`, `chunk_seconds: 2.5`, `hop_seconds: 1.0`             |
| `DEGRADATIONS`  | `mp3_bitrates: [64,96,128,192]`, `aac_bitrates: [64,96,128]`, `opus_bitrates: [48,64,96]`, `prob_lowpass: 0.3`, `lowpass_cutoff_hz: 8000` |
| `MODEL`         | `name: WaveformUNet`, `base_channels: 32`, `depth: 5`, `kernel_size: 15`, `use_ema: true`, `ema_decay: 0.999` |
| `LOSS`          | `lambda_l1: 0.3`, `lambda_mr_stft: 0.5`, `lambda_mel: 0.2`, `lambda_highband: 0.1`        |
| `TRAINING`      | `epochs: 100`, `batch_size: 4`, `learning_rate: 3.0e-4`, `mixed_precision: true`, `num_workers: 4` |
| `DATA`          | `train_split: 0.9`, `seed: 42`                                                           |

Logging is configured by the `LOGGING` dict (console DEBUG + file INFO to `arete.log`, overridable via `LOG_PATH` env var).

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

## Code Conventions

- **Naming**: modules `snake_case.py`; tests `tests/test_<module>.py`; classes `PascalCase`; functions `snake_case` (no underscore prefix); constants `UPPER_SNAKE_CASE` in `arete/settings.py`
- **Architecture**: strict layering — `arete/library/` (pure data, no I/O), `arete/services/<system>.py` (one per external system), `arete/models/<arch>.py` (neural modules)
- **Do not use**: `print()` (use `logging`), PEP 585 typing (`Tuple[X]`/`List[X]` — use PEP 604 `X | None`/`list[X]`), mutable default arguments, inline comments, emojis in code
- **Imports**: always at top of file; local imports inside functions only for circular dependency resolution

## Step-by-Step Training Plan

| Step | Action                                                                               |
| ---- | ------------------------------------------------------------------------------------ |
| 1    | Place lossless masters (FLAC/WAV/AIFF/MP4/M4A) in `data/raw/`                        |
| 2    | Run `uv run python main.py validate --data-dir data/raw`                             |
| 3    | Run `uv run python main.py train --model-type waveform`                              |
| 4    | Evaluate objective metrics: LSD, multi-resolution STFT loss on validation            |
| 5    | Listen on 20-30 held-out tracks across genres (electronic, acoustic, vocal)          |
| 6    | Switch to `--model-type stft` and compare                                            |
| 7    | Tune loss weights in `arete/settings.py` (LOSS dict) if high-freq is over- or under-corrected |
| 8    | Optional: add stem-aware pipeline via demucs                                         |

## Evaluation Metrics

- **LSD** (Log-Spectral Distance) — primary objective metric for bandwidth extension
- **Multi-resolution STFT loss** on validation set
- **SI-SDR** — signal-to-distortion ratio (less meaningful for music, use as secondary)
- **MUSHRA-like blind listening** — play degraded / DSP baseline / model output in random order and rate perceptual quality

## DSP Baselines

Located in `arete/baselines/dsp.py`. Compare your model against:

| Baseline            | Description                                        |
| ------------------- | -------------------------------------------------- |
| `upsample_baseline` | Downsample -> upsample cycle (codec approximation) |
| `wiener_denoise`    | Spectral subtraction noise reduction               |
| `harmonic_bwe`      | Copy lower harmonics into upper frequency band     |

## Risks & Known Issues

- **Pleasant hallucination**: model may add synthetic "air" that sounds nice but doesn't match the original. Monitor high-band loss on validation.
- **Codec overfitting**: mix MP3 / AAC / Opus / resampling in each batch (already default).
- **Genre bias**: validate separately on electronic, acoustic, vocal, and dense mix tracks.

## Stack

| Library                | Role                                           |
| ---------------------- | ---------------------------------------------- |
| `torch` + `torchaudio` | Model, STFT, resampling, training              |
| `librosa`              | Audio analysis and offline preprocessing       |
| `soundfile`            | File I/O in degradation pipeline               |
| `auraloss`             | MultiResolutionSTFTLoss, MelSTFTLoss           |
| `ffmpeg`               | On-the-fly codec degradations (MP3/AAC/Opus)   |
| `uv`                   | Fast Python packaging and virtual environments |
| `ruff`                 | Linting and formatting                         |
| `ty`                   | Type checking                                  |
