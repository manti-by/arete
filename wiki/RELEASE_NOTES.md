# Release Notes

## v0.2.1 (2026-07-08)

### New
- Real-time training progress via `tqdm` progress bars with live loss updates
- Epoch-level summary logging with train/val loss, LR, epoch time, elapsed, and ETA
- Total training time logged on completion

### Fixed
- `batch_size` reduced from 16 to 4 to avoid CUDA OOM on GPUs with ~5.6 GiB VRAM
- Missing `window=` argument in `torch.stft()` calls in `highband_loss` and `FallbackMRSTFTLoss`
- Corrupted audio files now produce silent chunk fallback instead of crashing DataLoader
- `torchaudio.info()` removed in torchaudio 2.x — replaced with `soundfile.info()` for metadata
- MP3 removed from supported extensions (unsupported by `soundfile`)
- Deprecated `torch.cuda.amp.GradScaler`/`autocast` migrated to `torch.amp` API

### Changed
- Training now logs start/end summaries with batch counts and total elapsed time
- Consolidated Makefile target ordering

---

## v0.2.0 (2026-07-07) — Project Restructure

### Major restructuring
- Migrated entire codebase from `src/audio_upscaler/` to top-level package `arete/`
- Adopted demetra-style strict layering: `library/`, `services/`, `models/`
- CLI migrated from Click to argparse with subcommands: `train`, `enhance`, `info`, `validate`
- Python minimum bumped 3.11 → 3.13
- License changed MIT → AGPL-3.0

### Configuration
- All hyperparameters moved from `configs/default.yaml` into `arete/settings.py`
- Flat constants grouped into typed dicts: `AUDIO`, `DEGRADATIONS`, `MODEL`, `LOSS`, `TRAINING`, `DATA`
- Environment override via `ARETE_CONFIG` env var
- `configs/default.yaml` and `scripts/prepare_dataset.py` deleted

### Logging
- Added `logging.config.dictConfig` setup (copied from demetra)
- Console (DEBUG) and file (INFO) handlers with timestamp formatting
- Logs to `arete.log`

### Validation module
- New `arete/validate.py` — `cmd_validate` CLI handler
- New `arete/services/validation.py` — `validate_dataset()` function

### Testing
- Test coverage raised from 44% → 79%
- 7 new test files, 60+ tests total
- Click tests replaced with direct argparse tests
- All existing tests updated for the new module layout

### Tooling
- Ruff config: line-length 100 → 120, target-version py313
- Type checking with `ty` (new dev dependency)
- Pre-commit hooks: ruff + pyupgrade + bandit
- CI workflow simplified inline steps
- Makefile rewritten with targets: `install`, `update`, `check`, `test`, `test-cov`
- `opencode.json` added with `ty` LSP server config

### Other
- `AGENTS.md` created with full development commands and conventions
- Dataset validation now skips corrupt files with silent chunk fallback
- Fallback decode in `__getitem__` via `soundfile.read()`
- Removed ~2300 lines of old code

---

## v0.1.0 (2026-06-25) — Initial Release

### Initial project scaffold
- Basic project structure under `src/audio_upscaler/`
- WaveformUNet and STFTUNet architectures
- EMA (Exponential Moving Average) module
- On-the-fly degradation pipeline via ffmpeg (MP3, AAC, Opus, resampling, lowpass, clipping)
- CombinedAudioLoss: L1 + MultiResolutionSTFT + Mel + high-band loss
- Click-based CLI with `train`, `enhance`, `info` commands
- YAML-based configuration
- Overlap-add inference engine (`Enhancer`)
- Classical DSP baselines (`upsample_baseline`, `wiener_denoise`, `harmonic_bwe`)
- Initial test suite covering models, losses, dataset, baselines
- TensorBoard logging, checkpoint saving/loading
- Initial test coverage ~44%
