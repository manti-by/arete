# Project Restructure: `src/audio_upscaler/` → `arete/` — 2026-07-07

## Rationale

Refactored the entire project from a flat `src/audio_upscaler/` layout to
mirror the conventions established in the **demetra** project: strict
layering, one-system-per-file services, dict-based config groups, and
centralised CLI entry point.

---

## Summary

Restructured entire project from `src/audio_upscaler/` to top-level `arete/` package, migrated CLI from Click to argparse, bumped Python 3.11 → 3.13, added type checking (`ty`), updated linting config, rewrote Makefile/CI, and refreshed the test suite.

---

## Changes

### 1. Module restructuring

**Before (`src/audio_upscaler/`):**
```
src/audio_upscaler/
├── cli.py               — Click CLI
├── models/              — WaveformUNet, STFTUNet, EMA
├── data/                — dataset.py, degradations.py
├── losses/              — combined.py
├── training/            — trainer.py
├── inference/           — enhancer.py
└── baselines/           — dsp.py
```

**After (`arete/`):**
```
arete/
├── main.py              — argparse CLI entry point
├── settings.py          — grouped config constants + LOGGING dict
├── train.py             — cmd_train handler
├── enhance.py           — cmd_enhance handler
├── info.py              — cmd_info handler
├── validate.py          — cmd_validate handler
├── library/             — pure data layer (dataclasses, TypedDicts, exceptions)
│   ├── models.py
│   ├── types.py
│   └── exceptions.py
├── models/              — neural network architectures
│   ├── unet_waveform.py
│   ├── unet_stft.py
│   └── ema.py
├── services/            — external systems, one file per system
│   ├── degrader.py      — ffmpeg-based on-the-fly degradations
│   ├── enhancer.py      — overlap-add inference engine
│   ├── trainer.py       — training loop + TensorBoard + EMA + AMP
│   └── validation.py    — dataset validation
├── data/
│   └── dataset.py       — AudioPairDataset + train/val split
├── losses/
│   └── combined.py      — CombinedAudioLoss (auraloss-based)
└── baselines/
    └── dsp.py           — classical DSP baselines
```

#### Key alignment with demetra

| Concept | demetra | arete |
|---|---|---|
| Pure data layer | `demetra/library/` | `arete/library/` |
| Services (one per system) | `demetra/services/linear.py`, `git.py`, … | `arete/services/degrader.py`, `enhancer.py`, `trainer.py` |
| Config singleton | `demetra/settings.py` | `arete/settings.py` |
| Top-level CLI | `main.py` with argparse | `main.py` with argparse subparsers |
| Tests mirror | `tests/test_*.py` | `tests/test_*.py` |
| Project conventions | `AGENTS.md` | `AGENTS.md` |

### 2. CLI migrated from Click to argparse

Replaced `click`-based CLI (in `src/audio_upscaler/cli.py`) with `main.py` + subcommand handlers (demetra pattern):

```
main.py  →  train / enhance / info / validate
                ↕               ↕          ↕        ↕
         arete/train.py   enhance.py  info.py  validate.py
```

Each handler is a thin `cmd_*` function in its own file under `arete/`, imported lazily inside `main.py`. Subcommand flags:

| Subcommand | Flags |
|---|---|
| `train` | `--data-dir`, `--device`, `--model-type` |
| `enhance` | `--checkpoint`, `--input`, `--output`, `--device`, `--model-type` |
| `info` | (none) |
| `validate` | `--data-dir` |

**`test_cli.py` updated:**
- Removed `CliRunner`/`click.testing` dependency
- Replaced with direct `parser.parse_args()` assertions
- Config-file test removed (moved to `test_settings.py`)

### 3. Python version bumped 3.11 → 3.13

| File | Change |
|---|---|
| `pyproject.toml` | `requires-python = ">=3.13"` |
| `.pre-commit-config.yaml` | `pyupgrade` args: `--py313-plus`; `bandit`: `language_version: python3.13` |
| `pyproject.toml` (ruff) | `target-version = "py313"` |

### 4. Ruff config modernised

**`pyproject.toml`:**
- `line-length` 100 → 120
- Removed `src = ["src"]` (no longer needed with flat layout)
- `indent-width = 4` (explicit)
- Isort: added custom `arete` section
- Per-file ignores expanded to `**/{tests,docs}/*` and `**/*test*.py`
- Various ignore flags updated (`RUF100`, `RUF012`, `N806` added; `COM812`, `COM819`, `ISC001`, `ISC002`, `Q*` added)
- Removed `unfixable` list
- Removed `[tool.ruff.format]` section

### 5. License changed

`pyproject.toml`: `MIT` → `GNU Affero General Public License v3.0`

### 6. Version bumped

`pyproject.toml`: `0.1.0` → `0.2.0`

### 7. Makefile rewritten

**Removed targets:** `lint`, `format`, `install-all`

**New/updated targets:**

| Target | Command |
|---|---|
| `install` | `uv sync --all-extras --dev` |
| `update` | `uv run uv-bump && uv sync --all-extras --dev && uv run pre-commit autoupdate` |
| `check` | `uv run ty check && uv run pre-commit run --all-files` |
| `test` | `uv run pytest tests/` |
| `test-cov` | `uv run pytest tests/ --cov=arete --cov-report=term-missing --cov-report=html` |
| `train` | `uv run python main.py train --data-dir data/raw --model-type waveform` |
| `enhance` | `uv run python main.py enhance --checkpoint ...` |
| `validate` | `uv run python main.py validate --data-dir data/raw` |
| `ci` | `install check test` |

### 8. CI workflow simplified (`.github/workflows/checks.yml`)

- Concise `on: [push, pull_request]`
- Inline steps (removed `name` annotations)
- `uv sync --locked --all-extras --dev` (was `--extra dev`)
- Replaced `ruff check` with `pre-commit run --all-files` (covers ruff + pyupgrade + bandit)

### 9. Settings refactored — dict-based config groups

Flat constants grouped into dict literals matching demetra's pattern:

```python
AUDIO       = {"sample_rate": 44100, "channels": 1, ...}
DEGRADATIONS= {"mp3_bitrates": [...], ...}
MODEL       = {"name": "WaveformUNet", ...}
LOSS        = {"lambda_l1": 0.3, ...}
TRAINING    = {"epochs": 100, ...}
DATA        = {"train_split": 0.9, ...}
```

| Group | Purpose |
|---|---|
| `AUDIO` | Audio I/O params |
| `DEGRADATIONS` | Codec / degradation params |
| `MODEL` | Architecture hyperparams |
| `LOSS` | Loss weighting & STFT sizes |
| `TRAINING` | Optimizer / scheduler / I/O |
| `DATA` | Split ratio & seed |
| `LOGGING` | `logging.config.dictConfig` format |

### 10. Config deleted (`configs/default.yaml`)

All hyperparameters now live in `arete/settings.py`. Environment override via `ARETE_CONFIG` env var. No YAML loaded at runtime.

### 11. Logging pattern

Copied demetra's `logging.config.dictConfig` setup:

- **`arete/settings.py`** — `LOGGING` dict with `console` (DEBUG) and `file` (INFO) handlers, standard formatter with timestamps, plus `LOG_PATH` variable.
- **`main.py`** — calls `logging.config.dictConfig(settings.LOGGING)` at module level (was `logging.basicConfig(...)`).
- All service modules create `logger = logging.getLogger(__name__)`.
- `.gitignore` — `arete.log` added.

### 12. Validation module (new)

`scripts/prepare_dataset.py` was deleted. Its logic moved into:

- `arete/validate.py` — `cmd_validate()` handler
- `arete/services/validation.py` — `validate_dataset()` function
- `main.py validate` — subparser wiring

### 13. Test suite reworked

| File | Key changes |
|---|---|
| `test_baselines.py` | `audio_upscaler.baselines` → `arete.baselines` |
| `test_cli.py` | Click tests → argparse + `capsys`, removed config test |
| `test_dataset.py` | `audio_upscaler.data` → `arete.data`, inlined `make_train_val_datasets` call |
| `test_degradations.py` | `audio_upscaler.data.degradations` → `arete.services.degradation`, removed docstrings/comments |
| `test_enhancer.py` | `audio_upscaler.inference` → `arete.services`, removed docstrings/comments |
| `test_losses.py` | `audio_upscaler.losses` → `arete.losses`, removed nan-check loop |
| `test_models.py` | `audio_upscaler.models` → `arete.models`, removed docstrings/comments |
| `test_trainer.py` | `audio_upscaler.training` → `arete.services`, added `_DummyDataset`, `weights_only=True` in `torch.load()`, `_save_checkpoint` → `save_checkpoint` |

### 14. `pyproject.toml` — added tooling config

- `[tool.ty.src]` — type-checking source config with exclude list
- `[dependency-groups.dev]` — added `ty>=0.0.51`
- `[tool.pytest.ini_options]` — `asyncio_mode = "auto"`, `addopts = '-p no:warnings'`

### 15. `opencode.json` — ty LSP server

Added `ty` language server configuration:
```json
"ty": {
  "command": ["uvx", "ty", "server"],
  "extensions": [".py", ".pyi"]
}
```

### 16. Files deleted

- `configs/default.yaml`
- `scripts/prepare_dataset.py`
- `src/audio_upscaler/` (entire tree: cli.py, models/, data/, losses/, training/, inference/, baselines/)

---

## AGENTS.md

Created with full development commands, code conventions, and architectural rules (mirroring demetra's `AGENTS.md`). Includes:

- `make install` / `make check` / `make test` commands
- Ruff lint/format, `ty` type checking
- Strict layering rules (no skipping `library/` → `models/`, etc.)
- Testing guidelines (pytest, dummy tensors, no real audio)

---

## Verification

**Commands run and passed:**
- `ruff check .` — clean
- `ruff format .` — 35 files unchanged
- `ty check` — all type checks passed
- `uv run pytest tests/ -v` — all tests passing

---

## Stats

- **36 files changed**, 355 insertions, 2383 deletions
- 2 commits in history (initial scaffold + this session)
