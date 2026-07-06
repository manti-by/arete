# arete - AGENTS.md

## Project Overview

Arete is an audio restoration / bandwidth extension model (DSEE-like) for music, built with PyTorch and torchaudio. It trains neural networks to restore high frequencies and remove codec artifacts from degraded audio.

## Project Structure

- `main.py`: CLI entry point (argparse: train / enhance / info)
- `arete/settings.py`: Core configuration constants
- `arete/library/`: Pure data layer (dataclasses, TypedDicts, exceptions). No I/O.
- `arete/models/`: Neural network architectures (WaveformUNet, STFTUNet, EMA)
- `arete/services/`: External system integrations (Degrader, Enhancer, Trainer)
- `arete/data/`: Dataset and data loading utilities
- `arete/baselines/`: Classical DSP baselines for comparison
- `arete/settings.py`: Hyperparameter source of truth
- `tests/`: Comprehensive test suite
- `data/raw/`: Place lossless audio tracks here
- `data/degraded/`: Optional pre-degraded cache
- `data/processed/`: Optional pre-chunked cache

## Development Commands

### Package Management
```bash
make install
make update
```

### Running
```bash
uv run python main.py train --data-dir data/raw
uv run python main.py enhance --checkpoint <ckpt> --input <file> --output <out>
uv run python main.py info
```

### Lint / Format / Type Check
```bash
make check              # ty check + pre-commit (ruff lint + format)
uv run ruff check .     # lint only
uv run ruff format .    # format only
uv run ty check         # type check only
```

### Test
```bash
make test
make test-cov
```

## Language & Environment

- Python >=3.13 (see `pyproject.toml`)
- Ruff enforces style and linting (120 char line length)
- Use type hints for all functions
- Use f-strings only (never `.format()` or `%` formatting)
- Use list/dict/set comprehensions over `map`/`filter`
- Prefer `pathlib.Path` over `os.path`
- Follow PEP 257 for docstrings where used
- Use named arguments instead of positional in function/method calls

## Code Conventions

**Naming**:
- Modules: `snake_case.py`; tests mirror at `tests/test_<module>.py`
- Classes: `PascalCase`; dataclasses in `library/models.py`, TypedDicts in `library/types.py`
- Functions: `snake_case`; never prefix with `_` (no private/underscore-prefixed functions)
- Constants: `UPPER_SNAKE_CASE`; live in `arete/settings.py`

**Architecture** (strict layering, no skipping):
- `arete/library/` - pure: dataclasses, TypedDicts, exceptions. No I/O.
- `arete/services/<system>.py` - one external system per file
- `arete/models/<arch>.py` - neural network modules

**Do NOT use**: `print()` (use `logging`), PEP 585 typing (`Tuple[X]/List[X]/Dict[X]` - use PEP 604 `X | None / list[X]`), mutable default arguments (use `field(default_factory=...)`), inline comments, emojis in code.

**Imports**: Always at the top of the file. Local imports inside functions only for circular dependency resolution.

## Testing Guidelines

- Use `pytest` for tests
- Tests live in `tests/` directory
- Run with `make test` or `uv run pytest tests/`
- Tests use dummy `torch.randn` tensors, no real audio files needed
- `conftest.py` for shared fixtures

## Dependencies

**Core**: torch, torchaudio, torchcodec, librosa, soundfile, auraloss, numpy, scipy, pyyaml, tqdm, tensorboard
**Dev**: pytest, pytest-cov, ruff, pre-commit, bandit, ty
**Optional**: demucs (for stem-aware enhancement)

## Config

Hyperparameter source of truth: `arete/settings.py`. All training and inference uses these constants. Environment overrides via `ARETE_CONFIG` env var.
