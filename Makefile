install:
	uv sync --extra dev

install-all:
	uv sync --extra dev --extra demucs

lint:
	uv run ruff check src/ tests/ scripts/

format:
	uv run ruff format src/ tests/ scripts/

check:
	uv run pre-commit run --all-files

test:
	uv run pytest

test-cov:
	uv run pytest --cov=audio_upscaler --cov-report=term-missing

train:
	uv run audio-upscaler train --data-dir data/raw

info:
	uv run audio-upscaler info

validate:
	uv run python scripts/prepare_dataset.py --data-dir data/raw

update:
	uv sync --upgrade
	uv run pre-commit autoupdate

ci: install lint test
