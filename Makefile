SHELL := /bin/bash

check:
	git add .
	uv run ty check
	uv run pre-commit run

install:
	uv sync --all-extras --dev

update:
	uv run uv-bump
	uv sync --all-extras --dev
	uv run pre-commit autoupdate

test:
	uv run pytest tests/

test-cov:
	uv run pytest tests/ --cov=arete --cov-report=term-missing --cov-report=html

info:
	uv run python main.py info

train:
	uv run python main.py train --data-dir data/raw --model-type waveform

enhance:
	uv run python main.py enhance --checkpoint <ckpt> --input <file> --output <out>

validate:
	uv run python main.py validate --data-dir data/raw

ci: install check test
