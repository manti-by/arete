from __future__ import annotations

import logging
from pathlib import Path

import soundfile as sf


logger = logging.getLogger(__name__)


SUPPORTED = {".wav", ".flac", ".aiff", ".aif"}


def validate_dataset(data_dir: str | Path = "data/raw") -> None:
    root = Path(data_dir)
    files = [p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED]
    files.sort()

    if not files:
        logger.warning("No supported audio files found in %s", root)
        return

    total_duration = 0.0
    errors: list[tuple[Path, str]] = []

    for path in files:
        try:
            info = sf.info(str(path))
            duration = info.frames / info.samplerate
            total_duration += duration
            logger.info(
                "%s  %s Hz  %sch  %.1fs",
                path.name.ljust(50),
                info.samplerate,
                info.channels,
                duration,
            )
        except (OSError, ValueError) as e:
            errors.append((path, str(e)))
            logger.error("%s: %s", path.name, e)

    logger.info("Total files  : %d", len(files))
    logger.info("Total duration: %.1f min (%.2f h)", total_duration / 60, total_duration / 3600)
    logger.info("Errors       : %d", len(errors))

    if errors:
        logger.warning("Files with errors:")
        for path, msg in errors:
            logger.warning("  %s: %s", path, msg)
