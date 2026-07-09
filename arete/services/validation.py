import logging
from pathlib import Path

import soundfile as sf
import torchaudio


logger = logging.getLogger(__name__)


SUPPORTED = {".wav", ".flac", ".aiff", ".aif", ".mp4", ".m4a"}


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
        sf_ok = True
        try:
            info = sf.info(str(path))
            duration = info.frames / info.samplerate
            total_duration += duration
        except (OSError, ValueError):
            sf_ok = False
            duration = 0.0

        decode_ok = True
        decode_err = ""
        try:
            torchaudio.load(str(path), num_frames=1)
        except RuntimeError as e:
            decode_ok = False
            decode_err = str(e)
            if not sf_ok:
                errors.append((path, decode_err))
                logger.error("%s: %s", path.name, decode_err)

        if sf_ok and decode_ok:
            logger.info(
                "%s  %s Hz  %sch  %.1fs",
                path.name.ljust(50),
                info.samplerate,
                info.channels,
                duration,
            )
        elif sf_ok and not decode_ok:
            errors.append((path, decode_err))
            logger.error("%s: corrupt / unreadable by torchcodec: %s", path.name, decode_err)

    logger.info("Total files  : %d", len(files))
    logger.info("Total duration: %.1f min (%.2f h)", total_duration / 60, total_duration / 3600)
    logger.info("Errors       : %d", len(errors))

    if errors:
        logger.warning("Files with errors:")
        for path, msg in errors:
            logger.warning("  %s: %s", path, msg)
