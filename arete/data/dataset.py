import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

from arete.services.degradation import Degrader


logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".wav", ".flac", ".aiff", ".aif", ".mp4", ".m4a"}


def discover_audio_files(root: str | Path) -> list[Path]:
    root = Path(root)
    files = [p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    files.sort()
    return files


class AudioPairDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        sample_rate: int = 44100,
        chunk_seconds: float = 2.5,
        mono: bool = True,
        degrader: Degrader | None = None,
        files: list[Path] | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_len = int(chunk_seconds * sample_rate)
        self.mono = mono
        self.degrader = degrader or Degrader(sample_rate=sample_rate)

        if files is not None:
            self.files = list(files)
        else:
            self.files = discover_audio_files(root)

        if not self.files:
            raise ValueError(f"No audio files found under {root!r}")

        self._index = self.build_index()

        if not self._index:
            raise ValueError(f"Could not build any valid chunks from files under {root!r}")

    def build_index(self) -> list[tuple[int, int]]:
        index: list[tuple[int, int]] = []
        for fi, path in enumerate(self.files):
            try:
                sinfo = sf.info(str(path))
                n_samples = int(sinfo.frames * self.sample_rate / sinfo.samplerate)
            except (OSError, ValueError):
                try:
                    waveform, _ = torchaudio.load(str(path))
                    n_samples = waveform.shape[-1]
                except RuntimeError:
                    logger.warning("Skipping unreadable file: %s", path)
                    continue
            starts = range(0, max(1, n_samples - self.chunk_len), self.chunk_len)
            index.extend((fi, s) for s in starts)
        return index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        fi, start = self._index[index]
        path = self.files[fi]

        try:
            waveform, sr = torchaudio.load(str(path))
        except RuntimeError:
            logger.warning("Failed to decode %s, returning silence", path)
            waveform = torch.zeros(1, self.chunk_len)
            sr = self.sample_rate

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        if self.mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        end = start + self.chunk_len
        if waveform.shape[-1] < end:
            pad = end - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        clean = waveform[:, start:end]

        rms = clean.pow(2).mean().sqrt().clamp(min=1e-8)
        clean = clean / rms * 0.1

        degraded = self.degrader.degrade(clean.clone())

        return degraded, clean


def make_train_val_datasets(
    root: str | Path,
    sample_rate: int = 44100,
    chunk_seconds: float = 2.5,
    mono: bool = True,
    train_split: float = 0.9,
    seed: int = 42,
    degrader: Degrader | None = None,
) -> tuple[AudioPairDataset, AudioPairDataset]:
    files = discover_audio_files(root)
    np.random.default_rng(seed).shuffle(files)
    n_train = max(1, int(len(files) * train_split))
    train_files, val_files = files[:n_train], files[n_train:]

    deg = degrader or Degrader(sample_rate=sample_rate)
    train_ds = AudioPairDataset(root, sample_rate, chunk_seconds, mono, deg, train_files)
    val_ds = AudioPairDataset(root, sample_rate, chunk_seconds, mono, deg, val_files)
    return train_ds, val_ds
