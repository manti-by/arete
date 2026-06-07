"""PyTorch Dataset for audio restoration training.

Loads clean audio files, slices them into fixed-length chunks,
and applies on-the-fly degradations to generate (degraded, clean) pairs.
"""

from __future__ import annotations

import random
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

from .degradations import Degrader


SUPPORTED_EXTENSIONS = {".wav", ".flac", ".aiff", ".aif", ".mp4", ".m4a"}


def _discover_audio_files(root: str | Path) -> list[Path]:
    root = Path(root)
    files = [p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    files.sort()
    return files


class AudioPairDataset(Dataset):
    """Dataset that returns (degraded, clean) waveform pairs.

    Args:
        root:            directory with clean audio files (lossless preferred)
        sample_rate:     target sample rate; files are resampled if needed
        chunk_seconds:   length of each training chunk in seconds
        mono:            convert to mono if True
        degrader:        Degrader instance; if None, uses default settings
        files:           optional explicit list of file paths
    """

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
            self.files = _discover_audio_files(root)

        if not self.files:
            raise ValueError(f"No audio files found under {root!r}")

        # Pre-compute (file_index, start_sample) index
        self._index = self._build_index()

    def _build_index(self) -> list[tuple[int, int]]:
        index: list[tuple[int, int]] = []
        for fi, path in enumerate(self.files):
            sinfo = sf.info(str(path))
            n_samples = int(sinfo.frames * self.sample_rate / sinfo.samplerate)
            starts = range(0, max(1, n_samples - self.chunk_len), self.chunk_len)
            index.extend((fi, s) for s in starts)
        return index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        fi, start = self._index[idx]
        path = self.files[fi]

        # Load audio
        waveform, sr = torchaudio.load(str(path))

        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        # Mono
        if self.mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Slice chunk
        end = start + self.chunk_len
        if waveform.shape[-1] < end:
            # Pad if file is shorter than chunk
            pad = end - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        clean = waveform[:, start:end]

        # Loudness normalise to -23 LUFS (approximate)
        rms = clean.pow(2).mean().sqrt().clamp(min=1e-8)
        clean = clean / rms * 0.1

        # Degrade
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
    """Split audio files into train/val datasets."""
    files = _discover_audio_files(root)
    rng = random.Random(seed)
    rng.shuffle(files)
    n_train = max(1, int(len(files) * train_split))
    train_files, val_files = files[:n_train], files[n_train:]

    deg = degrader or Degrader(sample_rate=sample_rate)
    train_ds = AudioPairDataset(root, sample_rate, chunk_seconds, mono, deg, train_files)
    val_ds = AudioPairDataset(root, sample_rate, chunk_seconds, mono, deg, val_files)
    return train_ds, val_ds
