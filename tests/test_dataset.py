from pathlib import Path

import numpy as np
import soundfile as sf

from arete.data import AudioPairDataset


SR = 22050


def make_temp_audio(directory: Path, sr: int = SR, duration: float = 3.0, n: int = 3) -> list[Path]:
    files = []
    for i in range(n):
        path = directory / f"track_{i:02d}.wav"
        samples = np.random.uniform(-0.5, 0.5, (int(sr * duration),)).astype(np.float32)
        sf.write(str(path), samples, sr)
        files.append(path)
    return files


class TestAudioPairDataset:
    def test_basic_loading(self, tmp_path: Path) -> None:
        make_temp_audio(tmp_path)
        ds = AudioPairDataset(root=tmp_path, sample_rate=SR, chunk_seconds=1.0, mono=True)
        assert len(ds) > 0

    def test_output_shapes(self, tmp_path: Path) -> None:
        make_temp_audio(tmp_path, n=2)
        chunk_sec = 1.0
        ds = AudioPairDataset(root=tmp_path, sample_rate=SR, chunk_seconds=chunk_sec, mono=True)
        degraded, clean = ds[0]
        expected_len = int(chunk_sec * SR)
        assert degraded.shape == (1, expected_len), f"Got {degraded.shape}"
        assert clean.shape == (1, expected_len), f"Got {clean.shape}"

    def test_len_matches_index(self, tmp_path: Path) -> None:
        make_temp_audio(tmp_path, n=2, duration=5.0)
        ds = AudioPairDataset(root=tmp_path, sample_rate=SR, chunk_seconds=2.0, mono=True)
        assert len(ds) == len(ds._index)

    def test_stereo_no_mono(self, tmp_path: Path) -> None:
        path = tmp_path / "stereo.wav"
        samples = np.random.uniform(-0.5, 0.5, (2, int(SR * 2))).astype(np.float32).T
        sf.write(str(path), samples, SR)
        ds = AudioPairDataset(root=tmp_path, sample_rate=SR, chunk_seconds=1.0, mono=False)
        degraded, clean = ds[0]
        assert degraded.shape[0] == 2
        assert clean.shape[0] == 2

    def test_explicit_file_list(self, tmp_path: Path) -> None:
        files = make_temp_audio(tmp_path, n=1)
        ds = AudioPairDataset(root=tmp_path, sample_rate=SR, chunk_seconds=1.0, mono=True, files=files)
        assert len(ds) > 0

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        import pytest

        with pytest.raises(ValueError, match="No audio files"):
            AudioPairDataset(root=tmp_path, sample_rate=SR, chunk_seconds=1.0, mono=True)

    def test_make_train_val_datasets(self, tmp_path: Path) -> None:
        from arete.data.dataset import make_train_val_datasets

        make_temp_audio(tmp_path, n=5)
        train_ds, val_ds = make_train_val_datasets(
            root=tmp_path,
            sample_rate=SR,
            chunk_seconds=1.0,
            mono=True,
            train_split=0.8,
            seed=42,
        )
        assert len(train_ds) > 0
        assert len(val_ds) > 0
        total_files = len(train_ds.files) + len(val_ds.files)
        assert total_files == 5

    def test_file_shorter_than_chunk(self, tmp_path: Path) -> None:
        path = tmp_path / "short.wav"
        samples = np.random.uniform(-0.5, 0.5, (int(SR * 0.5),)).astype(np.float32)
        sf.write(str(path), samples, SR)
        ds = AudioPairDataset(root=tmp_path, sample_rate=SR, chunk_seconds=2.0, mono=True)
        degraded, clean = ds[0]
        assert degraded.shape[-1] == int(SR * 2)
        assert clean.shape[-1] == int(SR * 2)

    def test_unsupported_extension_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "track.txt").write_text("not audio")
        make_temp_audio(tmp_path, n=1)
        ds = AudioPairDataset(root=tmp_path, sample_rate=SR, chunk_seconds=1.0, mono=True)
        assert len(ds) > 0
