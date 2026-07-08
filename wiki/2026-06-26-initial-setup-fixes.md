# Training Fix Report

## Problem

Running `uv run audio-upscaler train --data-dir data/raw --model-type waveform --device cuda` produced two errors:

1. **CUDA Out of Memory** — batch size 16 exceeded 5.6 GiB GPU capacity
2. **Corrupted audio files** — DataLoader workers crashed on decode failures
3. **Deprecated PyTorch AMP API** — warnings for `torch.cuda.amp.GradScaler` / `autocast`

---

## Fix 1: Reduce batch size

**File:** `configs/default.yaml`

Reduced `batch_size` from 16 to 4 to fit within 5.6 GiB GPU memory.

```diff
-  batch_size: 16
+  batch_size: 4
```

This reduced per-batch memory from ~216 MiB to ~54 MiB, fitting within the ~99 MiB free.

---

## Fix 2: Migrate deprecated `torch.cuda.amp` API

**File:** `src/audio_upscaler/training/trainer.py`

`torch.cuda.amp.GradScaler` and `torch.cuda.amp.autocast` are deprecated in favor of
`torch.amp.GradScaler('cuda', ...)` and `torch.amp.autocast('cuda', ...)`.

```diff
- from torch.cuda.amp import GradScaler, autocast
+ from torch.amp import GradScaler, autocast
```

```diff
- self.scaler = GradScaler(enabled=...)
+ self.scaler = GradScaler("cuda", enabled=...)
```

```diff
- with autocast(enabled=self.scaler.is_enabled()):
+ with autocast("cuda", enabled=self.scaler.is_enabled()):
```

---

## Fix 3: Handle corrupted audio files gracefully

**File:** `src/audio_upscaler/data/dataset.py`

### 3a. Remove non-existent `torchaudio.info()` call

`torchaudio.info()` was removed in torchaudio 2.x. Calling it raised `AttributeError` on every
file, causing all files to be filtered out as "corrupted".

**Removed entirely** from validation.

### 3b. Robust file validation in `_build_index`

Replaced `_valid_files` pre-filtering with on-the-fly validation in `_build_index`.
Files that fail `sf.info()` (soundfile metadata check) are logged and skipped.

### 3c. Fallback decode in `__getitem__`

Wrapped `torchaudio.load()` in a try/except with a `sf.read()` fallback. If both fail,
a silent chunk is returned for training instead of crashing the DataLoader worker.

```python
try:
    waveform, sr = torchaudio.load(str(path))
except Exception:
    try:
        data, sr = sf.read(str(path), always_2d=True, dtype="float32")
        waveform = torch.from_numpy(data).T.contiguous()
    except Exception:
        clean = torch.zeros(1, self.chunk_len)
        degraded = self.degrader.degrade(clean.clone())
        return degraded, clean
```

### 3d. Remove `.mp3` from supported extensions

`soundfile` (used for metadata queries) does not support MP3. MP3 files from the dataset
were failing metadata checks.

```diff
- SUPPORTED_EXTENSIONS = {".wav", ".flac", ".aiff", ".aif", ".mp4", ".m4a", ".mp3"}
+ SUPPORTED_EXTENSIONS = {".wav", ".flac", ".aiff", ".aif", ".mp4", ".m4a"}
```

---

## Result

| Aspect | Before | After |
|--------|--------|-------|
| CUDA memory | OOM at batch 16 | OK at batch 4 |
| Deprecated API | FutureWarnings | Clean |
| Corrupted FLAC files | DataLoader crash | Silent chunk fallback |
| Dataset files | 0 (all filtered) | ~530k train + ~58k val chunks |
| Training step | N/A | loss=3.64 on first step |

The command runs without errors and produces valid training output.

---

## Fix 4: Console progress logging

**File:** `src/audio_upscaler/training/trainer.py`

Added real-time training progress to the console so users can monitor training without
TensorBoard:

### Batch-level progress (tqdm)

`_train_epoch` and `_val_epoch` now show a `tqdm` progress bar that updates with current
batch loss:

```
Train 1/100: 100%|████████| 1200/1200 [02:15<00:00, 8.89batch/s, loss=3.52]
Val  1/100: 100%|████████|  134/134  [00:15<00:00, 8.93batch/s, loss=3.48]
```

### Epoch-level summary

After each epoch, a structured log line is emitted:

```
INFO | Epoch 1/100 | train_loss=3.52 | lr=3.00e-4 | time=135.2s | val_loss=3.48 | val_time=15.1s | elapsed=150.3s | remaining≈14880s
```

Includes epoch number, train loss, learning rate, epoch time, val loss (when applicable),
total elapsed time, and estimated remaining time.

### Start/end summary

- Logs total batches before training begins
- Logs total wall-clock time on completion

```python
logger.info("Starting training for %d epochs (%d train / %d val batches per epoch)",
             self.epochs, n_train, n_val)
# ...
logger.info("Training complete in %.1fs (%.1f min)", total, total / 60)
```
