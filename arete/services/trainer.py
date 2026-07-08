from __future__ import annotations

import logging
import time as time_mod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from arete.losses import CombinedAudioLoss
from arete.models import EMA


logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict[str, Any],
        device: str = "cuda",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        tc = config["training"]
        lc = config["loss"]
        ac = config["audio"]

        self.epochs = tc["epochs"]
        self.grad_clip = tc.get("grad_clip", 1.0)
        self.val_every = tc.get("val_every_n_epochs", 5)
        self.save_every = tc.get("save_every_n_epochs", 10)

        self.optimizer = AdamW(model.parameters(), lr=tc["learning_rate"])
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs - tc.get("warmup_epochs", 5),
        )

        amp_device = "cuda" if "cuda" in device else device
        self.scaler = GradScaler(amp_device, enabled=(tc.get("mixed_precision", True) and device != "cpu"))

        self.criterion = CombinedAudioLoss(
            sample_rate=ac["sample_rate"],
            lambda_l1=lc["lambda_l1"],
            lambda_mr_stft=lc["lambda_mr_stft"],
            lambda_mel=lc["lambda_mel"],
            lambda_highband=lc["lambda_highband"],
            highband_cutoff_hz=lc["highband_cutoff_hz"],
            fft_sizes=lc["mr_stft_fft_sizes"],
            hop_sizes=lc["mr_stft_hop_sizes"],
            win_sizes=lc["mr_stft_win_sizes"],
            n_mels=lc["mel_n_mels"],
        ).to(device)

        model_cfg = config["model"]
        if model_cfg.get("use_ema", True):
            self.ema = EMA(model, decay=model_cfg.get("ema_decay", 0.999))
        else:
            self.ema = None

        log_dir = Path(tc.get("log_dir", "runs"))
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.ckpt_dir = Path(tc.get("checkpoint_dir", "checkpoints"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0

    def fit(self) -> None:
        n_train = len(self.train_loader)
        n_val = len(self.val_loader)
        logger.info(
            "Starting training for %d epochs (%d train / %d val batches per epoch)",
            self.epochs,
            n_train,
            n_val,
        )
        start_time = time_mod.time()

        for epoch in range(1, self.epochs + 1):
            t0 = time_mod.time()
            train_losses = self.train_epoch(epoch)
            self.scheduler.step()
            t_train = time_mod.time() - t0

            self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)
            for k, v in train_losses.items():
                self.writer.add_scalar(f"train/{k}", v, epoch)

            val_losses: dict[str, float] | None = None
            t_val = 0.0
            if epoch % self.val_every == 0:
                t0 = time_mod.time()
                val_losses = self.val_epoch(epoch)
                t_val = time_mod.time() - t0
                for k, v in val_losses.items():
                    self.writer.add_scalar(f"val/{k}", v, epoch)

            if epoch % self.save_every == 0:
                self.save_checkpoint(epoch)

            elapsed = time_mod.time() - start_time
            remaining = elapsed / epoch * (self.epochs - epoch)
            log = (
                f"Epoch {epoch}/{self.epochs} | train_loss={train_losses['total']:.4f}"
                f" | lr={self.optimizer.param_groups[0]['lr']:.2e}"
                f" | time={t_train:.1f}s"
            )
            if val_losses is not None:
                log += f" | val_loss={val_losses['total']:.4f} | val_time={t_val:.1f}s"
            log += f" | elapsed={elapsed:.1f}s | remaining≈{remaining:.0f}s"
            logger.info(log)

        total = time_mod.time() - start_time
        logger.info("Training complete in %.1fs (%.1f min)", total, total / 60)
        self.save_checkpoint("final")
        self.writer.close()

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        accum: dict[str, float] = {}
        n = 0

        pbar = tqdm(self.train_loader, desc=f"Train {epoch}/{self.epochs}", leave=False)
        for degraded, clean in pbar:
            degraded = degraded.to(self.device)
            clean = clean.to(self.device)

            self.optimizer.zero_grad()

            amp_device = "cuda" if "cuda" in self.device else self.device
            with autocast(device_type=amp_device, enabled=self.scaler.is_enabled()):
                pred = self.model(degraded)
                loss, components = self.criterion(pred, clean)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.ema:
                self.ema.update(self.model)

            for k, v in components.items():
                accum[k] = accum.get(k, 0.0) + v
            n += 1
            self.global_step += 1
            pbar.set_postfix(loss=f"{components.get('total', loss.item()):.4f}")

        return {k: v / n for k, v in accum.items()}

    @torch.no_grad()
    def val_epoch(self, epoch: int) -> dict[str, float]:
        eval_model = self.ema.shadow if self.ema else self.model
        eval_model.eval()
        accum: dict[str, float] = {}
        n = 0

        pbar = tqdm(self.val_loader, desc=f"Val  {epoch}/{self.epochs}", leave=False)
        for degraded, clean in pbar:
            degraded = degraded.to(self.device)
            clean = clean.to(self.device)
            pred = eval_model(degraded)
            _, components = self.criterion(pred, clean)
            for k, v in components.items():
                accum[k] = accum.get(k, 0.0) + v
            n += 1
            pbar.set_postfix(loss=f"{components.get('total', 0.0):.4f}")

        return {k: v / n for k, v in accum.items()}

    def save_checkpoint(self, epoch: int | str) -> None:
        path = self.ckpt_dir / f"checkpoint_epoch_{epoch}.pt"
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        if self.ema:
            state["ema"] = self.ema.shadow.state_dict()
        torch.save(state, path)
        logger.info("Saved checkpoint: %s", path)
