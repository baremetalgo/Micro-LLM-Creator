from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from .config import ModelConfig, TrainingConfig, dataclass_to_jsonable
from .model import MicroGPT


class TokenDataset(Dataset):
    """Sliding-window token dataset for next-token prediction."""

    def __init__(self, tokens: list[int], context_length: int) -> None:
        """Create a token dataset.

        Args:
            tokens: Complete token stream.
            context_length: Number of input tokens per sample.

        Raises:
            ValueError: If there are not enough tokens.
        """

        if len(tokens) <= context_length:
            raise ValueError("Not enough tokens for the selected context length")
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.context_length = context_length

    def __len__(self) -> int:
        """Return the number of sliding windows available.

        Returns:
            Dataset length.
        """

        return len(self.tokens) - self.context_length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one input/target token window.

        Args:
            index: Starting token index.

        Returns:
            Pair of input tokens and next-token targets.
        """

        chunk = self.tokens[index : index + self.context_length + 1]
        return chunk[:-1], chunk[1:]


@dataclass
class TrainingResult:
    """Result returned after training.

    Attributes:
        checkpoint_path: Final model checkpoint path.
        summary_path: Training summary JSON path.
        final_train_loss: Final epoch training loss.
        final_val_loss: Final validation loss when available.
        stopped: Whether training was stopped by the user.
    """

    checkpoint_path: Path
    summary_path: Path
    final_train_loss: float
    final_val_loss: Optional[float]
    stopped: bool = False


def emit_progress(
    progress: Optional[Callable[[Any], None]],
    message: str,
    percent: Optional[int] = None,
    **metrics: Any,
) -> None:
    """Emit training progress if a callback is available.

    Args:
        progress: Optional callback for progress dictionaries.
        message: Human-readable status message.
        percent: Optional progress percentage.
        **metrics: Optional structured metrics for UI dashboards.
    """

    if progress:
        progress({"message": message, "percent": percent, **metrics})


def set_seed(seed: int) -> None:
    """Set random seeds for repeatable training.

    Args:
        seed: Integer seed value.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_tokens(tokens: list[int], validation_split: float) -> tuple[list[int], list[int]]:
    """Split tokens into train and validation streams.

    Args:
        tokens: Full token stream.
        validation_split: Fraction reserved for validation.

    Returns:
        Pair of training tokens and validation tokens.
    """

    split_at = int(len(tokens) * (1.0 - validation_split))
    split_at = max(1, min(split_at, len(tokens) - 1))
    return tokens[:split_at], tokens[split_at:]


def make_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a warmup and linear-decay scheduler.

    Args:
        optimizer: Optimizer to schedule.
        total_steps: Total optimizer steps.
        warmup_steps: Number of warmup steps.

    Returns:
        Lambda learning-rate scheduler.
    """

    warmup_steps = min(warmup_steps, max(total_steps - 1, 1))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.1, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(model: MicroGPT, loader: DataLoader, device: str, pad_token_id: int) -> float:
    """Evaluate validation loss.

    Args:
        model: Model to evaluate.
        loader: Validation data loader.
        device: Device used for evaluation.
        pad_token_id: Token ID ignored in loss.

    Returns:
        Mean validation loss.
    """

    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=pad_token_id,
            )
            losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(len(losses), 1)


def latest_checkpoint(checkpoints_dir: Path) -> Optional[Path]:
    """Find the newest checkpoint in a folder.

    Args:
        checkpoints_dir: Directory containing checkpoint files.

    Returns:
        Newest checkpoint path, or ``None``.
    """

    checkpoints = sorted(
        checkpoints_dir.glob("checkpoint_*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return checkpoints[0] if checkpoints else None


def train_model(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    train_tokens: list[int],
    val_tokens: list[int],
    pad_token_id: int,
    progress: Optional[Callable[[Any], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> TrainingResult:
    """Train a MicroGPT model.

    Args:
        model_config: Architecture settings.
        training_config: Optimizer, device, and checkpoint settings.
        train_tokens: Training token stream.
        val_tokens: Validation token stream.
        pad_token_id: Token ID ignored by cross-entropy loss.
        progress: Optional callback receiving progress dictionaries.
        should_stop: Optional callback returning true when training should stop.

    Returns:
        Training result with checkpoint and summary paths.
    """

    set_seed(training_config.seed)
    training_config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = training_config.output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    emit_progress(progress, "Building model...", 2)
    model = MicroGPT(model_config).to(training_config.device)
    emit_progress(progress, "Preparing token batches...", 4)
    train_loader = DataLoader(
        TokenDataset(train_tokens, model_config.context_length),
        batch_size=training_config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = None
    if len(val_tokens) > model_config.context_length:
        val_loader = DataLoader(
            TokenDataset(val_tokens, model_config.context_length),
            batch_size=training_config.batch_size,
            shuffle=False,
            drop_last=False,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=training_config.weight_decay,
    )
    steps_per_epoch = max(len(train_loader) // training_config.gradient_accumulation, 1)
    total_steps = max(steps_per_epoch * training_config.epochs, 1)
    scheduler = make_scheduler(optimizer, total_steps, training_config.warmup_steps)
    use_amp = training_config.use_amp and training_config.device == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    global_step = 0
    start_epoch = 0
    final_train_loss = 0.0
    final_val_loss: Optional[float] = None

    resume_path = training_config.resume_from_checkpoint if training_config.resume else None
    if resume_path is None and training_config.resume:
        resume_path = latest_checkpoint(checkpoints_dir)
    if resume_path and Path(resume_path).exists():
        emit_progress(progress, f"Resuming from checkpoint: {resume_path}", 6)
        checkpoint = torch.load(resume_path, map_location=training_config.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint and use_amp:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        global_step = int(checkpoint.get("global_step", 0))
        start_epoch = min(int(checkpoint.get("epoch", 0)), training_config.epochs)
        final_train_loss = float(checkpoint.get("train_loss", 0.0))
        final_val_loss = checkpoint.get("val_loss")
        emit_progress(progress, f"Checkpoint loaded at step {global_step}.", 8)
    else:
        emit_progress(progress, "Starting new training run.", 6)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    last_metric_time = perf_counter()
    step_time_window: list[float] = []
    for epoch in range(start_epoch, training_config.epochs):
        epoch_losses: list[float] = []
        for batch_index, (x, y) in enumerate(train_loader):
            if should_stop and should_stop():
                final_train_loss = sum(epoch_losses) / max(len(epoch_losses), 1) if epoch_losses else final_train_loss
                stopped_path = checkpoints_dir / f"checkpoint_stopped_step_{global_step}.pt"
                save_checkpoint(
                    stopped_path,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    model_config,
                    training_config,
                    global_step,
                    epoch,
                    final_train_loss,
                    final_val_loss,
                )
                emit_progress(progress, f"Training stopped. Resume checkpoint saved: {stopped_path}", 100)
                summary_path = training_config.output_dir / "training_summary.json"
                summary = {
                    "model_config": dataclass_to_jsonable(model_config),
                    "training_config": dataclass_to_jsonable(training_config),
                    "final_train_loss": final_train_loss,
                    "final_val_loss": final_val_loss,
                    "total_steps": global_step,
                    "stopped": True,
                    "resume_checkpoint": str(stopped_path),
                    "parameters": sum(p.numel() for p in model.parameters()),
                }
                summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                return TrainingResult(stopped_path, summary_path, final_train_loss, final_val_loss, stopped=True)
            x = x.to(training_config.device)
            y = y.to(training_config.device)
            with autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                    ignore_index=pad_token_id,
                )
                loss = loss / training_config.gradient_accumulation

            scaler.scale(loss).backward()
            if (batch_index + 1) % training_config.gradient_accumulation == 0:
                scaler.unscale_(optimizer)
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
                grad_norm = float(grad_norm_tensor.item() if hasattr(grad_norm_tensor, "item") else grad_norm_tensor)
                weight_norm = math.sqrt(
                    sum(float(parameter.detach().float().norm(2).item()) ** 2 for parameter in model.parameters())
                )
                learning_rate = float(scheduler.get_last_lr()[0])
                update_ratio = learning_rate * grad_norm / max(weight_norm, 1e-12)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                now = perf_counter()
                step_seconds = max(now - last_metric_time, 1e-9)
                last_metric_time = now
                step_time_window.append(step_seconds)
                step_time_window = step_time_window[-50:]
                average_step_seconds = sum(step_time_window) / max(len(step_time_window), 1)
                remaining_steps = max(total_steps - global_step, 0)
                eta_seconds = remaining_steps * average_step_seconds
                samples_seen = training_config.batch_size * training_config.gradient_accumulation
                tokens_seen = samples_seen * model_config.context_length
                vram_allocated_gb = None
                vram_reserved_gb = None
                if training_config.device.startswith("cuda") and torch.cuda.is_available():
                    device_index = torch.cuda.current_device()
                    vram_allocated_gb = torch.cuda.memory_allocated(device_index) / (1024 ** 3)
                    vram_reserved_gb = torch.cuda.memory_reserved(device_index) / (1024 ** 3)
                current_progress = 8 + int(86 * min(global_step, total_steps) / max(total_steps, 1))
                emit_progress(
                    progress,
                    f"Epoch {epoch + 1}/{training_config.epochs}, step {global_step}/{total_steps}, loss {float(loss.item() * training_config.gradient_accumulation):.4f}",
                    current_progress,
                    epoch=epoch + 1,
                    total_epochs=training_config.epochs,
                    step=global_step,
                    total_steps=total_steps,
                    train_loss=float(loss.item() * training_config.gradient_accumulation),
                    val_loss=final_val_loss,
                    learning_rate=learning_rate,
                    grad_norm=grad_norm,
                    weight_norm=weight_norm,
                    update_ratio=update_ratio,
                    tokens_per_second=tokens_seen / step_seconds,
                    samples_per_second=samples_seen / step_seconds,
                    step_seconds=step_seconds,
                    average_step_seconds=average_step_seconds,
                    eta_seconds=eta_seconds,
                    remaining_steps=remaining_steps,
                    vram_allocated_gb=vram_allocated_gb,
                    vram_reserved_gb=vram_reserved_gb,
                )

                if (
                    val_loader is not None
                    and training_config.eval_interval > 0
                    and global_step % training_config.eval_interval == 0
                ):
                    final_val_loss = evaluate(model, val_loader, training_config.device, pad_token_id)
                    emit_progress(
                        progress,
                        f"Validation loss at step {global_step}: {final_val_loss:.4f}",
                        current_progress,
                        epoch=epoch + 1,
                        total_epochs=training_config.epochs,
                        step=global_step,
                        total_steps=total_steps,
                        train_loss=epoch_losses[-1] if epoch_losses else None,
                        val_loss=final_val_loss,
                    )

                if training_config.save_interval > 0 and global_step % training_config.save_interval == 0:
                    save_checkpoint(
                        checkpoints_dir / f"checkpoint_{global_step}.pt",
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        model_config,
                        training_config,
                        global_step,
                        epoch + 1,
                        final_train_loss,
                        final_val_loss,
                    )
                    emit_progress(progress, f"Saved checkpoint at step {global_step}.", current_progress)

            epoch_losses.append(float(loss.item() * training_config.gradient_accumulation))

        final_train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        if val_loader is not None:
            final_val_loss = evaluate(model, val_loader, training_config.device, pad_token_id)
        print(f"epoch {epoch + 1}/{training_config.epochs}: train_loss={final_train_loss:.4f}")
        save_checkpoint(
            checkpoints_dir / f"checkpoint_epoch_{epoch + 1}.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            model_config,
            training_config,
            global_step,
            epoch + 1,
            final_train_loss,
            final_val_loss,
        )
        emit_progress(
            progress,
            f"Epoch {epoch + 1} complete. Checkpoint saved.",
            8 + int(86 * (epoch + 1) / max(training_config.epochs, 1)),
            epoch=epoch + 1,
            total_epochs=training_config.epochs,
            step=global_step,
            total_steps=total_steps,
            train_loss=final_train_loss,
            val_loss=final_val_loss,
        )

    checkpoint_path = training_config.output_dir / "final_model.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        scaler,
        model_config,
        training_config,
        global_step,
        training_config.epochs,
        final_train_loss,
        final_val_loss,
    )
    summary_path = training_config.output_dir / "training_summary.json"
    summary = {
        "model_config": dataclass_to_jsonable(model_config),
        "training_config": dataclass_to_jsonable(training_config),
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "total_steps": global_step,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    emit_progress(
        progress,
        "Training complete.",
        100,
        epoch=training_config.epochs,
        total_epochs=training_config.epochs,
        step=global_step,
        total_steps=total_steps,
        train_loss=final_train_loss,
        val_loss=final_val_loss,
    )
    return TrainingResult(checkpoint_path, summary_path, final_train_loss, final_val_loss)


def save_checkpoint(
    path: Path,
    model: MicroGPT,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    global_step: int,
    epoch: int,
    train_loss: float,
    val_loss: Optional[float],
) -> None:
    """Save a resumable training checkpoint.

    Args:
        path: Destination checkpoint path.
        model: Model being trained.
        optimizer: Optimizer state to save.
        scheduler: Learning-rate scheduler state to save.
        scaler: AMP scaler state to save.
        model_config: Model configuration.
        training_config: Training configuration.
        global_step: Current optimizer step.
        epoch: Current epoch number.
        train_loss: Most recent training loss.
        val_loss: Most recent validation loss.
    """

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "model_config": dataclass_to_jsonable(model_config),
            "training_config": dataclass_to_jsonable(training_config),
            "global_step": global_step,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        path,
    )
