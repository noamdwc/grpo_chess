"""Distillation training: train student ChessTransformer on teacher soft labels.

Usage:
    python -m src.distill.distill --config distill.yaml
"""

import argparse
import math
import os
import time
import random
import string
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader

from src.models import ChessTransformer, ChessTransformerConfig
from src.checkpoint_compat import register_legacy_checkpoint_aliases
from src.distill.distill_dataset import DistillDataset, collate_distill_batch
from src.configs.config_loader import load_yaml_file, dict_to_dataclass
from src.evaluator import Evaluator, StockfishEvalCallback
from src.eval_utils import EvalConfig
from src.chess.policy_player import PolicyConfig
from src.chess.stockfish import StockfishConfig, resolve_stockfish_path


@dataclass
class DistillConfig:
    lr: float = 1e-4
    batch_size: int = 2048
    num_epochs: int = 10
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    checkpoint_dir: str = "checkpoints/distill"
    resume_from: Optional[str] = None
    pretrain_checkpoint: Optional[str] = None
    use_wandb: bool = True
    wandb_project: str = "chess-grpo-pretrain"
    num_workers: int = 4
    val_check_interval: float = 0.1
    eval_every_n_epochs: int = 1
    log_model_artifacts: bool = False
    auto_disable_wandb_if_missing_key: bool = True
    fail_on_nonfinite: bool = False
    max_nonfinite_batches: int = 50
    require_stockfish_eval: bool = False
    quality_gate_min_val_top1: float = 0.0
    quality_gate_epoch: int = 0


@dataclass
class DistillDatasetConfig:
    data_dir: str = "data/distill"
    top_k: int = 8
    eval_fraction: float = 0.05
    max_shards: Optional[int] = None


def ensure_distill_dataset(config_path: str, dataset_config: "DistillDatasetConfig") -> None:
    """Generate distillation shards from DeepMind data if dataset dir is empty."""
    data_dir = Path(dataset_config.data_dir)
    shard_files = sorted(data_dir.glob("shard_*.pt")) if data_dir.exists() else []
    if shard_files:
        return

    print(f"No distill shards found in {dataset_config.data_dir}; generating from deepmind_data config.")
    from src.distill.convert_deepmind_data import ConvertConfig, convert

    data = load_yaml_file(config_path)
    convert_cfg = dict_to_dataclass(ConvertConfig, data.get("deepmind_data", {}))
    convert_cfg.output_dir = dataset_config.data_dir
    convert(convert_cfg)


class DistillChessTransformer(pl.LightningModule):
    """Lightning module for distilling teacher soft labels into student model."""

    def __init__(
        self,
        transformer_config: ChessTransformerConfig,
        distill_config: DistillConfig,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["distill_config"])
        self.model = ChessTransformer(transformer_config)
        self.distill_config = distill_config
        self.transformer_config = transformer_config
        self._train_nonfinite_batches = 0
        self._val_nonfinite_batches = 0

        if distill_config.pretrain_checkpoint:
            self._load_pretrained_weights(distill_config.pretrain_checkpoint)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _module_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _finite_scalar(self, value: torch.Tensor | float, *, default: float = 0.0) -> torch.Tensor:
        """Convert any scalar-like value into a finite tensor for logging."""
        device = self._module_device()
        if isinstance(value, torch.Tensor):
            scalar = value.mean() if value.numel() > 1 else value
            if torch.isfinite(scalar):
                return scalar
            return torch.tensor(default, dtype=torch.float32, device=device)

        as_float = float(value)
        if math.isfinite(as_float):
            return torch.tensor(as_float, dtype=torch.float32, device=device)
        return torch.tensor(default, dtype=torch.float32, device=device)

    def _sanitize_metrics(self, metrics: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: self._finite_scalar(value, default=0.0) for name, value in metrics.items()}

    def _handle_nonfinite_training(self, reason: str) -> torch.Tensor:
        self._train_nonfinite_batches += 1
        count = float(self._train_nonfinite_batches)
        self.log("train/nonfinite_batches", count, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/loss", self._finite_scalar(float("nan"), default=1_000_000.0), prog_bar=True)
        print(f"Warning: skipping non-finite training batch ({reason}); count={self._train_nonfinite_batches}")

        if self.distill_config.fail_on_nonfinite:
            raise RuntimeError(f"Encountered non-finite training batch and fail_on_nonfinite=true ({reason})")
        if self._train_nonfinite_batches > self.distill_config.max_nonfinite_batches:
            raise RuntimeError(
                "Exceeded max_nonfinite_batches "
                f"({self._train_nonfinite_batches}>{self.distill_config.max_nonfinite_batches})"
            )
        # Return connected zero loss so optimizer step is a no-op without crashing autograd.
        return next(self.model.parameters()).sum() * 0.0

    def _handle_nonfinite_validation(self, reason: str) -> torch.Tensor:
        self._val_nonfinite_batches += 1
        count = float(self._val_nonfinite_batches)
        self.log("val/nonfinite_batches", count, prog_bar=False, sync_dist=True)
        self.log("val/loss", self._finite_scalar(float("nan"), default=1_000_000.0), prog_bar=True, sync_dist=True)
        print(f"Warning: non-finite validation batch ({reason}); count={self._val_nonfinite_batches}")
        return self._finite_scalar(1_000_000.0, default=1_000_000.0)

    def _load_pretrained_weights(self, checkpoint_path: str) -> None:
        print(f"Loading pretrained weights from: {checkpoint_path}")
        register_legacy_checkpoint_aliases()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = {}
            for k, v in checkpoint["state_dict"].items():
                if k.startswith("model."):
                    state_dict[k[6:]] = v
                elif k.startswith("policy_model."):
                    state_dict[k[13:]] = v
        else:
            state_dict = checkpoint

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning: missing keys: {missing}")
        if unexpected:
            print(f"Warning: unexpected keys: {unexpected}")
        print("Pretrained weights loaded.")

    def _compute_loss(
        self,
        logits: torch.Tensor,
        legal_masks: torch.Tensor,
        teacher_indices: torch.Tensor,
        teacher_probs: torch.Tensor,
        k_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute KD cross-entropy loss with teacher soft labels.

        Args:
            logits: Student output [B, 1968]
            legal_masks: Legal moves mask [B, 1968]
            teacher_indices: Teacher top-k action indices [B, max_k]
            teacher_probs: Teacher soft probabilities [B, max_k]
            k_mask: Valid entries mask [B, max_k]

        Returns:
            (loss, metrics_dict)
        """
        # Mask illegal moves
        masked_logits = logits.masked_fill(~legal_masks, float("-inf"))
        student_log_probs = F.log_softmax(masked_logits, dim=-1)  # [B, 1968]

        # Gather student log-probs at teacher action indices
        student_at_teacher = student_log_probs.gather(1, teacher_indices)  # [B, max_k]

        # Keep only teacher targets that are legal under this board's legal mask.
        teacher_is_legal = k_mask & legal_masks.gather(1, teacher_indices)
        teacher_probs_filtered = teacher_probs * teacher_is_legal.float()
        teacher_mass = teacher_probs_filtered.sum(dim=1)  # [B]
        valid_sample_mask = teacher_mass > 1e-8

        if not bool(valid_sample_mask.any()):
            raise ValueError(
                "No valid teacher targets in batch. "
                "Check distillation data generation/conversion legality filtering."
            )

        teacher_probs_norm = teacher_probs_filtered / teacher_mass.clamp_min(1e-8).unsqueeze(1)
        student_at_teacher = torch.where(
            teacher_is_legal, student_at_teacher, torch.zeros_like(student_at_teacher)
        )

        # Cross-entropy with legality-filtered, renormalized teacher soft labels
        loss_per_sample = -(teacher_probs_norm * student_at_teacher).sum(dim=1)
        loss = loss_per_sample[valid_sample_mask].mean()

        # Metrics
        with torch.no_grad():
            # Student's top-1 prediction
            student_top1 = masked_logits.argmax(dim=-1)  # [B]
            # Teacher's top-1 legal entry
            first_legal_idx = teacher_is_legal.float().argmax(dim=1, keepdim=True)
            teacher_top1 = teacher_indices.gather(1, first_legal_idx).squeeze(1)  # [B]

            if bool(valid_sample_mask.any()):
                top1_match = (student_top1[valid_sample_mask] == teacher_top1[valid_sample_mask]).float().mean()
            else:
                top1_match = torch.tensor(0.0, device=logits.device)

            # Top-5 match: does teacher's top-1 appear in student's top-5?
            _, student_top5 = masked_logits.topk(5, dim=-1)  # [B, 5]
            top5_hit = (student_top5 == teacher_top1.unsqueeze(-1)).any(dim=-1).float()
            if bool(valid_sample_mask.any()):
                top5_match = top5_hit[valid_sample_mask].mean()
            else:
                top5_match = torch.tensor(0.0, device=logits.device)

            # Student entropy
            probs = F.softmax(masked_logits, dim=-1)
            log_probs_full = F.log_softmax(masked_logits, dim=-1)
            entropy_terms = probs * log_probs_full
            entropy_terms = torch.where(
                torch.isfinite(entropy_terms),
                entropy_terms,
                torch.zeros_like(entropy_terms),
            )
            entropy = -entropy_terms.sum(dim=-1).mean()

            # KL divergence: D_KL(teacher || student) over teacher's top-k
            # = sum_k teacher_prob * (log(teacher_prob) - student_log_prob)
            teacher_log_probs = torch.log(teacher_probs_norm.clamp(min=1e-10))
            kl_per_sample = (teacher_probs_norm * (teacher_log_probs - student_at_teacher)).sum(dim=1)
            kl_divergence = kl_per_sample[valid_sample_mask].mean()

            teacher_entries_total = k_mask.float().sum().clamp_min(1.0)
            teacher_legal_fraction = teacher_is_legal.float().sum() / teacher_entries_total
            teacher_valid_sample_fraction = valid_sample_mask.float().mean()

        metrics = {
            "top1_match": top1_match,
            "top5_match": top5_match,
            "entropy": entropy,
            "kl_divergence": kl_divergence,
            "teacher_legal_fraction": teacher_legal_fraction,
            "teacher_valid_sample_fraction": teacher_valid_sample_fraction,
        }
        return loss, metrics

    def training_step(self, batch, batch_idx):
        board_tokens, legal_masks, teacher_indices, teacher_probs, k_mask = batch
        logits = self(board_tokens)
        if not torch.isfinite(logits).all():
            return self._handle_nonfinite_training("non-finite logits")
        try:
            loss, metrics = self._compute_loss(logits, legal_masks, teacher_indices, teacher_probs, k_mask)
        except ValueError as exc:
            return self._handle_nonfinite_training(str(exc))
        if not torch.isfinite(loss):
            return self._handle_nonfinite_training("non-finite loss")
        metrics = self._sanitize_metrics(metrics)

        self.log("train/loss", self._finite_scalar(loss, default=1_000_000.0), prog_bar=True)
        self.log("train/top1_match", metrics["top1_match"], prog_bar=True)
        self.log("train/top5_match", metrics["top5_match"])
        self.log("train/entropy", metrics["entropy"])
        self.log("train/kl_divergence", metrics["kl_divergence"])
        self.log("train/teacher_legal_fraction", metrics["teacher_legal_fraction"])
        self.log("train/teacher_valid_sample_fraction", metrics["teacher_valid_sample_fraction"])
        return loss

    def validation_step(self, batch, batch_idx):
        board_tokens, legal_masks, teacher_indices, teacher_probs, k_mask = batch
        logits = self(board_tokens)
        if not torch.isfinite(logits).all():
            return self._handle_nonfinite_validation("non-finite logits")
        try:
            loss, metrics = self._compute_loss(logits, legal_masks, teacher_indices, teacher_probs, k_mask)
        except ValueError as exc:
            return self._handle_nonfinite_validation(str(exc))
        if not torch.isfinite(loss):
            return self._handle_nonfinite_validation("non-finite loss")
        metrics = self._sanitize_metrics(metrics)

        self.log("val/loss", self._finite_scalar(loss, default=1_000_000.0), prog_bar=True, sync_dist=True)
        self.log("val/top1_match", metrics["top1_match"], prog_bar=True, sync_dist=True)
        self.log("val/top5_match", metrics["top5_match"], sync_dist=True)
        self.log("val/entropy", metrics["entropy"], sync_dist=True)
        self.log("val/kl_divergence", metrics["kl_divergence"], sync_dist=True)
        self.log("val/teacher_legal_fraction", metrics["teacher_legal_fraction"], sync_dist=True)
        self.log("val/teacher_valid_sample_fraction", metrics["teacher_valid_sample_fraction"], sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.distill_config.lr,
            weight_decay=self.distill_config.weight_decay,
        )

        def lr_lambda(current_step: int) -> float:
            warmup_steps = self.distill_config.warmup_steps
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def load_distill_config(
    path: str = "distill.yaml",
    overrides: dict = None,
) -> tuple[DistillConfig, DistillDatasetConfig, ChessTransformerConfig, EvalConfig, StockfishConfig, PolicyConfig]:
    data = load_yaml_file(path)

    if overrides:
        for section, section_overrides in overrides.items():
            if section in data:
                data[section].update(section_overrides)
            else:
                data[section] = section_overrides

    distill = dict_to_dataclass(DistillConfig, data.get("distill", {}))
    dataset = dict_to_dataclass(DistillDatasetConfig, data.get("dataset", {}))
    transformer = dict_to_dataclass(ChessTransformerConfig, data.get("transformer", {}))
    eval_cfg = dict_to_dataclass(EvalConfig, data.get("eval", {}))
    stockfish_cfg = dict_to_dataclass(StockfishConfig, data.get("stockfish", {}))
    policy_cfg = dict_to_dataclass(PolicyConfig, data.get("policy", {}))

    return distill, dataset, transformer, eval_cfg, stockfish_cfg, policy_cfg


def prepare_wandb_config(distill_config: DistillConfig) -> DistillConfig:
    """Normalize W&B env/key behavior and optionally disable logging when key is missing."""
    if not distill_config.use_wandb:
        return distill_config

    if "WANDB_API_KEY" not in os.environ and "WANDB_KEY" in os.environ:
        os.environ["WANDB_API_KEY"] = os.environ["WANDB_KEY"]

    if "WANDB_API_KEY" in os.environ:
        return distill_config

    if distill_config.auto_disable_wandb_if_missing_key:
        print("Warning: WANDB_API_KEY not found. Disabling wandb for this run.")
        return replace(distill_config, use_wandb=False)

    raise RuntimeError(
        "WANDB_API_KEY not found and auto_disable_wandb_if_missing_key=false. "
        "Set WANDB_API_KEY (or WANDB_KEY) or disable wandb."
    )


def build_stockfish_eval_callback(
    distill_config: DistillConfig,
    eval_cfg: EvalConfig,
    stockfish_cfg: StockfishConfig,
    policy_cfg: PolicyConfig,
) -> Optional[StockfishEvalCallback]:
    """Create Stockfish eval callback, optionally hard-failing when binary is missing."""
    try:
        resolved_stockfish = resolve_stockfish_path(stockfish_cfg.path)
    except FileNotFoundError as exc:
        if distill_config.require_stockfish_eval:
            raise RuntimeError(
                "Stockfish evaluation is required but no Stockfish binary was found. "
                "Set stockfish.path/STOCKFISH_PATH or install stockfish."
            ) from exc
        print(f"Warning: {exc}")
        print("Warning: Stockfish evaluation callback disabled for this run.")
        return None

    stockfish_cfg = replace(stockfish_cfg, path=resolved_stockfish)
    print(f"Stockfish evaluation enabled with binary: {resolved_stockfish}")
    evaluator = Evaluator(eval_cfg=eval_cfg, policy_cfg=policy_cfg, stockfish_cfg=stockfish_cfg)
    return StockfishEvalCallback(evaluator, every_n_epochs=distill_config.eval_every_n_epochs)


class DistillQualityGateCallback(Callback):
    """Fail fast when val/top1_match stays below a configured threshold."""

    def __init__(self, min_val_top1: float, gate_epoch: int):
        super().__init__()
        self.min_val_top1 = float(min_val_top1)
        self.gate_epoch = int(gate_epoch)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.min_val_top1 <= 0.0 or self.gate_epoch <= 0:
            return

        completed_epoch = trainer.current_epoch + 1  # current_epoch is 0-based.
        if completed_epoch < self.gate_epoch:
            return

        metric = trainer.callback_metrics.get("val/top1_match")
        if metric is None:
            raise RuntimeError(
                "Distill quality gate expected metric 'val/top1_match' but it was not logged."
            )

        metric_value = float(metric.detach().cpu() if isinstance(metric, torch.Tensor) else metric)
        if not math.isfinite(metric_value):
            raise RuntimeError(
                f"Distill quality gate metric val/top1_match is non-finite at epoch {completed_epoch}."
            )

        if metric_value + 1e-8 < self.min_val_top1:
            raise RuntimeError(
                "Distill quality gate failed: "
                f"val/top1_match={metric_value:.4f} < required {self.min_val_top1:.4f} "
                f"at epoch {completed_epoch}."
            )


def train(
    distill_config: DistillConfig,
    dataset_config: DistillDatasetConfig,
    transformer_config: ChessTransformerConfig,
    eval_cfg: EvalConfig = EvalConfig(),
    stockfish_cfg: StockfishConfig = StockfishConfig(),
    policy_cfg: PolicyConfig = PolicyConfig(),
) -> str:
    timestamp = time.strftime("%Y%m%d-%H%M")
    random_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    run_name = f"distill-{timestamp}-{random_suffix}"
    print(f"Run name: {run_name}")
    distill_config = prepare_wandb_config(distill_config)

    model = DistillChessTransformer(transformer_config, distill_config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create datasets (loads shards once for both splits)
    train_dataset, val_dataset = DistillDataset.load_train_eval(
        dataset_config.data_dir,
        eval_fraction=dataset_config.eval_fraction,
        max_shards=dataset_config.max_shards,
    )
    print(f"Train: {len(train_dataset):,} samples, Eval: {len(val_dataset):,} samples")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=distill_config.batch_size,
        shuffle=True,
        num_workers=distill_config.num_workers,
        collate_fn=collate_distill_batch,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=distill_config.batch_size,
        shuffle=False,
        num_workers=distill_config.num_workers,
        collate_fn=collate_distill_batch,
        pin_memory=True,
    )

    # Trainer
    checkpoint_dir = Path(distill_config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename=run_name + "-{epoch:02d}-{train/loss:.4f}",
            save_top_k=3,
            monitor="train/loss",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    logger = None
    if distill_config.use_wandb:
        logger = WandbLogger(
            project=distill_config.wandb_project,
            name=run_name,
            log_model=distill_config.log_model_artifacts,
        )

    stockfish_eval_callback = build_stockfish_eval_callback(
        distill_config=distill_config,
        eval_cfg=eval_cfg,
        stockfish_cfg=stockfish_cfg,
        policy_cfg=policy_cfg,
    )
    if stockfish_eval_callback is not None:
        callbacks.append(stockfish_eval_callback)
    if distill_config.quality_gate_min_val_top1 > 0.0 and distill_config.quality_gate_epoch > 0:
        callbacks.append(
            DistillQualityGateCallback(
                min_val_top1=distill_config.quality_gate_min_val_top1,
                gate_epoch=distill_config.quality_gate_epoch,
            )
        )

    trainer = pl.Trainer(
        max_epochs=distill_config.num_epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=distill_config.max_grad_norm,
        log_every_n_steps=50,
        val_check_interval=distill_config.val_check_interval,
    )

    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=distill_config.resume_from)

    # Save final checkpoint
    final_path = checkpoint_dir / "distill_final.pt"
    torch.save(
        {
            "model_state_dict": model.model.state_dict(),
            "transformer_config": transformer_config,
            "distill_config": distill_config,
        },
        final_path,
    )
    print(f"\nDistillation complete! Final checkpoint saved to {final_path}")
    return str(final_path)


def main():
    parser = argparse.ArgumentParser(description="Distill teacher model into student")
    parser.add_argument("--config", type=str, default="distill.yaml", help="Path to config file")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs")
    parser.add_argument("--resume_from", type=str, help="Resume from checkpoint")
    parser.add_argument("--pretrain_checkpoint", type=str, help="Initialize from pretrained weights")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    overrides = {"distill": {}}
    if args.lr:
        overrides["distill"]["lr"] = args.lr
    if args.batch_size:
        overrides["distill"]["batch_size"] = args.batch_size
    if args.num_epochs:
        overrides["distill"]["num_epochs"] = args.num_epochs
    if args.resume_from:
        overrides["distill"]["resume_from"] = args.resume_from
    if args.pretrain_checkpoint:
        overrides["distill"]["pretrain_checkpoint"] = args.pretrain_checkpoint
    if args.no_wandb:
        overrides["distill"]["use_wandb"] = False

    distill_config, dataset_config, transformer_config, eval_cfg, stockfish_cfg, policy_cfg = load_distill_config(
        args.config, overrides=overrides if any(v for v in overrides.values()) else None
    )

    ensure_distill_dataset(args.config, dataset_config)
    train(distill_config, dataset_config, transformer_config, eval_cfg, stockfish_cfg, policy_cfg)


if __name__ == "__main__":
    main()
