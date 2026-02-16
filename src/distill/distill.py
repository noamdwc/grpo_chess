"""Distillation training: train student ChessTransformer on teacher soft labels.

Usage:
    python -m src.distill.distill --config distill.yaml
"""

import argparse
import time
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader

from src.models import ChessTransformer, ChessTransformerConfig
from src.distill.distill_dataset import DistillDataset, collate_distill_batch
from src.configs.config_loader import load_yaml_file, dict_to_dataclass
from src.evaluator import Evaluator, StockfishEvalCallback
from src.eval_utils import EvalConfig
from src.chess.policy_player import PolicyConfig
from src.chess.stockfish import StockfishConfig


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


@dataclass
class DistillDatasetConfig:
    data_dir: str = "data/distill"
    top_k: int = 8
    eval_fraction: float = 0.05


class DistillChessTransformer(pl.LightningModule):
    """Lightning module for distilling teacher soft labels into student model."""

    def __init__(
        self,
        transformer_config: ChessTransformerConfig,
        distill_config: DistillConfig,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = ChessTransformer(transformer_config)
        self.distill_config = distill_config
        self.transformer_config = transformer_config

        if distill_config.pretrain_checkpoint:
            self._load_pretrained_weights(distill_config.pretrain_checkpoint)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _load_pretrained_weights(self, checkpoint_path: str) -> None:
        print(f"Loading pretrained weights from: {checkpoint_path}")
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

        # Cross-entropy with teacher soft labels (masked by k_mask)
        loss = -(teacher_probs * student_at_teacher * k_mask.float()).sum(dim=1).mean()

        # Metrics
        with torch.no_grad():
            # Student's top-1 prediction
            student_top1 = masked_logits.argmax(dim=-1)  # [B]
            # Teacher's top-1 (first entry in teacher_indices)
            teacher_top1 = teacher_indices[:, 0]  # [B]
            top1_match = (student_top1 == teacher_top1).float().mean()

            # Top-5 match: does teacher's top-1 appear in student's top-5?
            _, student_top5 = masked_logits.topk(5, dim=-1)  # [B, 5]
            top5_match = (student_top5 == teacher_top1.unsqueeze(-1)).any(dim=-1).float().mean()

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
            teacher_log_probs = torch.log(teacher_probs.clamp(min=1e-10))
            kl_per_sample = (
                teacher_probs * (teacher_log_probs - student_at_teacher) * k_mask.float()
            ).sum(dim=1)
            kl_divergence = kl_per_sample.mean()

        metrics = {
            "top1_match": top1_match,
            "top5_match": top5_match,
            "entropy": entropy,
            "kl_divergence": kl_divergence,
        }
        return loss, metrics

    def training_step(self, batch, batch_idx):
        board_tokens, legal_masks, teacher_indices, teacher_probs, k_mask = batch
        logits = self(board_tokens)
        loss, metrics = self._compute_loss(logits, legal_masks, teacher_indices, teacher_probs, k_mask)

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/top1_match", metrics["top1_match"], prog_bar=True)
        self.log("train/top5_match", metrics["top5_match"])
        self.log("train/entropy", metrics["entropy"])
        self.log("train/kl_divergence", metrics["kl_divergence"])
        return loss

    def validation_step(self, batch, batch_idx):
        board_tokens, legal_masks, teacher_indices, teacher_probs, k_mask = batch
        logits = self(board_tokens)
        loss, metrics = self._compute_loss(logits, legal_masks, teacher_indices, teacher_probs, k_mask)

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/top1_match", metrics["top1_match"], prog_bar=True, sync_dist=True)
        self.log("val/top5_match", metrics["top5_match"], sync_dist=True)
        self.log("val/entropy", metrics["entropy"], sync_dist=True)
        self.log("val/kl_divergence", metrics["kl_divergence"], sync_dist=True)
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

    model = DistillChessTransformer(transformer_config, distill_config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create datasets (loads shards once for both splits)
    train_dataset, val_dataset = DistillDataset.load_train_eval(
        dataset_config.data_dir, eval_fraction=dataset_config.eval_fraction
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
            log_model=True,
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

    evaluator = Evaluator(eval_cfg=eval_cfg, policy_cfg=policy_cfg, stockfish_cfg=stockfish_cfg)
    trainer.callbacks.append(StockfishEvalCallback(
        evaluator, every_n_epochs=distill_config.eval_every_n_epochs,
    ))

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

    train(distill_config, dataset_config, transformer_config, eval_cfg, stockfish_cfg, policy_cfg)


if __name__ == "__main__":
    main()
