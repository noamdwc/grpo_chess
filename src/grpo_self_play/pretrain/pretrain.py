"""Pretraining script for chess model on Lichess games using PyTorch Lightning.

This script trains the ChessTransformer model using supervised learning
on expert moves from Lichess games before GRPO reinforcement learning.

Usage:
    python -m src.grpo_self_play.pretrain.pretrain --config pretrain.yaml

    # Or with overrides:
    python -m src.grpo_self_play.pretrain.pretrain --config pretrain.yaml \
        --lr 1e-4 --batch_size 512 --min_elo 1800
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader

from src.grpo_self_play.models import ChessTransformer, ChessTransformerConfig
from src.grpo_self_play.pretrain.pretrain_dataset import (
    ChessPretrainDataset,
    PretrainDatasetConfig,
    collate_pretrain_batch,
)
from src.grpo_self_play.configs.config_loader import (
    load_yaml_file,
    dict_to_dataclass,
)


@dataclass
class PretrainConfig:
    """Configuration for pretraining.

    Attributes:
        lr: Learning rate
        batch_size: Batch size for training
        num_epochs: Number of epochs to train
        warmup_steps: Number of warmup steps for learning rate
        weight_decay: Weight decay for AdamW
        max_grad_norm: Maximum gradient norm for clipping
        checkpoint_dir: Directory to save checkpoints
        resume_from: Path to checkpoint to resume from
        use_wandb: Whether to use Weights & Biases logging
        wandb_project: WandB project name
        label_smoothing: Label smoothing factor for cross-entropy
        num_workers: Number of DataLoader workers
        val_check_interval: Validation check interval (fraction of epoch or int steps)
    """
    lr: float = 1e-4
    batch_size: int = 256
    num_epochs: int = 1
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    checkpoint_dir: str = "checkpoints/pretrain"
    resume_from: Optional[str] = None
    use_wandb: bool = True
    wandb_project: str = "chess-grpo-pretrain"
    label_smoothing: float = 0.1
    num_workers: int = 4
    val_check_interval: float = 0.1


# Register as safe for torch.load with weights_only=True (PyTorch 2.6+ compatibility)
torch.serialization.add_safe_globals([PretrainConfig])


class PretrainChessTransformer(pl.LightningModule):
    """PyTorch Lightning module for pretraining chess policy with supervised learning.

    This module implements supervised learning on expert chess moves from Lichess games.
    The pretrained model can then be fine-tuned with GRPO reinforcement learning.

    Attributes:
        model: The ChessTransformer policy model
        pretrain_config: Pretraining configuration
        transformer_config: Model architecture configuration
    """

    def __init__(
        self,
        transformer_config: ChessTransformerConfig,
        pretrain_config: PretrainConfig,
    ):
        """Initialize pretraining module.

        Args:
            transformer_config: Configuration for the chess transformer model
            pretrain_config: Pretraining configuration
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = ChessTransformer(transformer_config)
        self.pretrain_config = pretrain_config
        self.transformer_config = transformer_config

        # For warmup scheduler
        self._num_training_steps = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor [batch, seq_len]

        Returns:
            Policy logits [batch, action_dim]
        """
        return self.model(x)

    def _compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        legal_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute cross-entropy loss with legal move masking.

        Args:
            logits: Model output logits [B, num_actions]
            targets: Target action indices [B]
            legal_masks: Legal moves mask [B, num_actions]

        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Validate shapes match
        B, action_dim = logits.shape
        if legal_masks.shape != (B, action_dim):
            raise ValueError(
                f"Shape mismatch: logits {logits.shape} vs legal_masks {legal_masks.shape}. "
                f"Expected legal_masks to be [{B}, {action_dim}]"
            )
        if targets.shape != (B,):
            raise ValueError(
                f"Shape mismatch: targets {targets.shape} vs expected [{B}]"
            )
        
        # Validate target actions are within bounds
        max_target = targets.max().item()
        min_target = targets.min().item()
        if max_target >= action_dim or min_target < 0:
            raise ValueError(
                f"Target action indices out of bounds: min={min_target}, max={max_target}, "
                f"action_dim={action_dim}. This suggests a mismatch between dataset action "
                f"space and model action_dim."
            )
        
        # Validate target actions are legal (should always be true, but check defensively)
        target_legal = legal_masks.gather(1, targets.unsqueeze(1)).squeeze(1)
        if not target_legal.all():
            illegal_count = (~target_legal).sum().item()
            illegal_indices = (~target_legal).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                f"Found {illegal_count} illegal target actions in batch (out of {B}). "
                f"First few batch indices: {illegal_indices[:10]}. "
                f"This should not happen - dataset should filter these out."
            )
        
        # Check for NaN or Inf in raw logits (before masking)
        if not torch.isfinite(logits).all():
            nan_count = (~torch.isfinite(logits)).sum().item()
            raise ValueError(
                f"Found {nan_count} non-finite values in raw logits before masking. "
                f"This suggests the model is outputting NaN/Inf."
            )
        
        # Mask illegal moves to -inf
        masked_logits = logits.masked_fill(~legal_masks, float('-inf'))
        
        # Check that each sample has at least one legal move (before checking masked logits)
        legal_per_sample = legal_masks.sum(dim=1)
        if (legal_per_sample == 0).any():
            empty_samples = (legal_per_sample == 0).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                f"Found {len(empty_samples)} samples with no legal moves. "
                f"Batch indices: {empty_samples[:10]}. This should not happen."
            )
        
        # Check masked logits: each sample must have at least one finite logit (legal move)
        finite_per_sample = torch.isfinite(masked_logits).sum(dim=1)
        if (finite_per_sample == 0).any():
            bad_samples = (finite_per_sample == 0).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                f"Found {len(bad_samples)} samples with all -inf logits after masking. "
                f"Batch indices: {bad_samples[:10]}. This means no legal moves have finite logits."
            )
        
        # Ensure target actions are not masked (defensive check)
        target_logits = masked_logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        if not torch.isfinite(target_logits).all():
            inf_count = (~torch.isfinite(target_logits)).sum().item()
            raise ValueError(
                f"Found {inf_count} target actions with -inf logits after masking. "
                f"This means target actions are being masked as illegal, which should not happen."
            )

        # Compute NLL loss (works correctly with -inf masked logits)
        nll_loss = F.cross_entropy(masked_logits, targets, reduction='mean')

        # Apply label smoothing only over legal moves to avoid inf from -inf logits
        # Standard F.cross_entropy with label_smoothing averages log_softmax over ALL
        # actions, but -inf logits cause smooth_loss = +inf
        eps = self.pretrain_config.label_smoothing
        if eps > 0:
            # Compute log_softmax (illegal moves will be -inf)
            log_probs = F.log_softmax(masked_logits, dim=-1)
            # Zero out illegal moves so they don't contribute to smoothing term
            log_probs_legal = log_probs.masked_fill(~legal_masks, 0.0)
            # Average only over legal moves
            num_legal = legal_masks.sum(dim=-1).float()  # [B]
            smooth_loss = -log_probs_legal.sum(dim=-1) / num_legal  # [B]
            loss = (1 - eps) * nll_loss + eps * smooth_loss.mean()
        else:
            loss = nll_loss

        # Check if loss is infinite or NaN
        if not torch.isfinite(loss):
            # Additional debugging info
            target_logits_debug = masked_logits.gather(1, targets.unsqueeze(1)).squeeze(1)
            print(f"DEBUG: Loss is {loss.item()}")
            print(f"DEBUG: NLL loss: {nll_loss.item()}")
            if eps > 0:
                print(f"DEBUG: Smooth loss mean: {smooth_loss.mean().item()}")
            print(f"DEBUG: Logits shape: {logits.shape}")
            print(f"DEBUG: Legal masks shape: {legal_masks.shape}")
            print(f"DEBUG: Targets range: [{targets.min().item()}, {targets.max().item()}]")
            print(f"DEBUG: Target logits range: [{target_logits_debug.min().item():.2f}, {target_logits_debug.max().item():.2f}]")
            print(f"DEBUG: Legal moves per sample: min={legal_per_sample.min().item()}, max={legal_per_sample.max().item()}")
            raise ValueError(
                f"Loss is {loss.item()}. This can happen if:\n"
                f"1. Target actions are out of bounds\n"
                f"2. Target actions are masked as illegal\n"
                f"3. Model outputs contain NaN/Inf\n"
                f"4. All logits are -inf (no legal moves)"
            )

        # Compute metrics
        with torch.no_grad():
            # Top-1 accuracy
            predictions = masked_logits.argmax(dim=-1)
            accuracy = (predictions == targets).float().mean()

            # Top-5 accuracy
            _, top5_preds = masked_logits.topk(5, dim=-1)
            top5_correct = (top5_preds == targets.unsqueeze(-1)).any(dim=-1)
            top5_accuracy = top5_correct.float().mean()

            # Entropy of the distribution (measure of confidence)
            probs = F.softmax(masked_logits, dim=-1)
            log_probs = F.log_softmax(masked_logits, dim=-1)
            # Handle -inf * 0 = nan by replacing with 0
            entropy_terms = probs * log_probs
            entropy_terms = torch.where(
                torch.isfinite(entropy_terms),
                entropy_terms,
                torch.zeros_like(entropy_terms)
            )
            entropy = -entropy_terms.sum(dim=-1).mean()

            # Perplexity - clamp to avoid inf
            perplexity = torch.exp(loss.clamp(max=50))

        metrics = {
            'accuracy': accuracy,
            'top5_accuracy': top5_accuracy,
            'entropy': entropy,
            'perplexity': perplexity,
        }

        return loss, metrics

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Perform a training step.

        Args:
            batch: Tuple of (boards, actions, legal_masks)
            batch_idx: Batch index

        Returns:
            Loss value
        """
        boards, actions, legal_masks = batch

        # Forward pass
        logits = self(boards)

        # Compute loss and metrics
        loss, metrics = self._compute_loss(logits, actions, legal_masks)

        # Log metrics
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/accuracy', metrics['accuracy'], prog_bar=True)
        self.log('train/top5_accuracy', metrics['top5_accuracy'])
        self.log('train/entropy', metrics['entropy'])
        self.log('train/perplexity', metrics['perplexity'])

        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Perform a validation step.

        Args:
            batch: Tuple of (boards, actions, legal_masks)
            batch_idx: Batch index

        Returns:
            Loss value
        """
        boards, actions, legal_masks = batch

        # Forward pass
        logits = self(boards)

        # Compute loss and metrics
        loss, metrics = self._compute_loss(logits, actions, legal_masks)

        # Log metrics
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        self.log('val/accuracy', metrics['accuracy'], prog_bar=True, sync_dist=True)
        self.log('val/top5_accuracy', metrics['top5_accuracy'], sync_dist=True)
        self.log('val/entropy', metrics['entropy'], sync_dist=True)
        self.log('val/perplexity', metrics['perplexity'], sync_dist=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler.

        Returns:
            Dictionary with optimizer and lr_scheduler configuration
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.pretrain_config.lr,
            weight_decay=self.pretrain_config.weight_decay,
        )

        # Linear warmup + cosine decay scheduler
        def lr_lambda(current_step: int) -> float:
            warmup_steps = self.pretrain_config.warmup_steps
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0  # After warmup, use constant LR (or add cosine decay)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }


def get_pretrain_trainer(
    pretrain_config: PretrainConfig,
    run_name: str,
) -> pl.Trainer:
    """Create a PyTorch Lightning trainer for pretraining.

    Args:
        pretrain_config: Pretraining configuration
        run_name: Name for this training run

    Returns:
        Configured PyTorch Lightning trainer
    """
    # Create checkpoint directory
    checkpoint_dir = Path(pretrain_config.checkpoint_dir)
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
        LearningRateMonitor(logging_interval='step'),
    ]

    logger = None
    if pretrain_config.use_wandb:
        logger = WandbLogger(
            project=pretrain_config.wandb_project,
            name=run_name,
            log_model=True,
        )

    trainer = pl.Trainer(
        max_epochs=pretrain_config.num_epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=pretrain_config.max_grad_norm,
        log_every_n_steps=50,
        val_check_interval=pretrain_config.val_check_interval,
    )

    return trainer


def load_pretrain_config(
    path: str = "pretrain.yaml",
    overrides: dict = None,
) -> tuple[PretrainConfig, PretrainDatasetConfig, ChessTransformerConfig]:
    """Load pretraining configuration from YAML file.

    Args:
        path: Path to config file (relative to configs dir or absolute)
        overrides: Optional dict of overrides

    Returns:
        Tuple of (PretrainConfig, PretrainDatasetConfig, ChessTransformerConfig)
    """
    data = load_yaml_file(path)

    if overrides:
        for section, section_overrides in overrides.items():
            if section in data:
                data[section].update(section_overrides)
            else:
                data[section] = section_overrides

    pretrain = dict_to_dataclass(PretrainConfig, data.get('pretrain', {}))
    dataset = dict_to_dataclass(PretrainDatasetConfig, data.get('dataset', {}))
    transformer = dict_to_dataclass(ChessTransformerConfig, data.get('transformer', {}))

    return pretrain, dataset, transformer


def train(
    pretrain_config: PretrainConfig,
    dataset_config: PretrainDatasetConfig,
    transformer_config: ChessTransformerConfig,
) -> str:
    """Main pretraining function.

    Args:
        pretrain_config: Pretraining configuration
        dataset_config: Dataset configuration
        transformer_config: Model configuration

    Returns:
        Path to final checkpoint
    """
    import time
    import random
    import string

    # Generate run name
    timestamp = time.strftime("%Y%m%d-%H%M")
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    run_name = f"pretrain-{timestamp}-{random_suffix}"
    print(f"Run name: {run_name}")

    # Create model
    model = PretrainChessTransformer(transformer_config, pretrain_config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create datasets
    train_dataset = ChessPretrainDataset(dataset_config)

    # Create validation dataset using hash-based split
    val_dataset_config = PretrainDatasetConfig(
        min_elo=dataset_config.min_elo,
        max_samples=10000,  # Smaller validation set
        skip_first_n_moves=dataset_config.skip_first_n_moves,
        skip_last_n_moves=dataset_config.skip_last_n_moves,
        sample_positions_per_game=1,  # Less samples per game for validation
        is_eval=True,  # Use eval portion of hash-based split
        eval_fraction=dataset_config.eval_fraction,
        cache_path=dataset_config.cache_path,
    )
    val_dataset = ChessPretrainDataset(val_dataset_config)
    print(f"Train: {len(train_dataset):,} samples, Eval: {len(val_dataset):,} samples")

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=pretrain_config.batch_size,
        shuffle=True,  # Shuffle for training
        num_workers=pretrain_config.num_workers,
        collate_fn=collate_pretrain_batch,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=pretrain_config.batch_size,
        shuffle=False,
        num_workers=max(1, pretrain_config.num_workers // 2),
        collate_fn=collate_pretrain_batch,
        pin_memory=True,
    )

    # Create trainer
    trainer = get_pretrain_trainer(pretrain_config, run_name)

    # Resume from checkpoint if specified
    ckpt_path = pretrain_config.resume_from

    # Train
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)

    # Save final checkpoint in a standard location
    final_path = Path(pretrain_config.checkpoint_dir) / "pretrain_final.pt"
    torch.save({
        'model_state_dict': model.model.state_dict(),
        'transformer_config': transformer_config,
        'pretrain_config': pretrain_config,
    }, final_path)

    print(f"\nPretraining complete! Final checkpoint saved to {final_path}")
    return str(final_path)


def main():
    """Main entry point for pretraining script."""
    parser = argparse.ArgumentParser(description="Pretrain chess model on Lichess games")
    parser.add_argument("--config", type=str, default="pretrain.yaml",
                        help="Path to config file")

    # Allow command-line overrides for common parameters
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs")
    parser.add_argument("--min_elo", type=int, help="Minimum player ELO")
    parser.add_argument("--max_samples", type=int, help="Max samples per epoch")
    parser.add_argument("--resume_from", type=str, help="Resume from checkpoint")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")

    args = parser.parse_args()

    # Build overrides from command-line arguments
    overrides = {'pretrain': {}, 'dataset': {}}

    if args.lr:
        overrides['pretrain']['lr'] = args.lr
    if args.batch_size:
        overrides['pretrain']['batch_size'] = args.batch_size
    if args.num_epochs:
        overrides['pretrain']['num_epochs'] = args.num_epochs
    if args.resume_from:
        overrides['pretrain']['resume_from'] = args.resume_from
    if args.no_wandb:
        overrides['pretrain']['use_wandb'] = False
    if args.min_elo:
        overrides['dataset']['min_elo'] = args.min_elo
    if args.max_samples:
        overrides['dataset']['max_samples'] = args.max_samples

    # Load config
    pretrain_config, dataset_config, transformer_config = load_pretrain_config(
        args.config,
        overrides=overrides if any(v for v in overrides.values()) else None
    )

    # Run training
    train(pretrain_config, dataset_config, transformer_config)


if __name__ == "__main__":
    main()
