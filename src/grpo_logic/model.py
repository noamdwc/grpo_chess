from typing import Optional
import torch
import pytorch_lightning as pl
import chess

from dataclasses import dataclass

from src.models import ChessTransformer, ChessTransformerConfig
from src.grpo_logic.loss import grpo_ppo_loss
from src.grpo_logic.mode_utils import temporary_eval
from src.grpo_logic.sampling import sample_trajectories_batched
from src.eval_utils import EvalConfig
from src.chess.policy_player import PolicyConfig
from src.chess.searcher import SearchConfig
from src.chess.stockfish import StockfishConfig
from src.pretrain.pretrain_load_config import PretrainLoadConfig
from src.checkpoint_compat import (
    load_checkpoint_with_compat,
    load_state_dict_with_checkpoint_compat,
    register_legacy_checkpoint_aliases,
)


@dataclass
class GRPOConfig:
    """Configuration for GRPO (Group Relative Policy Optimization) training.

    Attributes:
        lr: Learning rate for optimizer
        num_trajectories: Number of trajectory groups to sample per batch
        trajectory_depth: Maximum depth of each trajectory
        clip_ratio: PPO clipping ratio (epsilon)
        kl_coef: KL divergence penalty coefficient (beta)
        eval_every_n_epochs: Frequency of evaluation runs
        ppo_steps: Number of optimization steps per sampled trajectory batch
        rollout_temperature: Temperature for action sampling during rollouts (>1 increases exploration)
        enable_safety_checks: Whether to abort training when clip fraction stays too high
        safety_patience_steps: Number of training steps to tolerate violations before aborting
        max_clip_fraction: If mean_clip_fraction > this for too long -> abort
        teacher_forcing_prob: Probability of using Stockfish for rival (opponent) moves
        teacher_forcing_depth: Stockfish search depth for teacher forcing moves
    """
    lr: float = 1e-6
    num_trajectories: int = 4
    trajectory_depth: int = 5
    clip_ratio: float = 0.2
    kl_coef: float = 0.001
    eval_every_n_epochs: int = 10
    ppo_steps: int = 1
    rollout_temperature: float = 1.0

    # Safety checks on training dynamics
    enable_safety_checks: bool = False
    safety_patience_steps: int = 1000
    max_clip_fraction: float = 0.95

    # Teacher forcing: use Stockfish for rival moves during trajectory sampling
    teacher_forcing_prob: float = 0.0
    teacher_forcing_depth: int = 4


# Register as safe for torch.load with weights_only=True (PyTorch 2.6+ compatibility)
torch.serialization.add_safe_globals([GRPOConfig])


class GRPOChessTransformer(pl.LightningModule):
    """PyTorch Lightning module for training chess policy with GRPO.

    This module implements Group Relative Policy Optimization (GRPO) for training
    a chess transformer policy. It maintains both a current policy and an old policy
    for computing importance sampling ratios in the PPO loss.
    """
    automatic_optimization = False  # Manual optimization for ppo_steps

    def __init__(self,
                 transformer_config: ChessTransformerConfig,
                 grpo_config: GRPOConfig,
                 eval_cfg: EvalConfig | None = None,
                 stockfish_cfg: StockfishConfig | None = None,
                 policy_cfg: PolicyConfig | None = None,
                 searcher_cfg: SearchConfig | None = None,
                 pretrain_cfg: PretrainLoadConfig | None = None):
        super().__init__()
        self.save_hyperparameters()
        self.policy_model = ChessTransformer(transformer_config)
        self.old_policy_model = ChessTransformer(transformer_config)

        # Load pretrained weights if specified
        if pretrain_cfg and pretrain_cfg.checkpoint_path:
            self._load_pretrained_weights(pretrain_cfg)

        self._sync_old_policy()

        # Safety-check state
        self._high_clip_steps: int = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the current policy model.
        
        Args:
            x: Input tensor [batch, seq_len]
            
        Returns:
            Policy logits [batch, action_dim]
        """
        return self.policy_model(x)

    def _old_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the old (frozen) policy model.
        
        Args:
            x: Input tensor [batch, seq_len]
            
        Returns:
            Policy logits [batch, action_dim]
        """
        return self.old_policy_model(x)

    def _sync_old_policy(self) -> None:
        """Synchronize old policy model with current policy and freeze it."""
        self.old_policy_model.load_state_dict(self.policy_model.state_dict())
        # Freeze old policy parameters
        for param in self.old_policy_model.parameters():
            param.requires_grad = False

    def _load_pretrained_weights(self, pretrain_cfg: PretrainLoadConfig) -> None:
        """Load pretrained weights from a checkpoint.

        Args:
            pretrain_cfg: Pretrain configuration with checkpoint path and freeze settings
        """
        checkpoint_path = pretrain_cfg.checkpoint_path
        print(f"Loading pretrained weights from: {checkpoint_path}")

        register_legacy_checkpoint_aliases()
        checkpoint = load_checkpoint_with_compat(checkpoint_path, map_location="cpu", weights_only=False)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            # Lightning checkpoint format - extract policy_model weights
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('model.'):
                    # From PretrainChessTransformer
                    state_dict[k[6:]] = v  # Remove 'model.' prefix
                elif k.startswith('policy_model.'):
                    # From GRPOChessTransformer
                    state_dict[k[13:]] = v  # Remove 'policy_model.' prefix
        else:
            # Assume it's a raw state dict
            state_dict = checkpoint

        # Load into policy model
        missing, unexpected = load_state_dict_with_checkpoint_compat(
            self.policy_model,
            state_dict,
            checkpoint_path=checkpoint_path,
            strict=False,
        )
        if missing:
            print(f"Warning: Missing keys in pretrained checkpoint: {missing}")
        if unexpected:
            print(f"Warning: Unexpected keys in pretrained checkpoint: {unexpected}")

        print(f"Successfully loaded pretrained weights")

        # Optionally freeze transformer layers
        if pretrain_cfg.freeze_layers > 0:
            self._freeze_transformer_layers(pretrain_cfg.freeze_layers)

    def _freeze_transformer_layers(self, num_layers: int) -> None:
        """Freeze the first N transformer encoder layers.

        Args:
            num_layers: Number of layers to freeze (from the bottom)
        """
        # Freeze embedding and positional encoding
        for param in self.policy_model.embedding.parameters():
            param.requires_grad = False
        self.policy_model.pos_encoding.requires_grad = False

        # Freeze specified number of transformer layers
        for i, layer in enumerate(self.policy_model.transformer.layers):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
                print(f"Froze transformer layer {i}")

        # Count trainable parameters
        trainable = sum(p.numel() for p in self.policy_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.policy_model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def _log_rewards_metrics(self, batch_group_rewards: torch.Tensor, prefix: str = "train/") -> None:
        """Log reward statistics for monitoring training progress.
        
        Args:
            batch_group_rewards: Group rewards tensor [B, G]
            prefix: Prefix for log keys (default: "train/")
        """
        mean_r = batch_group_rewards.mean()
        best = batch_group_rewards.max()
        gap = best - mean_r

        self.log(prefix + "reward_mean", mean_r, prog_bar=True)
        self.log(prefix + "reward_std", batch_group_rewards.std())
        self.log(prefix + "reward_p50", batch_group_rewards.median())
        self.log(prefix + "reward_p90", batch_group_rewards.quantile(0.9))
        self.log(prefix + "reward_best", best)
        self.log(prefix + "reward_gap", gap)

    def on_train_epoch_start(self) -> None:
        """Called at the start of each training epoch. Syncs old policy."""
        self._sync_old_policy()

    def _ppo_step(
        self,
        trajectories_states: torch.Tensor,
        trajectories_actions: torch.Tensor,
        trajectories_old_log_probs: torch.Tensor,
        trajectories_legal_masks: torch.Tensor | None,
        step_rewards: torch.Tensor,
        effective_pad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, object]:
        """Perform a single PPO optimization step.

        Args:
            trajectories_states: State tensors [B, G, T, SEQ]
            trajectories_actions: Action indices [B, G, T]
            trajectories_old_log_probs: Log probs from old policy [B, G, T]
            trajectories_legal_masks: Legal move masks [B, G, T, A] or None
            step_rewards: Per-step rewards [B, G, T]
            effective_pad_mask: Mask for valid steps [B, G, T]

        Returns:
            Tuple of (loss, loss_info)
        """
        # Compute new log probs with dropout disabled, while keeping gradients enabled.
        new_log_probs = self._compute_new_log_probs(
            trajectories_states=trajectories_states,
            trajectories_actions=trajectories_actions,
            trajectories_legal_masks=trajectories_legal_masks,
        )

        loss, loss_info = grpo_ppo_loss(
            new_log_probs,
            trajectories_old_log_probs,
            step_rewards,
            effective_pad_mask,
            clip_ratio=self.hparams.grpo_config.clip_ratio,
            kl_coef=self.hparams.grpo_config.kl_coef,
            return_info=True,
        )

        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss encountered: {loss.item()}")

        return loss, loss_info

    def _compute_new_log_probs(
        self,
        trajectories_states: torch.Tensor,
        trajectories_actions: torch.Tensor,
        trajectories_legal_masks: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute trainable log-probs for the current policy with dropout off."""
        with temporary_eval(self.policy_model):
            return self.policy_model.get_group_log_probs(
                trajectories_states,
                trajectories_actions,
                trajectories_legal_masks,
                temperature=self.hparams.grpo_config.rollout_temperature,
            )

    def _compute_old_log_probs(
        self,
        trajectories_states: torch.Tensor,
        trajectories_actions: torch.Tensor,
        trajectories_legal_masks: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute reference log-probs for the old policy with dropout off and no-grad."""
        with temporary_eval(self.old_policy_model), torch.no_grad():
            return self.old_policy_model.get_group_log_probs(
                trajectories_states,
                trajectories_actions,
                trajectories_legal_masks,
                temperature=self.hparams.grpo_config.rollout_temperature,
            )

    def _run_safety_checks(self, loss_info) -> None:
        """Run safety checks on training dynamics and abort if clip fraction stays too high."""
        cfg = self.hparams.grpo_config
        if not cfg.enable_safety_checks:
            return

        if loss_info.mean_clip_fraction.item() > cfg.max_clip_fraction:
            self._high_clip_steps += 1
        else:
            self._high_clip_steps = 0

        self.log("train/safety_high_clip_steps", float(self._high_clip_steps))

        if self._high_clip_steps >= cfg.safety_patience_steps:
            raise RuntimeError(
                f"Safety check triggered: clip fraction "
                f"{loss_info.mean_clip_fraction.item():.3f} > {cfg.max_clip_fraction} "
                f"for {self._high_clip_steps} consecutive steps."
            )

    def training_step(self, batch_fens: list[str], batch_idx: int) -> None:
        """Perform a training step with multiple PPO optimization iterations.

        Samples trajectories once, then performs ppo_steps optimization iterations
        on the same sampled data to improve compute efficiency.

        Args:
            batch_fens: List of FEN strings representing starting positions
            batch_idx: Batch index (unused)
        """
        opt = self.optimizers()

        boards = [chess.Board(start_fen) for start_fen in batch_fens]
        boards = [board for board in boards if not board.is_game_over()]
        if not boards:
            return  # Skip if game over

        with temporary_eval(self.old_policy_model), torch.no_grad():
            trajectories_sample = sample_trajectories_batched(
                self.old_policy_model,
                boards,
                self.hparams.grpo_config.num_trajectories,
                self.hparams.grpo_config.trajectory_depth,
                temperature=self.hparams.grpo_config.rollout_temperature,
                teacher_forcing_prob=self.hparams.grpo_config.teacher_forcing_prob,
                teacher_forcing_depth=self.hparams.grpo_config.teacher_forcing_depth,
            )
        if trajectories_sample is None:
            return  # Skip if no moves

        # Extract trajectory components (sampled once, reused for ppo_steps)
        trajectories_old_log_probs = trajectories_sample.trajectories_log_probs  # [B, G, T]
        trajectories_actions = trajectories_sample.trajectories_actions  # [B, G, T]
        trajectories_states = trajectories_sample.trajectories_states  # [B, G, T, SEQ]
        batch_group_rewards = trajectories_sample.group_rewards  # [B, G] (for logging)
        step_rewards = trajectories_sample.step_rewards  # [B, G, T]
        pad_mask = trajectories_sample.pad_mask  # [B, G, T]
        trajectories_legal_masks = trajectories_sample.trajectories_legal_masks  # [B, G, T, A] or None
        teacher_forced_mask = trajectories_sample.teacher_forced_mask  # [B, G, T], True=teacher-forced

        # Add starting player mask (only consider moves from the starting player's perspective)
        _, _, T = pad_mask.shape
        t = torch.arange(T, device=pad_mask.device)
        start_player_mask = (t % 2 == 0)[None, None, :]  # [1, 1, T]
        # Exclude teacher-forced actions from policy-gradient updates.
        effective_pad_mask = pad_mask & start_player_mask & (~teacher_forced_mask)  # [B, G, T]

        ppo_steps = self.hparams.grpo_config.ppo_steps

        # Perform multiple PPO optimization steps on the same sampled trajectories
        for ppo_step_idx in range(ppo_steps):
            loss, loss_info = self._ppo_step(
                trajectories_states,
                trajectories_actions,
                trajectories_old_log_probs,
                trajectories_legal_masks,
                step_rewards,
                effective_pad_mask,
            )

            # Manual optimization step
            opt.zero_grad()
            self.manual_backward(loss)
            self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            opt.step()

        # Standard logging (log final ppo_step metrics)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train_total_loss", loss)
        self.log("train/ppo_loss", loss_info.ppo_loss)
        self.log("train/kl_divergence", loss_info.kl_div)
        self.log("train/ratio", loss_info.mean_ratio)
        self.log("train/clip_fraction", loss_info.mean_clip_fraction)
        self.log("train/entropy", loss_info.entropy)
        self.log("train/advantage_mean", loss_info.advantage_mean)
        self.log("train/advantage_std", loss_info.advantage_std)
        self.log("train/pad_fraction", 1.0 - pad_mask.float().mean())
        self.log("train/trajectory_length", pad_mask.float().sum(dim=-1).mean())
        # Fraction of rollout steps that were teacher-forced.
        valid_rollout_steps = pad_mask.sum()
        if valid_rollout_steps.item() > 0:
            teacher_forced_fraction = teacher_forced_mask[pad_mask].float().mean()
        else:
            teacher_forced_fraction = torch.tensor(0.0, device=pad_mask.device)
        self.log("train/teacher_forced_fraction", teacher_forced_fraction)
        self._log_rewards_metrics(batch_group_rewards, prefix="train/")

        # Log step rewards statistics (only for valid steps)
        valid_step_rewards = step_rewards[pad_mask]
        self.log("train/step_reward_mean", valid_step_rewards.mean())
        self.log("train/step_reward_std", valid_step_rewards.std())

        # Log raw centipawn step rewards (before normalization) for debugging
        raw_step_cp = trajectories_sample.raw_step_cp
        valid_raw_step_cp = raw_step_cp[pad_mask]
        self.log("train/raw_step_cp_mean", valid_raw_step_cp.mean())
        self.log("train/raw_step_cp_std", valid_raw_step_cp.std())
        self.log("train/raw_step_cp_abs_mean", valid_raw_step_cp.abs().mean())

        # Run safety checks on the final loss statistics
        self._run_safety_checks(loss_info)

    def configure_optimizers(self) -> torch.optim.Adam:
        """Configure optimizer for training.
        
        Returns:
            Adam optimizer with learning rate from GRPO config
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.grpo_config.lr)
