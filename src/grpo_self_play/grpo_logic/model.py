from typing import Optional
import torch
import pytorch_lightning as pl
import chess

from dataclasses import dataclass

from src.grpo_self_play.evaluator import Evaluator
from src.grpo_self_play.models import ChessTransformer, ChessTransformerConfig
from src.grpo_self_play.grpo_logic.loss import grpo_ppo_loss
from src.grpo_self_play.grpo_logic.sampling import sample_trajectories_batched
from src.grpo_self_play.eval_utils import EvalConfig
from src.grpo_self_play.chess.policy_player import PolicyConfig
from src.grpo_self_play.chess.searcher import SearchConfig
from src.grpo_self_play.chess.stockfish import StockfishConfig
from src.grpo_self_play.chess.boards_dataset import get_game_phase
from src.grpo_self_play.pretrain.pretrain_load_config import PretrainLoadConfig


class UnifiedEntropyRecovery:
    """Unified entropy recovery mechanism that coordinates multiple interventions.

    When entropy falls below floor, this class coordinates:
    1. Boosting entropy_coef (up to max_entropy_coef)
    2. Reducing kl_coef (down to min_kl_coef) to escape KL trap
    3. Increasing temperature (up to max_temperature) for diverse sampling

    Hard stops training when entropy falls below critical threshold.
    """

    def __init__(
        self,
        # Entropy thresholds
        entropy_floor: float = 1.5,
        entropy_critical: float = 0.5,
        steps_threshold: int = 100,
        critical_steps_threshold: int = 50,
        # Entropy coefficient
        entropy_boost_factor: float = 1.5,
        max_entropy_coef: float = 1.5,
        initial_entropy_coef: float = 0.1,
        # KL coefficient
        kl_reduction_factor: float = 0.5,
        min_kl_coef: float = 0.0,
        initial_kl_coef: float = 0.01,
        # Temperature
        temperature_boost: float = 0.2,
        max_temperature: float = 2.5,
        initial_temperature: float = 1.0,
    ):
        """
        Args:
            entropy_floor: Minimum entropy threshold before recovery kicks in
            entropy_critical: Hard stop threshold - training aborts below this
            steps_threshold: Consecutive steps below floor before taking action
            critical_steps_threshold: Consecutive steps below critical before stopping
            entropy_boost_factor: Multiply entropy_coef by this each intervention
            max_entropy_coef: Maximum allowed entropy_coef
            initial_entropy_coef: Starting entropy coefficient
            kl_reduction_factor: Multiply kl_coef by this each intervention
            min_kl_coef: Minimum allowed kl_coef (0 = full KL escape)
            initial_kl_coef: Starting KL coefficient
            temperature_boost: Add this to temperature each intervention
            max_temperature: Maximum allowed temperature
            initial_temperature: Starting temperature
        """
        self.entropy_floor = entropy_floor
        self.entropy_critical = entropy_critical
        self.steps_threshold = steps_threshold
        self.critical_steps_threshold = critical_steps_threshold

        self.entropy_boost_factor = entropy_boost_factor
        self.max_entropy_coef = max_entropy_coef

        self.kl_reduction_factor = kl_reduction_factor
        self.min_kl_coef = min_kl_coef

        self.temperature_boost = temperature_boost
        self.max_temperature = max_temperature

        # Current values (mutable)
        self.current_entropy_coef = initial_entropy_coef
        self.current_kl_coef = initial_kl_coef
        self.current_temperature = initial_temperature

        # State tracking
        self.consecutive_low_steps = 0
        self.consecutive_critical_steps = 0
        self.recovery_count = 0

    def check(self, entropy: float) -> dict:
        """Check entropy and update coefficients if needed.

        Args:
            entropy: Current entropy value

        Returns:
            Metrics dict for logging

        Raises:
            RuntimeError: If entropy stays below critical threshold for too long
        """
        metrics = {}

        # Critical entropy check - stop after patience exceeded
        if entropy < self.entropy_critical:
            self.consecutive_critical_steps += 1
            if self.consecutive_critical_steps >= self.critical_steps_threshold:
                raise RuntimeError(
                    f"CRITICAL ENTROPY COLLAPSE: Entropy={entropy:.4f} < critical={self.entropy_critical} "
                    f"for {self.consecutive_critical_steps} consecutive steps. "
                    f"Training stopped to preserve checkpoint. "
                    f"Recovery attempts: {self.recovery_count}, "
                    f"entropy_coef={self.current_entropy_coef:.4f}, "
                    f"kl_coef={self.current_kl_coef:.6f}, "
                    f"temperature={self.current_temperature:.2f}"
                )
            # Warn but don't stop yet
            if self.consecutive_critical_steps == 1:
                print(
                    f"WARNING: Entropy={entropy:.4f} dropped below critical={self.entropy_critical}. "
                    f"Will stop after {self.critical_steps_threshold} consecutive steps."
                )
        else:
            self.consecutive_critical_steps = 0

        # Floor check - recovery intervention
        if entropy < self.entropy_floor:
            self.consecutive_low_steps += 1

            if self.consecutive_low_steps >= self.steps_threshold:
                self._apply_recovery(entropy)
                self.consecutive_low_steps = 0
        else:
            self.consecutive_low_steps = 0

        # Build metrics
        metrics["recovery/entropy_floor"] = self.entropy_floor
        metrics["recovery/entropy_critical"] = self.entropy_critical
        metrics["recovery/consecutive_low_steps"] = self.consecutive_low_steps
        metrics["recovery/consecutive_critical_steps"] = self.consecutive_critical_steps
        metrics["recovery/below_floor"] = float(entropy < self.entropy_floor)
        metrics["recovery/below_critical"] = float(entropy < self.entropy_critical)
        metrics["recovery/recovery_count"] = self.recovery_count
        metrics["recovery/current_entropy_coef"] = self.current_entropy_coef
        metrics["recovery/current_kl_coef"] = self.current_kl_coef
        metrics["recovery/current_temperature"] = self.current_temperature

        return metrics

    def _apply_recovery(self, entropy: float) -> None:
        """Apply coordinated recovery: boost entropy, reduce KL, increase temperature."""
        old_entropy_coef = self.current_entropy_coef
        old_kl_coef = self.current_kl_coef
        old_temperature = self.current_temperature

        # 1. Boost entropy_coef (up to max)
        self.current_entropy_coef = min(
            self.current_entropy_coef * self.entropy_boost_factor,
            self.max_entropy_coef
        )

        # 2. Reduce kl_coef (down to min) - escape KL trap
        self.current_kl_coef = max(
            self.current_kl_coef * self.kl_reduction_factor,
            self.min_kl_coef
        )

        # 3. Increase temperature (up to max)
        self.current_temperature = min(
            self.current_temperature + self.temperature_boost,
            self.max_temperature
        )

        self.recovery_count += 1

        print(
            f"RECOVERY #{self.recovery_count}: entropy={entropy:.4f} < floor={self.entropy_floor}\n"
            f"  entropy_coef: {old_entropy_coef:.4f} -> {self.current_entropy_coef:.4f}\n"
            f"  kl_coef: {old_kl_coef:.6f} -> {self.current_kl_coef:.6f}\n"
            f"  temperature: {old_temperature:.2f} -> {self.current_temperature:.2f}"
        )


def compute_group_collapse_metrics(
    actions: torch.Tensor,
    group_rewards: torch.Tensor,
    step_rewards: torch.Tensor,
    pad_mask: torch.Tensor,
) -> dict:
    """Compute within-board group collapse metrics (Recommendation 4).

    These metrics directly measure whether all G trajectories from the same board
    are converging to the same moves, which is the key failure mode in entropy collapse.

    Args:
        actions: Action indices [B, G, T]
        group_rewards: Final rewards for each trajectory [B, G]
        step_rewards: Per-step rewards [B, G, T]
        pad_mask: Mask indicating valid steps [B, G, T], True=valid

    Returns:
        Dictionary of metrics for logging
    """
    B, _, T = actions.shape
    metrics = {}

    # 1. Action agreement: for each (b, t), what fraction of trajectories chose the most common action?
    # agreement[b,t] = max_count(actions[b,:,t]) / G
    action_agreement = torch.zeros(B, T, device=actions.device)
    for b in range(B):
        for t in range(T):
            if pad_mask[b, :, t].any():  # At least one valid trajectory at this timestep
                valid_actions = actions[b, pad_mask[b, :, t], t]
                if len(valid_actions) > 0:
                    # Count occurrences of each action
                    _, counts = valid_actions.unique(return_counts=True)
                    max_count = counts.max().item()
                    num_valid = pad_mask[b, :, t].sum().item()
                    action_agreement[b, t] = max_count / num_valid

    # Mask to only consider valid (b, t) pairs
    valid_bt_mask = pad_mask.any(dim=1)  # [B, T] - True if any trajectory valid at (b, t)
    valid_agreements = action_agreement[valid_bt_mask]

    if len(valid_agreements) > 0:
        metrics["group_collapse/action_agreement_mean"] = valid_agreements.mean().item()
        metrics["group_collapse/action_agreement_p90"] = valid_agreements.quantile(0.9).item()
        metrics["group_collapse/action_agreement_max"] = valid_agreements.max().item()
    else:
        metrics["group_collapse/action_agreement_mean"] = 0.0
        metrics["group_collapse/action_agreement_p90"] = 0.0
        metrics["group_collapse/action_agreement_max"] = 0.0

    # 2. Within-board reward diversity: std(group_rewards[b,:]) for each board b
    # This measures whether trajectories from the same starting position get similar rewards
    reward_std_within = group_rewards.std(dim=1)  # [B]
    metrics["group_collapse/reward_std_within_mean"] = reward_std_within.mean().item()
    metrics["group_collapse/reward_std_within_min"] = reward_std_within.min().item()

    # 3. Within-board step reward diversity: std(step_rewards[b,:,t]) for each (b, t)
    # Only compute for valid (b, t) pairs
    step_reward_std_within = torch.zeros(B, T, device=step_rewards.device)
    for b in range(B):
        for t in range(T):
            valid_mask_bt = pad_mask[b, :, t]
            if valid_mask_bt.sum() > 1:  # Need at least 2 valid trajectories for std
                step_reward_std_within[b, t] = step_rewards[b, valid_mask_bt, t].std().item()

    valid_step_stds = step_reward_std_within[valid_bt_mask]
    if len(valid_step_stds) > 0:
        metrics["group_collapse/step_reward_std_within_mean"] = valid_step_stds.mean().item()
        metrics["group_collapse/step_reward_std_within_min"] = valid_step_stds.min().item()
    else:
        metrics["group_collapse/step_reward_std_within_mean"] = 0.0
        metrics["group_collapse/step_reward_std_within_min"] = 0.0

    return metrics


class AdaptiveKLController:
    """Adapts KL coefficient to maintain target KL divergence (Recommendation 2).

    Implements a simple multiplicative controller that increases kl_coef when
    KL divergence exceeds target and decreases it when below target.
    """

    def __init__(self, initial_kl_coef: float, target_kl: float, adapt_rate: float,
                 kl_coef_min: float, kl_coef_max: float):
        """
        Args:
            initial_kl_coef: Starting KL coefficient
            target_kl: Target KL divergence value
            adapt_rate: Multiplicative factor for adjustment
            kl_coef_min: Minimum allowed kl_coef
            kl_coef_max: Maximum allowed kl_coef
        """
        self.current_kl_coef = initial_kl_coef
        self.target_kl = target_kl
        self.adapt_rate = adapt_rate
        self.kl_coef_min = kl_coef_min
        self.kl_coef_max = kl_coef_max

    def update(self, kl_div: float) -> dict:
        """Update KL coefficient based on current KL divergence.

        Args:
            kl_div: Current KL divergence value

        Returns:
            Metrics dict for logging
        """
        if kl_div > self.target_kl:
            self.current_kl_coef = min(self.current_kl_coef * self.adapt_rate, self.kl_coef_max)
        else:
            self.current_kl_coef = max(self.current_kl_coef / self.adapt_rate, self.kl_coef_min)

        return {
            "adaptive_kl/current_kl_coef": self.current_kl_coef,
            "adaptive_kl/target_kl": self.target_kl,
            "adaptive_kl/kl_ratio": kl_div / self.target_kl if self.target_kl > 0 else 0.0,
        }


@dataclass
class GRPOConfig:
    """Configuration for GRPO (Group Relative Policy Optimization) training.

    Attributes:
        lr: Learning rate for optimizer
        num_trajectories: Number of trajectory groups to sample per batch
        trajectory_depth: Maximum depth of each trajectory
        clip_ratio: PPO clipping ratio (epsilon)
        kl_coef: Initial KL divergence penalty coefficient
        entropy_coef: Initial entropy bonus coefficient
        eval_every_n_epochs: Frequency of evaluation runs

        # Unified Entropy Recovery - coordinates entropy_coef, kl_coef, and temperature
        use_entropy_recovery: Whether to enable unified entropy recovery
        entropy_floor: Entropy threshold below which recovery kicks in
        entropy_critical: Hard stop threshold - training aborts below this
        entropy_floor_steps: Consecutive steps below floor before taking action
        critical_steps_threshold: Consecutive steps below critical before stopping
        entropy_boost_factor: Multiply entropy_coef by this each recovery
        max_entropy_coef: Maximum allowed entropy_coef (prevents numerical instability)
        kl_reduction_factor: Multiply kl_coef by this each recovery (escape KL trap)
        min_kl_coef: Minimum kl_coef during recovery (0 = full KL escape)
        temperature_boost: Add to temperature each recovery
        max_temperature: Maximum allowed temperature

        # PPO-style multiple updates
        ppo_steps: Number of optimization steps per sampled trajectory batch

        # Rollout temperature for exploration
        rollout_temperature: Initial temperature for action sampling during rollouts

        # Safety checks on training dynamics
        enable_safety_checks: Whether to abort training when known-bad patterns persist
        safety_patience_steps: Number of training steps to tolerate violations before aborting
        max_clip_fraction: If mean_clip_fraction > this for too long -> abort
        max_kl_divergence: If KL > this for too long -> abort
    """
    # Learning rate: tuned from prior runs; 3e-5 was the most stable setting
    lr: float = 3e-5
    num_trajectories: int = 4
    trajectory_depth: int = 5
    clip_ratio: float = 0.2
    kl_coef: float = 0.01
    entropy_coef: float = 0.1
    eval_every_n_epochs: int = 10

    # Unified Entropy Recovery (replaces separate entropy floor and adaptive KL)
    # Coordinates: entropy_coef boosting, kl_coef reduction, temperature increase
    use_entropy_recovery: bool = True
    entropy_floor: float = 1.5           # Below this, trigger recovery
    entropy_critical: float = 0.5        # Below this, STOP training
    entropy_floor_steps: int = 100       # Consecutive steps before action
    critical_steps_threshold: int = 50   # Consecutive steps below critical before stop
    entropy_boost_factor: float = 1.5    # Multiply entropy_coef by this
    max_entropy_coef: float = 1.5        # Cap to prevent numerical instability
    kl_reduction_factor: float = 0.5     # Multiply kl_coef by this (escape KL trap)
    min_kl_coef: float = 0.0             # Floor for kl_coef (0 = full escape)
    temperature_boost: float = 0.2       # Add to temperature each recovery
    max_temperature: float = 2.5         # Cap on temperature

    # PPO-style multiple updates per sample
    ppo_steps: int = 1

    # Rollout temperature for exploration (>1 flattens distribution, increases entropy)
    # This is the INITIAL temperature; recovery may increase it
    rollout_temperature: float = 1.5     # Start higher to prevent early collapse

    # Safety checks on training dynamics (backup to recovery mechanism)
    enable_safety_checks: bool = True
    safety_patience_steps: int = 1000
    max_clip_fraction: float = 0.95
    max_kl_divergence: float = 0.1


class GRPOChessTransformer(pl.LightningModule):
    """PyTorch Lightning module for training chess policy with GRPO.

    This module implements Group Relative Policy Optimization (GRPO) for training
    a chess transformer policy. It maintains both a current policy and an old policy
    for computing importance sampling ratios in the PPO loss.

    Attributes:
        policy_model: Current policy model being trained
        old_policy_model: Frozen copy of policy for importance sampling
        evaluator: Evaluator for running games against Stockfish
        eval_every_n_epochs: Frequency of evaluation runs
        recovery: Optional unified entropy recovery mechanism
        automatic_optimization: Set to False for manual PPO steps
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
        """
        Initialize GRPO Chess Transformer.

        Args:
            transformer_config: Configuration for the chess transformer model
            grpo_config: GRPO training configuration
            eval_cfg: Optional evaluation configuration
            stockfish_cfg: Optional Stockfish configuration for evaluation
            policy_cfg: Optional policy player configuration
            searcher_cfg: Optional search configuration
            pretrain_cfg: Optional pretrain config for loading pretrained weights
        """
        super().__init__()
        self.save_hyperparameters()
        self.policy_model = ChessTransformer(transformer_config)
        self.old_policy_model = ChessTransformer(transformer_config)

        # Load pretrained weights if specified
        if pretrain_cfg and pretrain_cfg.checkpoint_path:
            self._load_pretrained_weights(pretrain_cfg)

        self._sync_old_policy()

        # Evaluation config
        self.eval_every_n_epochs = grpo_config.eval_every_n_epochs
        self.evaluator = Evaluator(eval_cfg=eval_cfg or EvalConfig(),
                                   policy_cfg=policy_cfg or PolicyConfig(),
                                   stockfish_cfg=stockfish_cfg or StockfishConfig(),
                                   searcher_cfg=searcher_cfg)

        # Unified Entropy Recovery - coordinates entropy_coef, kl_coef, and temperature
        self.recovery: UnifiedEntropyRecovery | None = None
        if grpo_config.use_entropy_recovery:
            self.recovery = UnifiedEntropyRecovery(
                entropy_floor=grpo_config.entropy_floor,
                entropy_critical=grpo_config.entropy_critical,
                steps_threshold=grpo_config.entropy_floor_steps,
                critical_steps_threshold=grpo_config.critical_steps_threshold,
                entropy_boost_factor=grpo_config.entropy_boost_factor,
                max_entropy_coef=grpo_config.max_entropy_coef,
                initial_entropy_coef=grpo_config.entropy_coef,
                kl_reduction_factor=grpo_config.kl_reduction_factor,
                min_kl_coef=grpo_config.min_kl_coef,
                initial_kl_coef=grpo_config.kl_coef,
                temperature_boost=grpo_config.temperature_boost,
                max_temperature=grpo_config.max_temperature,
                initial_temperature=grpo_config.rollout_temperature,
            )

        # Safety-check state (for tracking persistent violations)
        self._safety_step_idx: int = 0
        self._high_clip_steps: int = 0
        self._low_entropy_steps: int = 0
        self._high_kl_steps: int = 0

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

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

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
        missing, unexpected = self.policy_model.load_state_dict(state_dict, strict=False)
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

        self.log(prefix + "avg_reward", mean_r, prog_bar=True)
        self.log(prefix + "reward_std", batch_group_rewards.std())
        self.log(prefix + "reward_p50", batch_group_rewards.median())
        self.log(prefix + "reward_p90", batch_group_rewards.quantile(0.9))
        self.log(prefix + "reward_best", best)
        self.log(prefix + "reward_gap_best_minus_mean", gap)

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
    ) -> tuple[torch.Tensor, object, torch.Tensor]:
        """Perform a single PPO optimization step.

        Args:
            trajectories_states: State tensors [B, G, T, SEQ]
            trajectories_actions: Action indices [B, G, T]
            trajectories_old_log_probs: Log probs from old policy [B, G, T]
            trajectories_legal_masks: Legal move masks [B, G, T, A] or None
            step_rewards: Per-step rewards [B, G, T]
            effective_pad_mask: Mask for valid steps [B, G, T]

        Returns:
            Tuple of (loss, loss_info, new_log_probs)
        """
        # Compute new log probs with current policy
        new_log_probs = self.policy_model.get_group_log_probs(
            trajectories_states, trajectories_actions, trajectories_legal_masks
        )

        # Use current (possibly adapted) coefficients from recovery mechanism
        if self.recovery:
            kl_coef = self.recovery.current_kl_coef
            entropy_coef = self.recovery.current_entropy_coef
        else:
            kl_coef = self.hparams.grpo_config.kl_coef
            entropy_coef = self.hparams.grpo_config.entropy_coef

        loss, loss_info = grpo_ppo_loss(
            new_log_probs,
            trajectories_old_log_probs,
            step_rewards,
            effective_pad_mask,
            clip_ratio=self.hparams.grpo_config.clip_ratio,
            kl_coef=kl_coef,
            entropy_coef=entropy_coef,
            return_info=True,
        )

        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss encountered: {loss.item()}")

        return loss, loss_info, new_log_probs

    def _run_safety_checks(self, loss_info) -> None:
        """Run safety checks on training dynamics and abort if they persistently fail."""
        cfg = self.hparams.grpo_config
        if not cfg.enable_safety_checks:
            return

        self._safety_step_idx += 1

        # 1) PPO clipping saturation
        if loss_info.mean_clip_fraction.item() > cfg.max_clip_fraction:
            self._high_clip_steps += 1
        else:
            self._high_clip_steps = 0

        # 2) Entropy collapse
        if loss_info.entropy.item() < cfg.min_entropy:
            self._low_entropy_steps += 1
        else:
            self._low_entropy_steps = 0

        # 3) Excessive KL divergence
        if loss_info.kl_div.item() > cfg.max_kl_divergence:
            self._high_kl_steps += 1
        else:
            self._high_kl_steps = 0

        # Log safety counters for debugging
        self.log("safety/high_clip_steps", float(self._high_clip_steps))
        self.log("safety/low_entropy_steps", float(self._low_entropy_steps))
        self.log("safety/high_kl_steps", float(self._high_kl_steps))

        if (
            self._high_clip_steps >= cfg.safety_patience_steps
            or self._low_entropy_steps >= cfg.safety_patience_steps
            or self._high_kl_steps >= cfg.safety_patience_steps
        ):
            raise RuntimeError(
                "Safety checks triggered: training aborted due to persistent "
                f"bad dynamics (clip={loss_info.mean_clip_fraction.item():.3f}, "
                f"entropy={loss_info.entropy.item():.3f}, "
                f"kl={loss_info.kl_div.item():.4f}). "
                "Adjust GRPOConfig or investigate recent research docs."
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

        # Determine game phases for phase-specific logging
        board_phases = [get_game_phase(board) for board in boards]  # List of phases, one per board

        # Use recovery's temperature if available, otherwise use config default
        if self.recovery:
            temperature = self.recovery.current_temperature
        else:
            temperature = self.hparams.grpo_config.rollout_temperature

        trajectories_sample = sample_trajectories_batched(
            self.old_policy_model,
            boards,
            self.hparams.grpo_config.num_trajectories,
            self.hparams.grpo_config.trajectory_depth,
            temperature=temperature
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

        # Add starting player mask (only consider moves from the starting player's perspective)
        _, _, T = pad_mask.shape
        t = torch.arange(T, device=pad_mask.device)
        start_player_mask = (t % 2 == 0)[None, None, :]  # [1, 1, T]
        effective_pad_mask = pad_mask & start_player_mask  # [B, G, T]

        ppo_steps = self.hparams.grpo_config.ppo_steps

        # Perform multiple PPO optimization steps on the same sampled trajectories
        new_log_probs = None  # Will be set by last ppo_step for phase-specific logging
        for ppo_step_idx in range(ppo_steps):
            loss, loss_info, new_log_probs = self._ppo_step(
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

            # Unified entropy recovery - only on last ppo_step
            # Checks entropy and coordinates: entropy_coef boost, kl_coef reduction, temperature increase
            if ppo_step_idx == ppo_steps - 1 and self.recovery is not None:
                recovery_metrics = self.recovery.check(loss_info.entropy.item())
                for key, value in recovery_metrics.items():
                    self.log(key, value)

        # Within-board group collapse metrics (Recommendation 4) - log once per training_step
        collapse_metrics = compute_group_collapse_metrics(
            trajectories_actions, batch_group_rewards, step_rewards, pad_mask
        )
        for key, value in collapse_metrics.items():
            self.log(key, value)

        # Standard logging (log final ppo_step metrics)
        valid_mask = pad_mask.float()  # [B, G, T] 1 = real step

        self.log("train_total_loss", loss, prog_bar=True)
        self.log("pad_fraction", 1.0 - valid_mask.mean())
        self.log("avg_trajectory_length", pad_mask.float().sum(dim=-1).mean())

        self.log("mean_kl_divergence", loss_info.kl_div)
        self.log("mean_ratio", loss_info.mean_ratio)
        self.log("mean_clip_fraction", loss_info.mean_clip_fraction)
        self.log("ppo_loss", loss_info.ppo_loss)
        self.log("entropy", loss_info.entropy)
        # Loss without the entropy bonus term (PPO + KL only)
        self.log("train/loss_without_entropy", loss_info.loss_without_entropy)
        self.log("ppo_steps", float(ppo_steps))
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

        # Phase-specific logging (entropy, rewards per game phase)
        # new_log_probs has shape [B, G, T], board_phases has length B
        phase_metrics = self._compute_phase_metrics(
            new_log_probs, step_rewards, batch_group_rewards, pad_mask, board_phases
        )
        for key, value in phase_metrics.items():
            self.log(key, value)

        # Run safety checks on the final loss statistics
        self._run_safety_checks(loss_info)

    def _compute_phase_metrics(
        self,
        log_probs: torch.Tensor,  # [B, G, T]
        step_rewards: torch.Tensor,  # [B, G, T]
        group_rewards: torch.Tensor,  # [B, G]
        pad_mask: torch.Tensor,  # [B, G, T]
        board_phases: list[str],  # length B
    ) -> dict[str, float]:
        """Compute phase-specific metrics for logging.

        Computes entropy, rewards, and counts broken down by game phase
        (opening, middlegame, endgame).

        Args:
            log_probs: Log probabilities from the policy [B, G, T]
            step_rewards: Per-step rewards [B, G, T]
            group_rewards: Per-trajectory total rewards [B, G]
            pad_mask: Mask for valid steps [B, G, T]
            board_phases: Game phase for each starting position in batch

        Returns:
            Dictionary of metrics with keys like "train/phase_opening/entropy"
        """
        metrics = {}
        phases = ["opening", "middlegame", "endgame"]

        B, G, T = log_probs.shape

        for phase in phases:
            # Find batch indices for this phase
            phase_indices = [i for i, p in enumerate(board_phases) if p == phase]

            if not phase_indices:
                # No samples for this phase
                metrics[f"train/phase_{phase}/count"] = 0.0
                continue

            # Log count/fraction
            metrics[f"train/phase_{phase}/count"] = float(len(phase_indices))
            metrics[f"train/phase_{phase}/fraction"] = len(phase_indices) / B

            # Subset tensors for this phase [num_phase, G, T]
            phase_log_probs = log_probs[phase_indices]
            phase_step_rewards = step_rewards[phase_indices]
            phase_group_rewards = group_rewards[phase_indices]  # [num_phase, G]
            phase_pad_mask = pad_mask[phase_indices]

            # Entropy: H(π) ≈ -E[log π(a|s)]
            valid_log_probs = phase_log_probs[phase_pad_mask]
            if valid_log_probs.numel() > 0:
                phase_entropy = -valid_log_probs.mean().item()
                metrics[f"train/phase_{phase}/entropy"] = phase_entropy

            # Step rewards (valid steps only)
            valid_step_rewards = phase_step_rewards[phase_pad_mask]
            if valid_step_rewards.numel() > 0:
                metrics[f"train/phase_{phase}/step_reward_mean"] = valid_step_rewards.mean().item()
                metrics[f"train/phase_{phase}/step_reward_std"] = valid_step_rewards.std().item()

            # Group rewards (trajectory-level)
            if phase_group_rewards.numel() > 0:
                metrics[f"train/phase_{phase}/group_reward_mean"] = phase_group_rewards.mean().item()
                metrics[f"train/phase_{phase}/group_reward_std"] = phase_group_rewards.std().item()

        return metrics

    def configure_optimizers(self) -> torch.optim.Adam:
        """Configure optimizer for training.
        
        Returns:
            Adam optimizer with learning rate from GRPO config
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.grpo_config.lr)

    def _evaluate_against_stockfish(self) -> Optional[tuple[dict, list[str]]]:
        """Run a single game evaluation against Stockfish with current policy model.

        Returns:
            Tuple of (results_dict, pgns) or None if evaluation failed
            pgns is a list of PGN strings for all games played
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                results, _, pgns = self.evaluator.single_evaluation(self.policy_model)
            return results, pgns
        except Exception as e:
            self.logger.warning(f"Evaluation against Stockfish failed: {e}") if hasattr(self, 'logger') else print(f"Evaluation against Stockfish failed: {e}")
            return None
        finally:
            if was_training:
                self.train()

    def _log_stockfish_eval(self, results: dict) -> None:
        """Log scalar evaluation metrics from the Stockfish evaluation.
        
        Args:
            results: Dictionary containing evaluation results with keys:
                - games: Total number of games played
                - wins: Number of wins
                - draws: Number of draws
                - losses: Number of losses
                - score: Win rate (0-1)
                - elo_diff_vs_stockfish_approx: Approximate Elo difference
                - termination_reasons: Dict mapping termination reasons to counts
        """
        # Scalar stats
        self.log("eval_stockfish/games", results["games"])
        self.log("eval_stockfish/wins", results["wins"])
        self.log("eval_stockfish/draws", results["draws"])
        self.log("eval_stockfish/losses", results["losses"])
        self.log("eval_stockfish/score", results["score"], prog_bar=True)
        self.log("eval_stockfish/elo_diff", results["elo_diff_vs_stockfish_approx"], prog_bar=True)

        # Termination reasons as fractions
        games = results["games"] or 1
        for reason, cnt in results["termination_reasons"].items():
            frac = cnt / games
            self.log(f"eval_stockfish/term_{reason}", frac)
    
    def _log_pgns(self, pgns: list[str]) -> None:
        """Log PGNs to WandB as a text artifact.

        Args:
            pgns: List of PGN strings for all games played
        """
        if not pgns:
            return

        # Combine all PGNs into a single string
        combined_pgn = "\n\n".join(pgns)

        # Log to WandB if available
        if self.logger and hasattr(self.logger, 'experiment'):
            try:
                import wandb
                # Log as a text artifact
                self.logger.experiment.log({
                    "eval_stockfish/pgns": wandb.Html(f"<pre>{combined_pgn}</pre>"),
                    "eval_stockfish/pgn_text": combined_pgn,
                })
            except Exception as e:
                print(f"Failed to log PGNs to WandB: {e}")

    def on_train_epoch_end(self) -> None:
        """Called at the end of each training epoch. Runs evaluation if scheduled."""
        if (self.current_epoch + 1) % self.eval_every_n_epochs == 0:
            eval_result = self._evaluate_against_stockfish()
            if eval_result is not None:
                results, pgns = eval_result
                self._log_stockfish_eval(results)
                self._log_pgns(pgns)