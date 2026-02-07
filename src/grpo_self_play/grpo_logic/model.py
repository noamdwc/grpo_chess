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
from src.grpo_self_play.pretrain.pretrain_load_config import PretrainLoadConfig


class EntropyFloorMonitor:
    """Monitors entropy and takes action when it falls below a floor (Recommendation 1).

    Tracks consecutive steps where entropy is below a threshold and triggers
    configurable actions (warn, stop, or boost entropy_coef) when the threshold
    is breached for too long.
    """

    def __init__(self, floor: float, steps_threshold: int, action: str, boost_factor: float):
        """
        Args:
            floor: Minimum entropy threshold
            steps_threshold: Consecutive steps below floor before action
            action: Action to take ("warn", "stop", "boost")
            boost_factor: Factor to multiply entropy_coef when boosting
        """
        self.floor = floor
        self.steps_threshold = steps_threshold
        self.action = action
        self.boost_factor = boost_factor
        self.consecutive_low_steps = 0
        self.triggered = False

    def check(self, entropy: float, current_entropy_coef: float) -> tuple[float, dict]:
        """Check entropy and return updated entropy_coef and metrics.

        Args:
            entropy: Current entropy value
            current_entropy_coef: Current entropy coefficient

        Returns:
            Tuple of (new_entropy_coef, metrics_dict)
        """
        metrics = {}
        new_entropy_coef = current_entropy_coef

        if entropy < self.floor:
            self.consecutive_low_steps += 1

            if self.consecutive_low_steps >= self.steps_threshold and not self.triggered:
                self.triggered = True
                if self.action == "warn":
                    print(f"WARNING: Entropy collapse detected! Entropy={entropy:.4f} < floor={self.floor} "
                          f"for {self.consecutive_low_steps} consecutive steps.")
                elif self.action == "stop":
                    raise RuntimeError(
                        f"STOPPING: Entropy collapse detected! Entropy={entropy:.4f} < floor={self.floor} "
                        f"for {self.consecutive_low_steps} consecutive steps.")
                elif self.action == "boost":
                    new_entropy_coef = current_entropy_coef * self.boost_factor
                    print(f"BOOSTING entropy_coef: {current_entropy_coef:.4f} -> {new_entropy_coef:.4f} "
                          f"(entropy={entropy:.4f} < floor={self.floor})")
                    self.consecutive_low_steps = 0
                    self.triggered = False
        else:
            self.consecutive_low_steps = 0
            self.triggered = False

        metrics["entropy_floor/consecutive_low_steps"] = self.consecutive_low_steps
        metrics["entropy_floor/below_floor"] = float(entropy < self.floor)
        metrics["entropy_floor/current_entropy_coef"] = new_entropy_coef

        return new_entropy_coef, metrics


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
        kl_coef: KL divergence penalty coefficient (beta)
        entropy_coef: Entropy bonus coefficient (encourages exploration, prevents policy collapse)
        eval_every_n_epochs: Frequency of evaluation runs (not used in model, but useful for trainer)

        # Entropy floor monitoring (Recommendation 1)
        use_entropy_floor: Whether to enable entropy floor monitoring
        entropy_floor: Minimum entropy threshold for collapse detection
        entropy_floor_steps: Number of consecutive steps below floor before alert/action
        entropy_floor_action: Action to take when entropy floor is breached ("warn", "stop", "boost")
        entropy_boost_factor: Factor to multiply entropy_coef when boosting (if action="boost")

        # Adaptive KL controller (Recommendation 2)
        adaptive_kl: Whether to use adaptive KL coefficient
        target_kl: Target KL divergence value
        kl_adapt_rate: Rate at which to adjust kl_coef (higher = faster adaptation)
        kl_coef_min: Minimum allowed kl_coef
        kl_coef_max: Maximum allowed kl_coef

        # PPO-style multiple updates
        ppo_steps: Number of optimization steps per sampled trajectory batch (reuses samples)

        # Rollout temperature for exploration
        rollout_temperature: Temperature for action sampling during rollouts (>1 increases exploration)

        # Safety checks on training dynamics
        enable_safety_checks: Whether to abort training when known-bad patterns persist
        safety_patience_steps: Number of training steps to tolerate violations before aborting
        max_clip_fraction: If mean_clip_fraction > this for too long -> abort
        min_entropy: If entropy < this for too long -> abort
        max_kl_divergence: If KL >> target_kl for too long -> abort
    """
    # Clean run defaults (see research_docs/2026-02-06_loss-budget-and-monitor-analysis.md)
    lr: float = 1e-6             # Reduced: PPO signal now dominates gradient
    num_trajectories: int = 4
    trajectory_depth: int = 5
    clip_ratio: float = 0.2
    kl_coef: float = 0.001       # Reduced from 0.01 (was overridden to 0.1 by adaptive KL)
    entropy_coef: float = 0.0    # Removed: not in original GRPO loss, was 95% of gradient
    eval_every_n_epochs: int = 10

    # Entropy floor monitoring — disabled by default (never triggered in practice)
    use_entropy_floor: bool = False
    entropy_floor: float = 1.5
    entropy_floor_steps: int = 200
    entropy_floor_action: str = "boost"
    entropy_boost_factor: float = 2.0

    # Adaptive KL controller — disabled by default (saturated at max instantly)
    adaptive_kl: bool = False
    target_kl: float = 0.015
    kl_adapt_rate: float = 1.2
    kl_coef_min: float = 0.003
    kl_coef_max: float = 0.05

    # PPO-style multiple updates per sample
    ppo_steps: int = 1

    # Rollout temperature for exploration (>1 flattens distribution, increases entropy)
    rollout_temperature: float = 1.0

    # Safety checks on training dynamics
    enable_safety_checks: bool = False
    safety_patience_steps: int = 1000  # Number of training steps to tolerate violations
    # Thresholds derived from prior research docs
    max_clip_fraction: float = 0.95    # If mean_clip_fraction > this for too long -> abort
    min_entropy: float = 0.5           # If entropy < this for too long -> abort
    max_kl_divergence: float = 0.08    # If KL >> target_kl for too long -> abort

    # Teacher forcing: use Stockfish for rival moves during trajectory sampling
    teacher_forcing_prob: float = 0.0  # Probability of using Stockfish for rival (opponent) moves
    teacher_forcing_depth: int = 4     # Stockfish search depth for teacher forcing moves


# Register as safe for torch.load with weights_only=True (PyTorch 2.6+ compatibility)
torch.serialization.add_safe_globals([GRPOConfig])


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
        entropy_monitor: Optional entropy floor monitor (Recommendation 1)
        kl_controller: Optional adaptive KL controller (Recommendation 2)
        current_entropy_coef: Current entropy coefficient (mutable for entropy boosting)
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

        # Entropy floor monitor (Recommendation 1) - optional
        self.entropy_monitor: EntropyFloorMonitor | None = None
        if grpo_config.use_entropy_floor:
            self.entropy_monitor = EntropyFloorMonitor(
                floor=grpo_config.entropy_floor,
                steps_threshold=grpo_config.entropy_floor_steps,
                action=grpo_config.entropy_floor_action,
                boost_factor=grpo_config.entropy_boost_factor,
            )
        self.current_entropy_coef = grpo_config.entropy_coef

        # Adaptive KL controller (Recommendation 2) - optional
        self.kl_controller: AdaptiveKLController | None = None
        if grpo_config.adaptive_kl:
            self.kl_controller = AdaptiveKLController(
                initial_kl_coef=grpo_config.kl_coef,
                target_kl=grpo_config.target_kl,
                adapt_rate=grpo_config.kl_adapt_rate,
                kl_coef_min=grpo_config.kl_coef_min,
                kl_coef_max=grpo_config.kl_coef_max,
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

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

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
        # Compute new log probs with current policy (must match rollout temperature)
        new_log_probs = self.policy_model.get_group_log_probs(
            trajectories_states, trajectories_actions, trajectories_legal_masks,
            temperature=self.hparams.grpo_config.rollout_temperature,
        )

        # Use current (possibly adapted) coefficients
        kl_coef = self.kl_controller.current_kl_coef if self.kl_controller else self.hparams.grpo_config.kl_coef

        loss, loss_info = grpo_ppo_loss(
            new_log_probs,
            trajectories_old_log_probs,
            step_rewards,
            effective_pad_mask,
            clip_ratio=self.hparams.grpo_config.clip_ratio,
            kl_coef=kl_coef,
            entropy_coef=self.current_entropy_coef,
            return_info=True,
        )

        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss encountered: {loss.item()}")

        return loss, loss_info

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

        # Add starting player mask (only consider moves from the starting player's perspective)
        _, _, T = pad_mask.shape
        t = torch.arange(T, device=pad_mask.device)
        start_player_mask = (t % 2 == 0)[None, None, :]  # [1, 1, T]
        effective_pad_mask = pad_mask & start_player_mask  # [B, G, T]

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

            # Entropy floor monitoring (Recommendation 1) - only on last ppo_step
            if ppo_step_idx == ppo_steps - 1 and self.entropy_monitor is not None:
                self.current_entropy_coef, entropy_metrics = self.entropy_monitor.check(
                    loss_info.entropy.item(), self.current_entropy_coef
                )
                for key, value in entropy_metrics.items():
                    self.log(key, value)

            # Adaptive KL controller (Recommendation 2) - only on last ppo_step
            if ppo_step_idx == ppo_steps - 1 and self.kl_controller is not None:
                kl_metrics = self.kl_controller.update(loss_info.kl_div.item())
                for key, value in kl_metrics.items():
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
        self.log("train/advantage_mean", loss_info.advantage_mean)
        self.log("train/advantage_std", loss_info.advantage_std)
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

        # Run safety checks on the final loss statistics
        self._run_safety_checks(loss_info)

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