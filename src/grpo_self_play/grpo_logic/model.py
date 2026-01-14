from math import isfinite
from typing import Optional
import torch
import pytorch_lightning as pl
import chess

from dataclasses import dataclass

from src.grpo_self_play.evaluator import Evaluator
from src.grpo_self_play.models import ChessTransformer, ChessTransformerConfig
from src.grpo_self_play.grpo_logic.loss import grpo_ppo_loss, GRPOLossInfo
from src.grpo_self_play.grpo_logic.sampling import sample_trajectories_batched
from src.grpo_self_play.eval_utils import EvalConfig
from src.grpo_self_play.chess.policy_player import PolicyConfig
from src.grpo_self_play.chess.searcher import SearchConfig
from src.grpo_self_play.chess.stockfish import StockfishConfig


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
    """
    lr: float = 1e-4
    num_trajectories: int = 4
    trajectory_depth: int = 5
    clip_ratio: float = 0.2
    kl_coef: float = 0.01
    entropy_coef: float = 0.01  # Entropy bonus to prevent policy collapse
    eval_every_n_epochs: int = 10  # Not used in model, but useful for trainer


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
    """
    def __init__(self,
                 transformer_config: ChessTransformerConfig,
                 grpo_config: GRPOConfig,
                 eval_cfg: EvalConfig | None = None,
                 stockfish_cfg: StockfishConfig | None = None,
                 policy_cfg: PolicyConfig | None = None,
                 searcher_cfg: SearchConfig | None = None):
        """
        Initialize GRPO Chess Transformer.
        
        Args:
            transformer_config: Configuration for the chess transformer model
            grpo_config: GRPO training configuration
            eval_cfg: Optional evaluation configuration
            stockfish_cfg: Optional Stockfish configuration for evaluation
            policy_cfg: Optional policy player configuration
            searcher_cfg: Optional search configuration
        """
        super().__init__()
        self.save_hyperparameters()
        self.policy_model = ChessTransformer(transformer_config)
        self.old_policy_model = ChessTransformer(transformer_config)
        self._sync_old_policy()

        # Evaluation config
        self.eval_every_n_epochs = grpo_config.eval_every_n_epochs
        self.evaluator = Evaluator(eval_cfg=eval_cfg or EvalConfig(),
                                   policy_cfg=policy_cfg or PolicyConfig(),
                                   stockfish_cfg=stockfish_cfg or StockfishConfig(),
                                   searcher_cfg=searcher_cfg)

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

    def training_step(self, batch_fens: list[str], batch_idx: int) -> torch.Tensor:
        """Perform a single training step.
        
        Args:
            batch_fens: List of FEN strings representing starting positions
            batch_idx: Batch index (unused)
            
        Returns:
            Training loss tensor
        """
        boards = [chess.Board(start_fen) for start_fen in batch_fens]
        boards = [board for board in boards if not board.is_game_over()]
        if not boards:
            return torch.tensor(0.0, device=self.device, requires_grad=True)  # Skip if game over

        trajectories_sample = sample_trajectories_batched(
            self.old_policy_model,
            boards,
            self.hparams.grpo_config.num_trajectories,
            self.hparams.grpo_config.trajectory_depth
        )
        if trajectories_sample is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)  # Skip if no moves

        # Extract trajectory components
        trajectories_old_log_probs = trajectories_sample.trajectories_log_probs  # [B, G, T]
        trajectories_actions = trajectories_sample.trajectories_actions  # [B, G, T]
        trajectories_states = trajectories_sample.trajectories_states  # [B, G, T, SEQ]
        batch_group_rewards = trajectories_sample.group_rewards  # [B, G] (for logging)
        step_rewards = trajectories_sample.step_rewards  # [B, G, T]
        pad_mask = trajectories_sample.pad_mask  # [B, G, T]
        trajectories_legal_masks = trajectories_sample.trajectories_legal_masks  # [B, G, T, A] or None

        # Add starting player mask (only consider moves from the starting player's perspective)
        B, G, T = pad_mask.shape
        t = torch.arange(T, device=pad_mask.device)
        start_player_mask = (t % 2 == 0)[None, None, :]  # [1, 1, T]
        effective_pad_mask = pad_mask & start_player_mask  # [B, G, T]

        # Compute loss
        new_log_probs = self.policy_model.get_group_log_probs(trajectories_states,
                                                              trajectories_actions,
                                                              trajectories_legal_masks)

        loss, loss_info = grpo_ppo_loss(new_log_probs,
                             trajectories_old_log_probs,
                             step_rewards,
                             effective_pad_mask,
                             clip_ratio=self.hparams.grpo_config.clip_ratio,
                             kl_coef=self.hparams.grpo_config.kl_coef,
                             entropy_coef=self.hparams.grpo_config.entropy_coef,
                             return_info=True)
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss encountered: {loss.item()}")
        
        # Standard logging
        valid_mask = pad_mask.float()  # [B, G, T] 1 = real step

        self.log("train_total_loss", loss, prog_bar=True)
        self.log("pad_fraction", 1.0 - valid_mask.mean())
        self.log("avg_trajectory_length", pad_mask.float().sum(dim=-1).mean())

        self.log("mean_kl_divergence", loss_info.kl_div)
        self.log("mean_ratio", loss_info.mean_ratio)
        self.log("mean_clip_fraction", loss_info.mean_clip_fraction)
        self.log("ppo_loss", loss_info.ppo_loss)
        self.log("entropy", loss_info.entropy)
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

        return loss

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