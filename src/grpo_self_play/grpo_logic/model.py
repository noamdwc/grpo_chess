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
    lr: float = 1e-4
    num_trajectories: int = 4
    trajectory_depth: int = 5
    clip_ratio: float = 0.2
    kl_coef: float = 0.01
    eval_every_n_epochs: int = 10 # Not used in model, but useful for trainer


class GRPOChessTransformer(pl.LightningModule):
    def __init__(self,
                 transformer_config: ChessTransformerConfig,
                 grpo_config: GRPOConfig,
                 eval_cfg: EvalConfig | None = None,
                 stockfish_cfg: StockfishConfig | None = None,
                 policy_cfg: PolicyConfig | None = None,
                 searcher_cfg: SearchConfig | None = None):
        super().__init__()
        self.save_hyperparameters()
        self.policy_model = ChessTransformer(transformer_config)
        self.old_policy_model = ChessTransformer(transformer_config)
        self._sync_old_policy()

        # evaluation config
        self.eval_every_n_epochs = grpo_config.eval_every_n_epochs
        self.evaluator = Evaluator(eval_cfg=eval_cfg or EvalConfig(),
                                   policy_cfg=policy_cfg or PolicyConfig(),
                                   stockfish_cfg=stockfish_cfg or StockfishConfig(),
                                   searcher_cfg=searcher_cfg)

    def forward(self, x):
        return self.policy_model(x)

    def _old_forward(self, x):
        return self.old_policy_model(x)

    def _sync_old_policy(self):
        self.old_policy_model.load_state_dict(self.policy_model.state_dict())
        # Freeze
        for param in self.old_policy_model.parameters():
            param.requires_grad = False

    def _log_rewards_metrics(self, batch_group_rewards, prefix="train/"):
        mean_r = batch_group_rewards.mean()
        best = batch_group_rewards.max()
        gap = best - mean_r

        self.log(prefix + "avg_reward", mean_r, prog_bar=True)
        self.log(prefix + "reward_std", batch_group_rewards.std())
        self.log(prefix + "reward_p50", batch_group_rewards.median())
        self.log(prefix + "reward_p90", batch_group_rewards.quantile(0.9))
        self.log(prefix + "reward_best", best)
        self.log(prefix + "reward_gap_best_minus_mean", gap)

    def on_train_epoch_start(self):
        self._sync_old_policy()

    def training_step(self, batch_fens, batch_idx):
        boards = [chess.Board(start_fen) for start_fen in batch_fens]
        boards = [board for board in boards if not board.is_game_over()]
        if not boards: return 0.0 # Skip if game over

        trajectories_sample = sample_trajectories_batched(self.old_policy_model,
                                                          boards,
                                                          self.hparams.grpo_config.num_trajectories,
                                                          self.hparams.grpo_config.trajectory_depth)
        if trajectories_sample is None: return 0 # Skip if no moves

        trajectories_old_log_probs = trajectories_sample.trajectories_log_probs # [B, G, T]
        trajectories_actions = trajectories_sample.trajectories_actions # [B, G, T]
        trajectories_states = trajectories_sample.trajectories_states # [B, G, T, SEQ]
        batch_group_rewards = trajectories_sample.group_rewards # [B, G]
        pad_mask = trajectories_sample.pad_mask # [B, G, T]

        # Compute loss
        new_log_probs = self.policy_model.get_group_log_probs(trajectories_states,
                                                              trajectories_actions)

        loss, loss_info = grpo_ppo_loss(new_log_probs,
                             trajectories_old_log_probs,
                             batch_group_rewards,
                             pad_mask,
                             clip_ratio=self.hparams.grpo_config.clip_ratio,
                             kl_coef=self.hparams.grpo_config.kl_coef,
                             return_info=True)
        # Standard Logging
        valid_mask = (~pad_mask).float()  # [B, G, T] 1 = real step

        self.log("train_total_loss", loss, prog_bar=True)
        self.log("pad_fraction", 1.0 - valid_mask.mean())
        self.log('avg_trajectory_length', pad_mask.float().sum(dim=-1).mean())

        self.log("mean_kl_divergence", loss_info.kl_div)
        self.log("mean_ratio", loss_info.mean_ratio)
        self.log("mean_clip_fraction", loss_info.mean_clip_fraction)
        self.log("ppo_loss", loss_info.ppo_lose)
        self._log_rewards_metrics(batch_group_rewards, prefix="train/")
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.grpo_config.lr)

    def _evaluate_against_stockfish(self) -> Optional[dict]:
        '''
        Run a single game evaluation against Stockfish with current policy model.
        '''
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                results, _ = self.evaluator.single_evaluation(self.policy_model)
            return results
        except Exception as e:
            print(f"Evaluation against Stockfish failed: {e}")
            return None
        finally:
            if was_training:
                self.train()

    def _log_stockfish_eval(self, results: dict):
        """
        Log scalar evaluation metrics from the Stockfish eval.
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
    
    def on_train_epoch_end(self):
        if (self.current_epoch + 1) % self.eval_every_n_epochs == 0:
            results = self._evaluate_against_stockfish()
            if results is not None:
                self._log_stockfish_eval(results)