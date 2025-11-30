import torch
import pytorch_lightning as pl
import chess
from src.grpo_self_play.models import ChessTransformer, ChessTransformerConfig
from src.grpo_self_play.grpo_logic.loss import grpo_ppo_loss
from src.grpo_self_play.grpo_logic.sampling import sample_trajectories
from dataclasses import dataclass


@dataclass
class GRPOConfig:
    lr: float = 1e-4
    num_trajectories: int = 4
    trajectory_depth: int = 5
    clip_ratio: float = 0.2
    kl_coef: float = 0.01


class GRPOChessTransformer(pl.LightningModule):
    def __init__(self,
                 transformer_confg: ChessTransformerConfig,
                 grpo_config: GRPOConfig):
        super().__init__()
        self.save_hyperparameters()
        self.policy_model = ChessTransformer(transformer_confg)
        self.old_policy_model = ChessTransformer(transformer_confg)
        self._sync_old_policy()

    def forward(self, x):
        return self.policy_model(x)

    def _old_forward(self, x):
        return self.old_policy_model(x)

    def _sync_old_policy(self):
        self.old_policy_model.load_state_dict(self.policy_model.state_dict())
        # Freeze
        for param in self.old_policy_model.parameters():
            param.requires_grad = False

    def on_train_epoch_start(self):
        self._sync_old_policy()

    def training_step(self, batch_fens, batch_idx):
        batch_size = len(batch_fens)
        boards = [chess.Board(start_fen) for start_fen in batch_fens]
        boards = [board for board in boards if not board.is_game_over()]
        if not boards: return 0.0 # Skip if game over

        traj_sample = []
        for board_start in boards:
            trajectories_sample = sample_trajectories(self.old_policy_model,
                                                      self.hparams.grpo_config.num_trajectories,
                                                      self.hparams.grpo_config.trajectory_depth,
                                                      board_start)
            if trajectories_sample is None: continue # Skip if no moves
            traj_sample.append(trajectories_sample)
        if not traj_sample: return 0.0 # Skip if no trajectories

        # Stack tensors
        trajectories_old_log_probs = torch.stack(
            [sample.trajectories_log_probs for sample in traj_sample], dim=0) # [B, G, T]
        trajectories_actinos = torch.stack(
            [sample.trajectories_actinos for sample in traj_sample], dim=0) # [B, G, T]
        trajectories_states = torch.stack(
            [sample.trajectories_states for sample in traj_sample], dim=0) # [B, G, T, SEQ]
        batch_group_rewards = torch.stack(
            [sample.group_rewards for sample in traj_sample], dim=0) # [B, G]
        pad_mask = torch.stack(
            [sample.pad_mask for sample in traj_sample], dim=0) # [B, G, T]

        B, G, T = trajectories_old_log_probs.shape

        # Compute loss
        new_log_probs = self.policy_model.get_group_log_probs(trajectories_states,
                                                              trajectories_actinos)

        loss = grpo_ppo_loss(new_log_probs,
                             trajectories_old_log_probs,
                             batch_group_rewards,
                             pad_mask,
                             clip_ratio=self.hparams.grpo_config.clip_ratio,
                             kl_coef=self.hparams.grpo_config.kl_coef)
        # Standard Logging
        self.log("train_total_loss", loss, prog_bar=True)
        self.log("avg_reward", batch_group_rewards.mean(), prog_bar=True)
        self.log("reward_std", batch_group_rewards.std(), prog_bar=False)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.grpo_config.lr)