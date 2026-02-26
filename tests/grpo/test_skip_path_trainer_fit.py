import chess
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.grpo_logic.loss import GRPOLossInfo
from src.grpo_logic.model import GRPOChessTransformer, GRPOConfig
from src.grpo_logic.sampling import TrajectoriesSample
from src.models import ChessTransformerConfig
from src.searchless_chess_imports import SEQUENCE_LENGTH


def test_trainer_fit_handles_nograd_skip_path(monkeypatch):
    transformer_cfg = ChessTransformerConfig(
        vocab_size=300,
        embed_dim=32,
        num_layers=1,
        num_heads=4,
        action_dim=1968,
    )
    grpo_cfg = GRPOConfig(
        num_trajectories=2,
        trajectory_depth=2,
        ppo_steps=1,
    )
    model = GRPOChessTransformer(transformer_cfg, grpo_cfg)

    def fake_sample_trajectories(*_args, **_kwargs):
        B, G, T = 1, 2, 2
        device = next(model.parameters()).device
        return TrajectoriesSample(
            trajectories_log_probs=torch.zeros(B, G, T, device=device),
            trajectories_actions=torch.zeros(B, G, T, dtype=torch.long, device=device),
            trajectories_states=torch.ones(B, G, T, SEQUENCE_LENGTH, dtype=torch.long, device=device),
            group_rewards=torch.zeros(B, G, device=device),
            step_rewards=torch.zeros(B, G, T, device=device),
            pad_mask=torch.ones(B, G, T, dtype=torch.bool, device=device),
            trajectories_legal_masks=torch.ones(B, G, T, model.policy_model.action_size, dtype=torch.bool, device=device),
            raw_step_cp=torch.zeros(B, G, T, device=device),
            teacher_forced_mask=torch.zeros(B, G, T, dtype=torch.bool, device=device),
        )

    def fake_grpo_ppo_loss(*_args, **_kwargs):
        zero = torch.tensor(0.0)  # Finite scalar, intentionally no grad_fn.
        info = GRPOLossInfo(
            kl_div=zero,
            mean_ratio=zero,
            mean_clip_fraction=zero,
            ppo_loss=zero,
            entropy=zero,
            advantage_mean=zero,
            advantage_std=zero,
        )
        return zero, info

    monkeypatch.setattr("src.grpo_logic.model.sample_trajectories_batched", fake_sample_trajectories)
    monkeypatch.setattr("src.grpo_logic.model.grpo_ppo_loss", fake_grpo_ppo_loss)

    dataset = [chess.Board().fen()]
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
        limit_train_batches=1,
    )
    trainer.fit(model, dataloader)

    # No optimizer updates are expected when all PPO steps are skipped.
    assert trainer.global_step == 0
