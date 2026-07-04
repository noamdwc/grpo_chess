"""Lightning module for searchless AV-PPO post-training (docs/av_ppo_rebuild_plan.md)."""
from __future__ import annotations

import copy
from typing import Any

import chess
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from src.chess.chess_logic import action_to_move, build_legal_mask_from_fen
from src.chess.rewards import evaluate_leaf
from src.configs.config_loader import AVPPOExperimentConfig
from src.dm_port.av_adapter import AVAdapter, PPOActorCritic
from src.dm_port.stockfish_eval import load_dm_port_model
from src.grpo_logic.ppo_loss import safe_ppo_loss
from src.searchless_chess_imports import tokenize


def build_ppo_actor_critic(base_checkpoint: str, freeze_backbone: bool = False) -> PPOActorCritic:
    dm_model = load_dm_port_model(base_checkpoint)
    model = PPOActorCritic(AVAdapter(dm_model))
    if freeze_backbone:
        for param in model.av_adapter.parameters():
            param.requires_grad_(False)
    return model


class AVPPOLightningModule(pl.LightningModule):
    """Searchless PPO on the DM AV backbone.

    Per training step: sample one action per board from the old policy, score the
    post-action state with the configured leaf evaluator (Stockfish scalar by
    default), advantage = reward - critic value, then run `ppo_epochs` clipped
    PPO updates on that rollout batch and sync the old policy.
    """

    def __init__(self, experiment_cfg: AVPPOExperimentConfig) -> None:
        super().__init__()
        # ppo_epochs takes several optimizer steps per rollout batch, which
        # Lightning only allows with manual optimization.
        self.automatic_optimization = False
        if experiment_cfg.ppo.ppo_epochs < 1:
            raise ValueError(f"ppo.ppo_epochs must be >= 1, got {experiment_cfg.ppo.ppo_epochs}")
        self.cfg = experiment_cfg
        self.policy_model = build_ppo_actor_critic(
            experiment_cfg.model.base_checkpoint,
            freeze_backbone=experiment_cfg.model.freeze_backbone,
        )
        self.old_policy_model = copy.deepcopy(self.policy_model).eval()
        for param in self.old_policy_model.parameters():
            param.requires_grad_(False)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            [param for param in self.policy_model.parameters() if param.requires_grad],
            lr=self.cfg.ppo.lr,
        )

    def on_fit_start(self) -> None:
        # Log the actual optimization contract so runs cannot be misdiagnosed:
        # run lrreykii advertised ppo_epochs: 4 in its config while the training
        # loop silently ran one update per rollout.
        if self.logger is not None:
            self.logger.log_hyperparams(
                {
                    "ppo/ppo_epochs": self.cfg.ppo.ppo_epochs,
                    "ppo/clip_eps": self.cfg.ppo.clip_eps,
                    "ppo/kl_coef": self.cfg.ppo.kl_coef,
                    "ppo/entropy_coef": self.cfg.ppo.entropy_coef,
                    "ppo/critic_coef": self.cfg.ppo.critic_coef,
                    "ppo/rollout_temperature": self.cfg.ppo.rollout_temperature,
                    "ppo/reward_source": self.cfg.leaf_evaluator.mode,
                    "ppo/base_checkpoint": self.cfg.model.base_checkpoint,
                    "ppo/freeze_backbone": self.cfg.model.freeze_backbone,
                }
            )

    def _encode_batch(self, fens: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        board_tokens = torch.tensor(
            np.stack([tokenize(fen) for fen in fens]), dtype=torch.long, device=self.device
        )
        legal_masks = torch.stack([build_legal_mask_from_fen(fen, self.device) for fen in fens])
        return board_tokens, legal_masks

    def _actor_logprobs(
        self, model: PPOActorCritic, board_tokens: torch.Tensor, legal_masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Sampling and both old/new log-probs share rollout_temperature so the
        # importance ratio starts at exactly 1.0 (the 2026-02-07 temperature-
        # mismatch bug came from scoring at a different temperature than sampling).
        out = model(board_tokens, legal_masks)
        logprobs = F.log_softmax(out["actor_logits"] / self.cfg.ppo.rollout_temperature, dim=-1)
        return logprobs, out["value"]

    def _reward(self, board: chess.Board, pov_is_white: bool) -> float:
        return evaluate_leaf(
            board.fen(),
            pov_is_white=pov_is_white,
            mode=self.cfg.leaf_evaluator.mode,
            movetime_ms=self.cfg.leaf_evaluator.stockfish.movetime_ms,
        )

    def training_step(self, batch: Any, batch_idx: int) -> None:
        del batch_idx
        fens = batch["fen"] if isinstance(batch, dict) else list(batch)
        board_tokens, legal_masks = self._encode_batch(fens)

        with torch.no_grad():
            logprobs_old, _ = self._actor_logprobs(self.old_policy_model, board_tokens, legal_masks)
            actions = torch.multinomial(logprobs_old.exp(), num_samples=1).squeeze(-1)

        rewards: list[float] = []
        for fen, action_idx in zip(fens, actions.tolist()):
            board = chess.Board(fen)
            pov_is_white = board.turn == chess.WHITE
            move = action_to_move(board, action_idx)
            if move is None:
                raise RuntimeError(f"Sampled illegal action {action_idx} for {fen}")
            board.push(move)
            rewards.append(self._reward(board, pov_is_white))
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        selected_mask = torch.zeros_like(legal_masks)
        selected_mask[torch.arange(len(fens), device=self.device), actions] = True

        optimizer = self.optimizers()
        ratio_first = None
        for _ in range(self.cfg.ppo.ppo_epochs):
            logprobs_new, values = self._actor_logprobs(self.policy_model, board_tokens, legal_masks)
            advantages_per_board = rewards_t - values.detach()
            advantages = advantages_per_board[:, None].expand_as(logprobs_new)
            loss, info = safe_ppo_loss(
                logprobs_new,
                logprobs_old,
                advantages,
                values=values,
                targets=rewards_t,
                selected_mask=selected_mask,
                pad_mask=legal_masks,
                clip_eps=self.cfg.ppo.clip_eps,
                kl_coef=self.cfg.ppo.kl_coef,
                entropy_coef=self.cfg.ppo.entropy_coef,
                critic_coef=self.cfg.ppo.critic_coef,
                return_info=True,
            )
            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()
            if ratio_first is None:
                ratio_first = info.mean_ratio
            last_loss = loss.detach()

        # Rollouts are re-sampled every step, so sync the old policy per rollout
        # batch: the ppo_epochs loop is the entire off-policy window.
        self.old_policy_model.load_state_dict(self.policy_model.state_dict())

        with torch.no_grad():
            ratio_selected = (logprobs_new - logprobs_old)[selected_mask].exp()
            clip_fraction = ((ratio_selected - 1.0).abs() > self.cfg.ppo.clip_eps).float().mean()

        self.log_dict(
            {
                "train/loss": last_loss,
                "train_total_loss": last_loss,  # checkpoint callbacks monitor this key
                "train/ppo_loss": info.ppo_loss,
                "train/kl_divergence": info.kl_loss,
                "train/entropy": info.entropy,
                "train/critic_loss": info.critic_loss,
                "train/ratio": info.mean_ratio,
                # ≈1.0 by construction on the first ppo epoch; drift here means the
                # sampling and scoring paths diverged (temperature-mismatch class).
                "train/ratio_first_epoch": ratio_first,
                "train/clip_fraction": clip_fraction,
                "train/reward_mean": rewards_t.mean(),
                "train/advantage_mean": advantages_per_board.mean(),
                "train/advantage_std": advantages_per_board.std(unbiased=False),
                "train/value_mean": values.detach().mean(),
                "train/ppo_epochs_ran": float(self.cfg.ppo.ppo_epochs),
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(fens),
        )
