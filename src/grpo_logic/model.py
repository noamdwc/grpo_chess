"""Lightning module for reasoning-GRPO fine-tuning of the DM-ported transformer."""
from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

import chess
import pytorch_lightning as pl
import torch

from src.chess.rewards import evaluate_leaf
from src.chess.stockfish import StockfishConfig, StockfishPlayer
from src.configs.config_loader import ExperimentConfig, ModelConfig
from src.grpo_logic.reasoning_loss import ReasoningLossConfig, reasoning_loss
from src.reasoning.model import ReasoningModel, ReasoningModelConfig
from src.reasoning.real_play import play_final_and_respond
from src.reasoning.sampler import rollout_batch
from src.searchless_chess_imports import ACTION_TO_MOVE


def _build_reasoning_model(model_cfg: ModelConfig) -> ReasoningModel:
    rm_cfg = ReasoningModelConfig(
        dm_checkpoint=model_cfg.base_checkpoint,
        max_seq_len=model_cfg.max_seq_len,
        value_head_hidden=model_cfg.value_head.hidden_dim,
        freeze_body=model_cfg.freeze_body,
    )
    return ReasoningModel(rm_cfg)


class ReasoningGRPOLightningModule(pl.LightningModule):
    """Reasoning-GRPO training loop built around the DM-ported transformer."""

    def __init__(self, experiment_cfg: ExperimentConfig) -> None:
        super().__init__()
        self.cfg = experiment_cfg
        self.policy_model = _build_reasoning_model(experiment_cfg.model)
        self.old_policy_model = copy.deepcopy(self.policy_model).eval()
        for param in self.old_policy_model.parameters():
            param.requires_grad_(False)

        self.loss_cfg = ReasoningLossConfig(
            clip_ratio=experiment_cfg.grpo.clip_ratio,
            kl_coef=experiment_cfg.grpo.kl_coef,
            value_coef=experiment_cfg.model.value_head.coef,
        )
        self.frozen_rival_model: ReasoningModel | None = None
        if experiment_cfg.rival.mode == "frozen_dm_9m":
            rival_model_cfg = ReasoningModelConfig(
                dm_checkpoint=experiment_cfg.rival.frozen_dm_9m.checkpoint_path,
                max_seq_len=experiment_cfg.model.max_seq_len,
                value_head_hidden=experiment_cfg.model.value_head.hidden_dim,
                freeze_body=True,
            )
            self.frozen_rival_model = ReasoningModel(rival_model_cfg).eval()
            for param in self.frozen_rival_model.parameters():
                param.requires_grad_(False)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            [param for param in self.policy_model.parameters() if param.requires_grad],
            lr=self.cfg.grpo.lr,
        )

    def on_train_epoch_end(self) -> None:
        """Sync the old policy once per epoch to match the intended PPO contract."""
        self.old_policy_model.load_state_dict(self.policy_model.state_dict())
        self.old_policy_model.eval()

    def _recompute_log_probs(
        self,
        tokens: torch.Tensor,
        seq_lens: torch.Tensor,
        final_move_positions: torch.Tensor,
    ) -> torch.Tensor:
        rollout_temperature = self.cfg.grpo.rollout_temperature
        log_probs = self.policy_model.logprob_sequence(tokens, seq_lens, temperature=rollout_temperature)
        move_temperature = self.cfg.grpo.move_sampling_temperature
        if move_temperature == rollout_temperature:
            return log_probs

        move_log_probs = self.policy_model.logprob_sequence(tokens, seq_lens, temperature=move_temperature)
        batch_idx = torch.arange(tokens.shape[0], device=tokens.device)
        log_probs[batch_idx, final_move_positions] = move_log_probs[batch_idx, final_move_positions]
        return log_probs

    def _post_fen_after_rival(
        self,
        *,
        root_fen: str,
        final_token: int,
        stockfish_player: StockfishPlayer | None = None,
    ) -> tuple[str, bool]:
        rival_mode = self.cfg.rival.mode
        if rival_mode == "self_play":
            return play_final_and_respond(
                model=self.old_policy_model,
                root_fen=root_fen,
                m_final_token=final_token,
            )
        if rival_mode == "frozen_dm_9m":
            if self.frozen_rival_model is None:
                raise RuntimeError("frozen_dm_9m rival requested but frozen_rival_model is not initialized")
            return play_final_and_respond(
                model=self.frozen_rival_model,
                root_fen=root_fen,
                m_final_token=final_token,
            )
        if rival_mode == "stockfish":
            if stockfish_player is None:
                raise RuntimeError("stockfish rival requested but no stockfish player was provided")
            board = chess.Board(root_fen)
            board.push(chess.Move.from_uci(ACTION_TO_MOVE[final_token]))
            if board.is_game_over(claim_draw=True):
                return board.fen(), True
            rival_move = stockfish_player.act(board)
            if rival_move is None:
                return board.fen(), board.is_game_over(claim_draw=True)
            board.push(rival_move)
            return board.fen(), board.is_game_over(claim_draw=True)
        raise ValueError(f"Unknown rival mode: {rival_mode}")

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        del batch_idx
        root_fens = batch["fen"] if isinstance(batch, dict) else list(batch)
        group_size = self.cfg.grpo.k_samples_per_root
        rollout_temperature = self.cfg.grpo.rollout_temperature

        result = rollout_batch(
            model=self.old_policy_model,
            root_fens=root_fens,
            k_samples=group_size,
            min_think_tokens=self.cfg.reasoning.min_think_tokens,
            max_think_tokens=self.cfg.reasoning.max_think_tokens,
            rollout_temperature=rollout_temperature,
            move_sampling_temperature=self.cfg.grpo.move_sampling_temperature,
        )

        rewards: list[float] = []
        v_targets: list[float] = []
        stockfish_player = None
        if self.cfg.rival.mode == "stockfish":
            stockfish_player = StockfishPlayer(
                StockfishConfig(
                    path=self.cfg.stockfish.path,
                    skill_level=self.cfg.rival.stockfish.skill_level,
                    use_elo_limit=self.cfg.stockfish.use_elo_limit,
                    elo=self.cfg.stockfish.elo,
                    movetime_ms=self.cfg.rival.stockfish.movetime_ms,
                    threads=self.cfg.stockfish.threads,
                    hash_mb=self.cfg.stockfish.hash_mb,
                ),
                engine_name=f"reasoning_rival_{id(self)}",
            )
        try:
            for sample_idx, root_fen in enumerate(result.root_fens_replayed):
                pov_is_white = chess.Board(root_fen).turn == chess.WHITE
                final_token = int(
                    result.token_sequences[sample_idx, int(result.final_move_positions[sample_idx].item())].item()
                )
                post_fen, _ = self._post_fen_after_rival(
                    root_fen=root_fen,
                    final_token=final_token,
                    stockfish_player=stockfish_player,
                )
                rewards.append(
                    evaluate_leaf(post_fen, pov_is_white=pov_is_white, mode=self.cfg.leaf_evaluator.mode)
                )
                v_targets.append(
                    evaluate_leaf(
                        result.imagined_leaf_fens[sample_idx],
                        pov_is_white=pov_is_white,
                        mode=self.cfg.leaf_evaluator.mode,
                    )
                )
        finally:
            if stockfish_player is not None:
                stockfish_player.close()

        tokens = result.token_sequences.to(self.device)
        seq_lens = result.seq_lens.to(self.device)
        end_think_positions = result.end_think_positions.to(self.device)
        final_move_positions = result.final_move_positions.to(self.device)
        log_probs_old = result.log_probs_old.to(self.device)
        sampled_mask = result.sampled_mask.to(self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        v_target_t = torch.tensor(v_targets, dtype=torch.float32, device=self.device)

        log_probs_new = self._recompute_log_probs(tokens, seq_lens, final_move_positions)
        out_new = self.policy_model(tokens, seq_lens)
        v_hat = self.policy_model.value_at_end_think(out_new.hidden, end_think_positions)

        loss_result = reasoning_loss(
            result=replace(
                result,
                token_sequences=tokens,
                seq_lens=seq_lens,
                log_probs_old=log_probs_old,
                sampled_mask=sampled_mask,
                end_think_positions=end_think_positions,
                final_move_positions=final_move_positions,
            ),
            log_probs_new=log_probs_new,
            v_hat=v_hat,
            rewards=rewards_t,
            v_target=v_target_t,
            cfg=self.loss_cfg,
            group_size=group_size,
        )

        think_length_mean = (end_think_positions.float() - (result.seq_lens.new_full((), 78).float())).mean()
        reward_spread = rewards_t.view(-1, group_size).std(dim=1, unbiased=False).mean()
        self.log_dict(
            {
                "train/loss": loss_result.total.detach(),
                "train_total_loss": loss_result.total.detach(),
                "train/ppo_loss": loss_result.ppo,
                "train/kl_divergence": loss_result.kl,
                "train/L_value_mean": loss_result.value,
                "train/clip_fraction": loss_result.clip_fraction,
                "train/ratio": loss_result.ratio_mean,
                "train/advantage_mean": loss_result.advantage_mean,
                "train/advantage_std": loss_result.advantage_std,
                "train/reward_mean": rewards_t.mean(),
                "train/leaf_value_mean": v_target_t.mean(),
                "train/v_hat_mean": v_hat.detach().mean(),
                "train/v_hat_minus_R_move_mean": (v_hat.detach() - rewards_t).mean(),
                "train/think_length_mean": think_length_mean,
                "train/reward_spread_within_group": reward_spread,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss_result.total
