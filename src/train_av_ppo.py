"""Entry point for searchless AV-PPO post-training."""
from __future__ import annotations

import argparse
import dataclasses
import random
from typing import Any

import chess
import torch
from torch.utils.data import DataLoader

from src.chess.boards_dataset import ChessStartStatesDataset
from src.chess.chess_logic import action_to_move, board_to_tensor, get_legal_moves_mask
from src.chess.stockfish import StockfishManager, resolve_stockfish_path
from src.configs.config_loader import AVPPOExperimentConfig, load_av_ppo_config
from src.evaluator import Evaluator, StockfishEvalCallback
from src.grpo_logic.ppo_model import AVPPOLightningModule
from src.trainer import get_trainer


class AVEvalPolicy:
    """Greedy eval adapter: chess.Board -> DM FEN tokens -> PPOActorCritic actor logits."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model.eval()
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def act(self, board: chess.Board) -> chess.Move:
        board_tensor = board_to_tensor(board, self.device)
        legal_mask = get_legal_moves_mask(board, self.device).unsqueeze(0)
        logits = self.model(board_tensor, legal_mask)["actor_logits"]
        move = action_to_move(board, int(torch.argmax(logits, dim=-1).item()))
        if move is None:
            return random.choice(list(board.legal_moves))
        return move


class AVPPOEvaluator(Evaluator):
    """Evaluator wired to the AV actor-critic model."""

    def _make_policy(self, model: torch.nn.Module) -> AVEvalPolicy:
        return AVEvalPolicy(model)


def build_stockfish_eval_callback(cfg: AVPPOExperimentConfig) -> StockfishEvalCallback:
    try:
        resolved_stockfish = resolve_stockfish_path(cfg.stockfish.path)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Stockfish evaluation is required but no Stockfish binary was found. "
            "Set stockfish.path/STOCKFISH_PATH or install stockfish."
        ) from exc

    stockfish_cfg = dataclasses.replace(cfg.stockfish, path=resolved_stockfish)
    evaluator = AVPPOEvaluator(eval_cfg=cfg.eval, stockfish_cfg=stockfish_cfg)
    return StockfishEvalCallback(
        evaluator=evaluator,
        every_n_epochs=cfg.ppo.eval_every_n_epochs,
        model_attr="policy_model",
        metric_prefix="eval_stockfish",
        run_on_fit_start=True,
        init_metric_prefix="eval_stockfish_init",
    )


def train(
    config_path: str = "ppo_av.yaml",
    overrides: dict[str, dict[str, Any]] | None = None,
    resume_from_checkpoint: str | None = None,
) -> None:
    cfg = load_av_ppo_config(config_path, overrides=overrides)

    dataset = ChessStartStatesDataset(cfg.dataset)
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, num_workers=0)
    module = AVPPOLightningModule(cfg)
    trainer = get_trainer(
        num_epochs=cfg.training.num_epochs,
        checkpoint_dir=cfg.training.checkpoint_dir,
        checkpoint_every_n_epochs=cfg.training.checkpoint_every_n_epochs,
        keep_n_checkpoints=cfg.training.keep_n_checkpoints,
        use_wandb=cfg.training.use_wandb,
        wandb_project=cfg.training.wandb_project,
        callbacks=[build_stockfish_eval_callback(cfg)],
    )
    try:
        trainer.fit(module, dataloader, ckpt_path=resume_from_checkpoint)
    finally:
        # Stockfish engines keep non-daemon SimpleEngine threads alive; without
        # an explicit close the process never exits after fit (it hangs in
        # threading._shutdown), which strands cloud jobs. See
        # docs/reasoning_grpo/failure_modes.md ("Run hangs at exit").
        StockfishManager.close_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ppo_av.yaml")
    parser.add_argument("--resume_from_checkpoint", default=None)
    args = parser.parse_args()
    train(args.config, resume_from_checkpoint=args.resume_from_checkpoint)
