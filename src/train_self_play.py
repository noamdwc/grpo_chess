"""Entry point for reasoning-GRPO fine-tuning."""
from __future__ import annotations

import argparse
import copy
from typing import Any

import chess
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.chess.boards_dataset import ChessStartStatesDataset
from src.chess.policy_player import PolicyConfig
from src.chess.stockfish import resolve_stockfish_path
from src.configs.config_loader import load_experiment_config
from src.evaluator import Evaluator, StockfishEvalCallback
from src.grpo_logic.model import ReasoningGRPOLightningModule
from src.reasoning.tokens import BASE_VOCAB_SIZE, END_THINK_ID, FEN_LEN, MOVE_ID, THINK_ID
from src.searchless_chess_imports import ACTION_TO_MOVE, MOVE_TO_ACTION, tokenize
from src.trainer import get_trainer


class ReasoningEvalPolicy:
    """Adapter that exposes the refactored reasoning model through the eval policy API."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model.eval()
        self.device = next(model.parameters()).device

    def act(self, board: chess.Board) -> chess.Move:
        fen_tokens = torch.from_numpy(np.asarray(tokenize(board.fen()), dtype=np.int64)).to(self.device)
        seq = torch.zeros((1, FEN_LEN + 4), dtype=torch.long, device=self.device)
        seq[0, :FEN_LEN] = fen_tokens
        seq[0, FEN_LEN] = THINK_ID
        seq[0, FEN_LEN + 1] = END_THINK_ID
        seq[0, FEN_LEN + 2] = MOVE_ID
        seq_lens = torch.tensor([FEN_LEN + 3], dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model(seq[:, : FEN_LEN + 3], seq_lens)
        logits = out.logits[0, FEN_LEN + 2, :BASE_VOCAB_SIZE]

        legal_indices = [MOVE_TO_ACTION[move.uci()] for move in board.legal_moves if move.uci() in MOVE_TO_ACTION]
        if not legal_indices:
            raise RuntimeError(f"No legal moves available during eval for {board.fen()}")
        legal_logits = logits[legal_indices]
        best_idx = legal_indices[int(torch.argmax(legal_logits).item())]
        return chess.Move.from_uci(ACTION_TO_MOVE[best_idx])


class ReasoningEvaluator(Evaluator):
    """Evaluator wired to the sequence-based reasoning model."""

    def __init__(self, *, eval_cfg, stockfish_cfg) -> None:
        super().__init__(eval_cfg=eval_cfg, policy_cfg=PolicyConfig(greedy=True), stockfish_cfg=stockfish_cfg)

    def _make_policy(self, model):
        return ReasoningEvalPolicy(model)


def build_stockfish_eval_callback(cfg) -> StockfishEvalCallback:
    try:
        resolved_stockfish = resolve_stockfish_path(cfg.stockfish.path)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Stockfish evaluation is required but no Stockfish binary was found. "
            "Set stockfish.path/STOCKFISH_PATH or install stockfish."
        ) from exc

    stockfish_cfg = copy.copy(cfg.stockfish)
    stockfish_cfg.path = resolved_stockfish
    evaluator = ReasoningEvaluator(eval_cfg=cfg.eval, stockfish_cfg=stockfish_cfg)
    return StockfishEvalCallback(
        evaluator,
        every_n_epochs=cfg.grpo.eval_every_n_epochs,
        metric_prefix="eval_stockfish",
        run_on_fit_start=True,
        init_metric_prefix="eval_stockfish_init",
    )


def train(
    config_path: str = "default.yaml",
    overrides: dict[str, dict[str, Any]] | None = None,
    resume_from_checkpoint: str | None = None,
) -> None:
    cfg = load_experiment_config(config_path, overrides=overrides)

    dataset = ChessStartStatesDataset(cfg.dataset)
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, num_workers=0)
    module = ReasoningGRPOLightningModule(cfg)
    eval_callback = build_stockfish_eval_callback(cfg)
    trainer = get_trainer(
        num_epochs=cfg.training.num_epochs,
        checkpoint_dir=cfg.training.checkpoint_dir,
        checkpoint_every_n_epochs=cfg.training.checkpoint_every_n_epochs,
        keep_n_checkpoints=cfg.training.keep_n_checkpoints,
        use_wandb=cfg.training.use_wandb,
        wandb_project=cfg.training.wandb_project,
        callbacks=[eval_callback],
    )
    trainer.fit(module, dataloader, ckpt_path=resume_from_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="default.yaml")
    parser.add_argument("--resume_from_checkpoint", default=None)
    args = parser.parse_args()
    train(args.config, resume_from_checkpoint=args.resume_from_checkpoint)
