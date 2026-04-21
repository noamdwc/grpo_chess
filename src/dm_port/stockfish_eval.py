"""Helpers for evaluating the bare DM-port checkpoint against Stockfish."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np
import torch

from src.chess.policy_player import PolicyConfig
from src.dm_port.transformer import DMTransformer, DMTransformerConfig
from src.evaluator import Evaluator
from src.reasoning.model import ReasoningModel, ReasoningModelConfig
from src.searchless_chess_imports import MOVE_TO_ACTION, tokenize

DEFAULT_DM_PORT_CHECKPOINT = Path("checkpoints/dm_port/9M.pt")


def _bucket_values(num_return_buckets: int) -> torch.Tensor:
    full = np.linspace(0.0, 1.0, num_return_buckets + 1, dtype=np.float32)
    return torch.from_numpy((full[:-1] + full[1:]) / 2.0)


def load_dm_port_model(checkpoint_path: str | Path, *, device: torch.device | str = "cpu") -> DMTransformer:
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model = DMTransformer(DMTransformerConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def load_dm_port_and_reasoning_models(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    reasoning_max_seq_len: int = 120,
    value_head_hidden: int = 256,
) -> tuple[DMTransformer, ReasoningModel]:
    dm_model = load_dm_port_model(checkpoint_path, device=device)
    reasoning_model = ReasoningModel(
        ReasoningModelConfig(
            dm_checkpoint=str(checkpoint_path),
            max_seq_len=reasoning_max_seq_len,
            value_head_hidden=value_head_hidden,
        )
    ).to(device).eval()
    assert_transformer_body_matches(dm_model, reasoning_model)
    return dm_model, reasoning_model


def assert_transformer_body_matches(dm_model: DMTransformer, reasoning_model: ReasoningModel) -> None:
    dm_state = dm_model.state_dict()
    wrapped_state = reasoning_model.dm.state_dict()
    for key, dm_value in dm_state.items():
        if key.startswith("output_linear."):
            continue
        wrapped_value = wrapped_state[key]
        if key in {"token_embedding.weight", "pos_embedding.weight"}:
            shared = wrapped_value[: dm_value.shape[0]]
            if not torch.equal(shared.cpu(), dm_value.cpu()):
                raise AssertionError(f"Wrapped model diverged from DM checkpoint for {key}")
            continue
        if not torch.equal(wrapped_value.cpu(), dm_value.cpu()):
            raise AssertionError(f"Wrapped model diverged from DM checkpoint for {key}")


@dataclass
class DMPortEvalPolicy:
    """Action-value wrapper exposing the DM port through the eval policy API."""

    model: DMTransformer

    def __post_init__(self) -> None:
        self.model = self.model.eval()
        self.device = next(self.model.parameters()).device
        self._bucket_values = _bucket_values(self.model.config.output_size).to(self.device)

    @torch.no_grad()
    def act(self, board: chess.Board) -> chess.Move:
        legal_moves = sorted(
            [move for move in board.legal_moves if move.uci() in MOVE_TO_ACTION],
            key=lambda move: MOVE_TO_ACTION[move.uci()],
        )
        if not legal_moves:
            raise RuntimeError(f"No mapped legal moves available for {board.fen()}")

        fen_tokens = torch.from_numpy(np.asarray(tokenize(board.fen()), dtype=np.int64)).to(self.device)
        action_tokens = torch.tensor(
            [MOVE_TO_ACTION[move.uci()] for move in legal_moves],
            dtype=torch.long,
            device=self.device,
        )
        targets = fen_tokens.unsqueeze(0).repeat(len(legal_moves), 1)
        targets = torch.cat(
            [
                targets,
                action_tokens.unsqueeze(1),
                torch.zeros((len(legal_moves), 1), dtype=torch.long, device=self.device),
            ],
            dim=1,
        )
        log_probs = self.model(targets)[:, -1, :]
        win_probs = torch.exp(log_probs) @ self._bucket_values
        self._apply_repetition_draw_scores(board, legal_moves, win_probs)
        return legal_moves[int(torch.argmax(win_probs).item())]

    @staticmethod
    def _apply_repetition_draw_scores(
        board: chess.Board,
        legal_moves: list[chess.Move],
        win_probs: torch.Tensor,
    ) -> None:
        for idx, move in enumerate(legal_moves):
            board.push(move)
            if board.is_fivefold_repetition() or board.can_claim_threefold_repetition():
                win_probs[idx] = 0.5
            board.pop()


class DMPortEvaluator(Evaluator):
    """Evaluator wired to the bare DM-port action-value model."""

    def __init__(self, *, eval_cfg, stockfish_cfg) -> None:
        super().__init__(eval_cfg=eval_cfg, policy_cfg=PolicyConfig(greedy=True), stockfish_cfg=stockfish_cfg)

    def _make_policy(self, model):
        return DMPortEvalPolicy(model)
