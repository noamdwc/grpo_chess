"""Autoregressive rollout with per-sample imagined boards and legal masking."""
from dataclasses import dataclass

import chess
import numpy as np
import torch
import torch.nn.functional as F

from src.reasoning.model import ReasoningModel
from src.reasoning.tokens import (
    BASE_VOCAB_SIZE,
    END_THINK_ID,
    FEN_LEN,
    MOVE_ID,
    THINK_ID,
    TOTAL_VOCAB_SIZE,
)
from src.searchless_chess_imports import ACTION_TO_MOVE, MOVE_TO_ACTION, tokenize


@dataclass
class RolloutResult:
    token_sequences: torch.Tensor
    seq_lens: torch.Tensor
    log_probs_old: torch.Tensor
    sampled_mask: torch.Tensor
    legal_token_mask: torch.Tensor
    end_think_positions: torch.Tensor
    final_move_positions: torch.Tensor
    imagined_leaf_fens: list[str]
    root_fens_replayed: list[str]
    think_tokens: list[list[int]]


def _legal_mask_for_board(board: chess.Board, *, include_special: bool = False) -> torch.Tensor:
    """Return a [TOTAL_VOCAB_SIZE] bool tensor: True means allowed."""
    mask = torch.zeros(TOTAL_VOCAB_SIZE, dtype=torch.bool)
    for move in board.legal_moves:
        action = MOVE_TO_ACTION.get(move.uci())
        if action is not None:
            mask[action] = True
    if not include_special:
        return mask
    return mask


def _sample_with_mask(
    logits: torch.Tensor,
    allow_mask: torch.Tensor,
    temperature: float,
) -> tuple[int, float]:
    """Sample one allowed token and return its id plus log-prob."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not allow_mask.any():
        raise ValueError("allow_mask must permit at least one token")
    scaled = logits / temperature
    scaled = scaled.masked_fill(~allow_mask, float("-inf"))
    log_probs = F.log_softmax(scaled, dim=-1)
    probs = log_probs.exp()
    idx = int(torch.multinomial(probs, num_samples=1).item())
    return idx, float(log_probs[idx].item())


@torch.no_grad()
def rollout_batch(
    *,
    model: ReasoningModel,
    root_fens: list[str],
    k_samples: int,
    min_think_tokens: int,
    max_think_tokens: int,
    rollout_temperature: float,
    move_sampling_temperature: float = 1.0,
) -> RolloutResult:
    """Sample K reasoning sequences per root FEN."""
    if k_samples <= 0:
        raise ValueError("k_samples must be positive")
    if min_think_tokens < 0:
        raise ValueError("min_think_tokens must be non-negative")
    if max_think_tokens < min_think_tokens:
        raise ValueError("max_think_tokens must be >= min_think_tokens")

    device = next(model.parameters()).device
    num_roots = len(root_fens)
    num_samples = num_roots * k_samples
    max_len = model.config.max_seq_len

    boards: list[chess.Board] = []
    real_roots: list[chess.Board] = []
    fen_token_arrays: list[np.ndarray] = []
    root_fens_replayed: list[str] = []
    for fen in root_fens:
        fen_tokens = np.asarray(tokenize(fen), dtype=np.int64)
        for _ in range(k_samples):
            boards.append(chess.Board(fen))
            real_roots.append(chess.Board(fen))
            fen_token_arrays.append(fen_tokens)
            root_fens_replayed.append(fen)

    seq = torch.zeros((num_samples, max_len), dtype=torch.long, device=device)
    log_probs = torch.zeros((num_samples, max_len), dtype=torch.float32, device=device)
    sampled = torch.zeros((num_samples, max_len), dtype=torch.bool, device=device)
    legal_token_mask = torch.ones((num_samples, max_len, TOTAL_VOCAB_SIZE), dtype=torch.bool, device=device)

    for idx in range(num_samples):
        seq[idx, :FEN_LEN] = torch.from_numpy(fen_token_arrays[idx]).to(device)
        seq[idx, FEN_LEN] = THINK_ID
    cur_len = torch.full((num_samples,), FEN_LEN + 1, dtype=torch.long, device=device)

    think_tokens: list[list[int]] = [[] for _ in range(num_samples)]
    end_think_positions = torch.full((num_samples,), -1, dtype=torch.long, device=device)
    final_move_positions = torch.full((num_samples,), -1, dtype=torch.long, device=device)
    imagined_leaf_fens = [""] * num_samples
    ended = torch.zeros((num_samples,), dtype=torch.bool, device=device)

    for step in range(max_think_tokens + 1):
        active = (~ended).nonzero(as_tuple=True)[0]
        if active.numel() == 0:
            break
        step_max_len = int(cur_len.max().item())
        out = model(seq[:, :step_max_len], cur_len)
        for sample_idx in active.tolist():
            board = boards[sample_idx]
            logits = out.logits[sample_idx, int(cur_len[sample_idx].item()) - 1]
            legal = _legal_mask_for_board(board).to(device)
            if not legal.any():
                legal[END_THINK_ID] = True
            if step >= min_think_tokens:
                legal[END_THINK_ID] = True
            if step == max_think_tokens:
                legal[:] = False
                legal[END_THINK_ID] = True

            token_id, log_prob = _sample_with_mask(logits, legal, rollout_temperature)
            new_pos = int(cur_len[sample_idx].item())
            if new_pos >= max_len:
                raise ValueError("rollout sequence exceeded model max_seq_len")
            seq[sample_idx, new_pos] = token_id
            log_probs[sample_idx, new_pos] = log_prob
            sampled[sample_idx, new_pos] = True
            legal_token_mask[sample_idx, new_pos] = legal
            cur_len[sample_idx] += 1

            if token_id == END_THINK_ID:
                end_think_positions[sample_idx] = new_pos
                imagined_leaf_fens[sample_idx] = board.fen()
                ended[sample_idx] = True
                continue

            move = chess.Move.from_uci(ACTION_TO_MOVE[token_id])
            board.push(move)
            think_tokens[sample_idx].append(token_id)

    if (end_think_positions < 0).any():
        raise RuntimeError("rollout ended without recording <end_think> for every sample")

    for sample_idx in range(num_samples):
        move_pos = int(cur_len[sample_idx].item())
        if move_pos >= max_len:
            raise ValueError("rollout sequence exceeded model max_seq_len before <move>")
        seq[sample_idx, move_pos] = MOVE_ID
        cur_len[sample_idx] += 1

    step_max_len = int(cur_len.max().item())
    out = model(seq[:, :step_max_len], cur_len)
    for sample_idx in range(num_samples):
        logits = out.logits[sample_idx, int(cur_len[sample_idx].item()) - 1]
        legal = _legal_mask_for_board(real_roots[sample_idx]).to(device)
        legal[BASE_VOCAB_SIZE:] = False
        token_id, log_prob = _sample_with_mask(logits, legal, move_sampling_temperature)
        final_pos = int(cur_len[sample_idx].item())
        if final_pos >= max_len:
            raise ValueError("rollout sequence exceeded model max_seq_len at final move")
        seq[sample_idx, final_pos] = token_id
        log_probs[sample_idx, final_pos] = log_prob
        sampled[sample_idx, final_pos] = True
        legal_token_mask[sample_idx, final_pos] = legal
        final_move_positions[sample_idx] = final_pos
        cur_len[sample_idx] += 1

    token_sequences = seq[:, : int(cur_len.max().item())].clone()
    log_probs_old = log_probs[:, : token_sequences.shape[1]].clone()
    sampled_mask = sampled[:, : token_sequences.shape[1]].clone()
    legal_token_mask = legal_token_mask[:, : token_sequences.shape[1]].clone()

    return RolloutResult(
        token_sequences=token_sequences,
        seq_lens=cur_len.clone(),
        log_probs_old=log_probs_old,
        sampled_mask=sampled_mask,
        legal_token_mask=legal_token_mask,
        end_think_positions=end_think_positions.clone(),
        final_move_positions=final_move_positions.clone(),
        imagined_leaf_fens=imagined_leaf_fens,
        root_fens_replayed=root_fens_replayed,
        think_tokens=think_tokens,
    )
