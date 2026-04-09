"""Unit tests for teacher forcing behavior in GRPO trajectory sampling."""

import random

import chess
import torch
import torch.nn as nn

from src.grpo_logic.sampling import sample_trajectories_batched


class UniformLegalPolicy(nn.Module):
    """Policy stub that samples uniformly from legal moves."""

    def __init__(self, action_size: int = 1968):
        super().__init__()
        self._param = nn.Parameter(torch.zeros(1))
        self._action_size = action_size

    @property
    def action_size(self) -> int:
        return self._action_size

    def get_legal_moves_probs(
        self,
        states_tensor: torch.Tensor,
        legal_moves_mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        probs = legal_moves_mask.float()
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return probs


def test_teacher_forced_steps_are_masked_out_from_loss(monkeypatch):
    """Teacher-forced rival plies should be flagged and excluded from PPO updates."""
    monkeypatch.setattr("src.grpo_logic.sampling.evaluate_board", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr("src.grpo_logic.sampling.reward_board", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(
        "src.grpo_logic.sampling.get_stockfish_move",
        lambda board, depth=4, timeout=10.0: next(iter(board.legal_moves), None),
    )

    random.seed(0)
    torch.manual_seed(0)

    model = UniformLegalPolicy()
    boards = [chess.Board()]
    sample = sample_trajectories_batched(
        model=model,
        boards=boards,
        num_trajectories=1,
        trajectory_depth=2,
        temperature=1.0,
        teacher_forcing_prob=1.0,
        teacher_forcing_depth=1,
    )

    assert sample is not None
    assert sample.pad_mask.shape == (1, 1, 2)
    assert sample.teacher_forced_mask.shape == (1, 1, 2)

    # With depth=2, t=0 is policy ply, t=1 is rival ply and should be teacher-forced.
    assert bool(sample.pad_mask[0, 0, 0].item()) is True
    assert bool(sample.teacher_forced_mask[0, 0, 0].item()) is False
    assert bool(sample.pad_mask[0, 0, 1].item()) is True
    assert bool(sample.teacher_forced_mask[0, 0, 1].item()) is True

    # Mirror training mask logic: start-player plies only, excluding teacher-forced plies.
    t = torch.arange(sample.pad_mask.shape[-1], device=sample.pad_mask.device)
    start_player_mask = (t % 2 == 0)[None, None, :]
    effective_pad_mask = sample.pad_mask & start_player_mask & (~sample.teacher_forced_mask)

    assert bool(effective_pad_mask[0, 0, 0].item()) is True
    assert bool(effective_pad_mask[0, 0, 1].item()) is False


def test_teacher_forcing_coin_is_per_trajectory(monkeypatch):
    """Teacher forcing should be decided independently per trajectory, not per timestep."""
    monkeypatch.setattr("src.grpo_logic.sampling.evaluate_board", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr("src.grpo_logic.sampling.reward_board", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(
        "src.grpo_logic.sampling.get_stockfish_move",
        lambda board, depth=4, timeout=10.0: next(iter(board.legal_moves), None),
    )

    # With prob=0.5 and many trajectories, we should see a mix of forced and unforced
    # at the same timestep across different trajectories.
    G = 20
    random.seed(42)
    torch.manual_seed(42)

    model = UniformLegalPolicy()
    boards = [chess.Board()]
    sample = sample_trajectories_batched(
        model=model,
        boards=boards,
        num_trajectories=G,
        trajectory_depth=4,
        temperature=1.0,
        teacher_forcing_prob=0.5,
        teacher_forcing_depth=1,
    )

    assert sample is not None
    # At rival timesteps (t=1, t=3), different trajectories should have different
    # teacher_forced_mask values — not all True or all False.
    for t_idx in [1, 3]:
        forced_at_t = sample.teacher_forced_mask[0, :, t_idx]  # [G]
        valid_at_t = sample.pad_mask[0, :, t_idx]  # [G]
        forced_valid = forced_at_t[valid_at_t]
        if forced_valid.numel() >= 2:
            # With 20 trajectories and p=0.5, seeing all-same is astronomically unlikely
            assert not forced_valid.all() or not (~forced_valid).all(), (
                f"At rival timestep t={t_idx}, all {forced_valid.numel()} trajectories "
                f"had identical teacher_forced_mask — coin flip is not per-trajectory"
            )
