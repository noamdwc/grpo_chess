import torch
import pytest

from src.models import ChessTransformerConfig
from src.searchless_chess_imports import MOVE_TO_ACTION
from src.distill.convert_deepmind_data import make_grouped_sample
from src.distill.distill import DistillChessTransformer, DistillConfig


def test_make_grouped_sample_filters_illegal_moves() -> None:
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    moves_and_probs = [
        ("e2e4", 0.60),  # legal
        ("d2d4", 0.59),  # legal
        ("e7e5", 0.95),  # illegal for side-to-move in this FEN
    ]

    sample = make_grouped_sample(fen, moves_and_probs, top_k=3, temperature=1.0)
    assert sample is not None

    illegal_idx = MOVE_TO_ACTION["e7e5"]
    legal_indices = set(sample["teacher_action_indices"].tolist())
    assert illegal_idx not in legal_indices
    assert sample["teacher_probs"].sum().item() == pytest.approx(1.0, rel=1e-6)


def test_distill_loss_ignores_illegal_teacher_targets() -> None:
    cfg = ChessTransformerConfig(vocab_size=300, embed_dim=32, num_layers=1, num_heads=4, action_dim=16)
    model = DistillChessTransformer(cfg, DistillConfig())

    logits = torch.zeros(2, 16)
    legal_masks = torch.zeros(2, 16, dtype=torch.bool)
    legal_masks[0, [1, 2, 3]] = True
    legal_masks[1, [5, 6, 7]] = True

    teacher_indices = torch.tensor(
        [
            [1, 9],  # one legal (1), one illegal (9)
            [12, 13],  # both illegal
        ],
        dtype=torch.long,
    )
    teacher_probs = torch.tensor([[0.7, 0.3], [0.5, 0.5]], dtype=torch.float32)
    k_mask = torch.ones(2, 2, dtype=torch.bool)

    loss, metrics = model._compute_loss(
        logits=logits,
        legal_masks=legal_masks,
        teacher_indices=teacher_indices,
        teacher_probs=teacher_probs,
        k_mask=k_mask,
    )

    assert torch.isfinite(loss)
    assert metrics["teacher_legal_fraction"].item() == pytest.approx(0.25, rel=1e-6)
    assert metrics["teacher_valid_sample_fraction"].item() == pytest.approx(0.5, rel=1e-6)
