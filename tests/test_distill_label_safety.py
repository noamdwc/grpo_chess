import torch
import pytest
import numpy as np

from src.models import ChessTransformerConfig
from src.searchless_chess_imports import MOVE_TO_ACTION
from src.distill.convert_deepmind_data import make_grouped_sample
from src.distill.distill import DistillChessTransformer, DistillConfig
from src.distill.teacher import postprocess_teacher_output


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


def test_distill_hybrid_loss_combines_soft_and_hard_terms() -> None:
    cfg = ChessTransformerConfig(vocab_size=300, embed_dim=32, num_layers=1, num_heads=4, action_dim=8)
    model = DistillChessTransformer(
        cfg,
        DistillConfig(
            distill_lambda_soft=0.3,
            distill_lambda_hard=0.7,
            distill_target_top1_mix_alpha=0.0,
        ),
    )

    logits = torch.tensor([[2.0, 1.0, 0.5, 0.0, -10.0, -10.0, -10.0, -10.0]], dtype=torch.float32)
    legal_masks = torch.zeros(1, 8, dtype=torch.bool)
    legal_masks[0, [0, 1, 2, 3]] = True

    teacher_indices = torch.tensor([[0, 1, 2]], dtype=torch.long)
    teacher_probs = torch.tensor([[0.34, 0.33, 0.33]], dtype=torch.float32)
    k_mask = torch.ones(1, 3, dtype=torch.bool)

    loss, metrics = model._compute_loss(logits, legal_masks, teacher_indices, teacher_probs, k_mask)

    expected_total = 0.3 * metrics["loss_soft"] + 0.7 * metrics["loss_hard"]
    assert loss.item() == pytest.approx(expected_total.item(), rel=1e-6)
    assert metrics["loss_total"].item() == pytest.approx(expected_total.item(), rel=1e-6)
    assert metrics["loss_hard"].item() != pytest.approx(metrics["loss_soft"].item(), rel=1e-6)


def test_top1_mix_alpha_one_matches_hard_target_loss() -> None:
    cfg = ChessTransformerConfig(vocab_size=300, embed_dim=32, num_layers=1, num_heads=4, action_dim=8)
    model = DistillChessTransformer(
        cfg,
        DistillConfig(
            distill_lambda_soft=1.0,
            distill_lambda_hard=0.0,
            distill_target_top1_mix_alpha=1.0,
        ),
    )

    logits = torch.tensor([[1.2, 0.9, 0.1, -0.2, -10.0, -10.0, -10.0, -10.0]], dtype=torch.float32)
    legal_masks = torch.zeros(1, 8, dtype=torch.bool)
    legal_masks[0, [0, 1, 2, 3]] = True

    teacher_indices = torch.tensor([[1, 0, 2]], dtype=torch.long)
    teacher_probs = torch.tensor([[0.34, 0.33, 0.33]], dtype=torch.float32)
    k_mask = torch.ones(1, 3, dtype=torch.bool)

    _, metrics = model._compute_loss(logits, legal_masks, teacher_indices, teacher_probs, k_mask)
    assert metrics["loss_soft"].item() == pytest.approx(metrics["loss_hard"].item(), rel=1e-6)


def test_teacher_zscore_score_norm_sharpens_flat_topk_targets() -> None:
    # Scores are nearly tied: plain softmax is ~uniform, z-score should sharpen.
    near_tied_scores = [0.5000, 0.5001, 0.5002, 0.5003]
    log_probs = np.log(np.array([[1.0 - s, s] for s in near_tied_scores], dtype=np.float64))
    bucket_values = np.array([0.0, 1.0], dtype=np.float64)
    action_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    plain_actions, plain_probs = postprocess_teacher_output(
        log_probs=log_probs,
        bucket_values=bucket_values,
        action_indices=action_indices,
        top_k=4,
        temperature=1.0,
        score_norm_mode="plain",
    )
    zscore_actions, zscore_probs = postprocess_teacher_output(
        log_probs=log_probs,
        bucket_values=bucket_values,
        action_indices=action_indices,
        top_k=4,
        temperature=1.0,
        score_norm_mode="zscore",
    )

    assert int(plain_actions[0].item()) == 3
    assert int(zscore_actions[0].item()) == 3
    assert plain_probs[0].item() < 0.30
    assert zscore_probs[0].item() > plain_probs[0].item()
    assert zscore_probs[0].item() > 0.40
