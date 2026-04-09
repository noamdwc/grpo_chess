import os
from types import SimpleNamespace

import pytest
import torch

from src.chess.policy_player import PolicyConfig
from src.chess.stockfish import StockfishConfig
from src.distill.distill import (
    DistillChessTransformer,
    DistillConfig,
    DistillQualityGateCallback,
    build_stockfish_eval_callback,
    prepare_wandb_config,
)
from src.eval_utils import EvalConfig
from src.models import ChessTransformerConfig


def _make_tiny_batch(action_dim: int = 16, batch_size: int = 2):
    board_tokens = torch.randint(0, 300, (batch_size, 77), dtype=torch.long)
    legal_masks = torch.zeros((batch_size, action_dim), dtype=torch.bool)
    legal_masks[:, 0] = True
    teacher_indices = torch.zeros((batch_size, 1), dtype=torch.long)
    teacher_probs = torch.ones((batch_size, 1), dtype=torch.float32)
    k_mask = torch.ones((batch_size, 1), dtype=torch.bool)
    return board_tokens, legal_masks, teacher_indices, teacher_probs, k_mask


def _tiny_model(distill_cfg: DistillConfig) -> DistillChessTransformer:
    model_cfg = ChessTransformerConfig(
        vocab_size=300,
        embed_dim=32,
        num_layers=1,
        num_heads=4,
        action_dim=16,
    )
    model = DistillChessTransformer(model_cfg, distill_cfg)
    model.log = lambda *args, **kwargs: None  # type: ignore[method-assign]
    return model


def test_prepare_wandb_config_auto_disables_without_key(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_KEY", raising=False)

    cfg = DistillConfig(use_wandb=True, auto_disable_wandb_if_missing_key=True)
    updated = prepare_wandb_config(cfg)

    assert updated.use_wandb is False


def test_prepare_wandb_config_requires_key_when_auto_disable_off(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_KEY", raising=False)

    cfg = DistillConfig(use_wandb=True, auto_disable_wandb_if_missing_key=False)
    with pytest.raises(RuntimeError, match="WANDB_API_KEY not found"):
        prepare_wandb_config(cfg)


def test_prepare_wandb_config_maps_wandb_key(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_KEY", "abc123")

    cfg = DistillConfig(use_wandb=True, auto_disable_wandb_if_missing_key=True)
    updated = prepare_wandb_config(cfg)

    assert updated.use_wandb is True
    assert os.environ["WANDB_API_KEY"] == "abc123"


def test_training_step_skips_nonfinite_loss_when_allowed():
    model = _tiny_model(DistillConfig(fail_on_nonfinite=False, max_nonfinite_batches=3))

    def bad_compute_loss(*_args, **_kwargs):
        metrics = {
            "top1_match": torch.tensor(0.0),
            "top5_match": torch.tensor(0.0),
            "entropy": torch.tensor(0.0),
            "kl_divergence": torch.tensor(0.0),
            "teacher_legal_fraction": torch.tensor(0.0),
            "teacher_valid_sample_fraction": torch.tensor(0.0),
        }
        return torch.tensor(float("inf"), device=next(model.model.parameters()).device), metrics

    model._compute_loss = bad_compute_loss  # type: ignore[method-assign]
    loss = model.training_step(_make_tiny_batch(), batch_idx=0)

    assert torch.isfinite(loss)
    assert model._train_nonfinite_batches == 1


def test_training_step_raises_on_nonfinite_when_configured():
    model = _tiny_model(DistillConfig(fail_on_nonfinite=True, max_nonfinite_batches=3))

    def bad_compute_loss(*_args, **_kwargs):
        metrics = {
            "top1_match": torch.tensor(0.0),
            "top5_match": torch.tensor(0.0),
            "entropy": torch.tensor(0.0),
            "kl_divergence": torch.tensor(0.0),
            "teacher_legal_fraction": torch.tensor(0.0),
            "teacher_valid_sample_fraction": torch.tensor(0.0),
        }
        return torch.tensor(float("inf"), device=next(model.model.parameters()).device), metrics

    model._compute_loss = bad_compute_loss  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="fail_on_nonfinite=true"):
        model.training_step(_make_tiny_batch(), batch_idx=0)


def test_build_stockfish_eval_callback_skips_when_missing_and_optional(monkeypatch):
    monkeypatch.setattr(
        "src.distill.distill.resolve_stockfish_path",
        lambda _path: (_ for _ in ()).throw(FileNotFoundError("missing stockfish")),
    )

    callback = build_stockfish_eval_callback(
        distill_config=DistillConfig(require_stockfish_eval=False),
        eval_cfg=EvalConfig(),
        stockfish_cfg=StockfishConfig(),
        policy_cfg=PolicyConfig(),
    )

    assert callback is None


def test_build_stockfish_eval_callback_raises_when_missing_and_required(monkeypatch):
    monkeypatch.setattr(
        "src.distill.distill.resolve_stockfish_path",
        lambda _path: (_ for _ in ()).throw(FileNotFoundError("missing stockfish")),
    )

    with pytest.raises(RuntimeError, match="Stockfish evaluation is required"):
        build_stockfish_eval_callback(
            distill_config=DistillConfig(require_stockfish_eval=True),
            eval_cfg=EvalConfig(),
            stockfish_cfg=StockfishConfig(),
            policy_cfg=PolicyConfig(),
        )


def test_distill_quality_gate_skips_before_gate_epoch():
    callback = DistillQualityGateCallback(min_val_top1=0.08, gate_epoch=3)
    trainer = SimpleNamespace(
        current_epoch=1,  # Completed epoch 2
        callback_metrics={"val/top1_match": torch.tensor(0.01)},
    )

    callback.on_validation_epoch_end(trainer, pl_module=object())


def test_distill_quality_gate_raises_when_metric_below_threshold():
    callback = DistillQualityGateCallback(min_val_top1=0.08, gate_epoch=3)
    trainer = SimpleNamespace(
        current_epoch=2,  # Completed epoch 3
        callback_metrics={"val/top1_match": torch.tensor(0.03)},
    )

    with pytest.raises(RuntimeError, match="Distill quality gate failed"):
        callback.on_validation_epoch_end(trainer, pl_module=object())


def test_distill_quality_gate_passes_when_metric_meets_threshold():
    callback = DistillQualityGateCallback(min_val_top1=0.08, gate_epoch=3)
    trainer = SimpleNamespace(
        current_epoch=2,  # Completed epoch 3
        callback_metrics={"val/top1_match": torch.tensor(0.08)},
    )

    callback.on_validation_epoch_end(trainer, pl_module=object())


# ---------------------------------------------------------------------------
# Fixed FEN set — same positions as test_distill_label_safety._SAMPLE_FENS.
# Covers starting position, castling, promotion, en passant, endgames.
# ---------------------------------------------------------------------------
_SAMPLE_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",          # starting position
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",  # Italian Game
    "8/k1P5/8/8/8/8/8/7K w - - 0 1",                                       # pawn promotion
    "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",       # en passant
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",                                # castling both sides
    "4k3/8/8/8/8/8/8/4K3 w - - 0 1",                                        # bare kings
    "r2qkb1r/pp1bpppp/2n2n2/3p4/3P4/2N2N2/PP2PPPP/R1BQKB1R b KQkq - 3 6",  # middlegame
    "8/5k2/3p4/1p1Pp2p/pP2Pp1P/P4P1K/8/8 b - - 0 41",                     # pawn endgame
]


def test_masked_logits_do_not_affect_loss() -> None:
    """Perturbing an illegal-move logit by +1e6 must leave the loss unchanged.
    Catches masking bugs where the -inf fill is incomplete, the gather is applied
    before masking, or softmax normalisation leaks masked probability mass.
    """
    A, B, K = 16, 4, 3
    model = _tiny_model(DistillConfig())

    torch.manual_seed(99)
    logits = torch.randn(B, A)
    legal_masks = torch.zeros(B, A, dtype=torch.bool)
    legal_masks[:, :5] = True  # only actions 0-4 are legal; 5-15 are masked

    teacher_indices = torch.randint(0, 5, (B, K))
    teacher_probs = torch.softmax(torch.randn(B, K), dim=-1)
    k_mask = torch.ones(B, K, dtype=torch.bool)

    loss_baseline, _ = model._compute_loss(logits, legal_masks, teacher_indices, teacher_probs, k_mask)

    # Perturb a logit that corresponds to a masked (illegal) action.
    perturbed = logits.clone()
    perturbed[:, 10] += 1e6  # action 10 is illegal for all samples

    loss_perturbed, _ = model._compute_loss(perturbed, legal_masks, teacher_indices, teacher_probs, k_mask)

    assert abs(loss_baseline.item() - loss_perturbed.item()) < 1e-6, (
        f"Masked logit perturbation changed loss: "
        f"{loss_baseline.item():.6f} -> {loss_perturbed.item():.6f}"
    )


def test_eval_policy_produces_only_legal_moves() -> None:
    """Greedy PolicyPlayer must select a legal move for every board position.
    Catches bugs in action_to_move mapping or legal-mask construction that would
    produce illegal moves at eval time while training loss looks normal.
    """
    import chess
    from src.chess.policy_player import PolicyPlayer
    from src.models import ChessTransformer

    model_cfg = ChessTransformerConfig(
        vocab_size=300, embed_dim=32, num_layers=1, num_heads=4, action_dim=1968
    )
    torch.manual_seed(0)
    inner_model = ChessTransformer(model_cfg)
    player = PolicyPlayer(inner_model, device="cpu", cfg=PolicyConfig(greedy=True))

    for fen in _SAMPLE_FENS:
        board = chess.Board(fen)
        legal_moves = set(board.legal_moves)
        if not legal_moves:
            continue  # skip terminal positions

        move = player.act(board)
        assert move in legal_moves, (
            f"FEN={fen!r}: PolicyPlayer returned illegal move {move.uci()!r}"
        )


def test_tiny_overfit_loss_decreases() -> None:
    """Training on a fixed tiny batch must reduce loss substantially.
    Catches wrong gather/indexing, dead gradients, or loss-function bugs where
    the loss appears finite but does not propagate useful signal to the model.
    """
    import torch.optim as optim

    A, B = 16, 8
    cfg = ChessTransformerConfig(vocab_size=300, embed_dim=32, num_layers=1, num_heads=4, action_dim=A)
    model = DistillChessTransformer(cfg, DistillConfig(distill_lambda_soft=1.0, distill_lambda_hard=0.0))
    model.log = lambda *a, **kw: None  # type: ignore[method-assign]

    torch.manual_seed(123)
    board_tokens = torch.randint(0, 300, (B, 77))
    legal_masks = torch.zeros(B, A, dtype=torch.bool)
    legal_masks[:, :4] = True  # 4 legal actions per sample
    # Fixed teacher target: always action 0 with probability 1.0.
    teacher_indices = torch.zeros(B, 1, dtype=torch.long)
    teacher_probs = torch.ones(B, 1, dtype=torch.float32)
    k_mask = torch.ones(B, 1, dtype=torch.bool)

    optimizer = optim.Adam(model.model.parameters(), lr=5e-3)

    with torch.no_grad():
        logits0 = model.model(board_tokens)
        loss0, _ = model._compute_loss(logits0, legal_masks, teacher_indices, teacher_probs, k_mask)
    loss0_val = loss0.item()

    for _ in range(150):
        optimizer.zero_grad()
        logits = model.model(board_tokens)
        loss, _ = model._compute_loss(logits, legal_masks, teacher_indices, teacher_probs, k_mask)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits_final = model.model(board_tokens)
        loss_final, _ = model._compute_loss(
            logits_final, legal_masks, teacher_indices, teacher_probs, k_mask
        )

    assert loss_final.item() < 0.7 * loss0_val, (
        f"Loss did not decrease sufficiently after 150 steps: "
        f"initial={loss0_val:.4f}, final={loss_final.item():.4f}"
    )
