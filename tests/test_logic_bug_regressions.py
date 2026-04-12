"""Regression tests for documented logic bugs and logic errors."""

from types import SimpleNamespace

import chess
import pytest
import torch

from src.grpo_logic.reasoning_loss import ReasoningLossConfig, reasoning_loss
from src.reasoning.sampler import RolloutResult
from src.searchless_chess_imports import MOVE_TO_ACTION


def test_reasoning_loss_only_credits_final_move_position():
    result = RolloutResult(
        token_sequences=torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.long),
        seq_lens=torch.tensor([4, 4], dtype=torch.long),
        log_probs_old=torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        sampled_mask=torch.tensor([[False, True, True, True], [False, True, True, True]], dtype=torch.bool),
        legal_token_mask=torch.ones((2, 4, 1971), dtype=torch.bool),
        end_think_positions=torch.tensor([2, 2], dtype=torch.long),
        final_move_positions=torch.tensor([3, 3], dtype=torch.long),
        imagined_leaf_fens=[chess.STARTING_FEN, chess.STARTING_FEN],
        root_fens_replayed=[chess.STARTING_FEN, chess.STARTING_FEN],
        think_tokens=[[10, 11], [20, 21]],
    )

    log_probs_new = torch.tensor([[0.0, 5.0, 5.0, 2.0], [0.0, -5.0, -5.0, -2.0]], dtype=torch.float32)
    out = reasoning_loss(
        result=result,
        log_probs_new=log_probs_new,
        v_hat=torch.tensor([0.0, 0.0], dtype=torch.float32),
        rewards=torch.tensor([1.0, -1.0], dtype=torch.float32),
        v_target=torch.tensor([0.0, 0.0], dtype=torch.float32),
        cfg=ReasoningLossConfig(clip_ratio=10.0, kl_coef=0.0, value_coef=0.0),
        group_size=2,
    )

    # Only the final move should contribute to PPO. If earlier think tokens receive the
    # same sample-level advantage, the loss would average across all three sampled positions.
    expected = (-(torch.exp(torch.tensor(2.0)) / (2 ** 0.5)) + (torch.exp(torch.tensor(-2.0)) / (2 ** 0.5))) / 2
    assert out.ppo.item() == pytest.approx(expected.item(), rel=1e-5)


def test_play_stockfish_match_counts_missing_stockfish_reply_as_model_win(monkeypatch):
    import src.evaluator as evaluator

    class DummyStockfish:
        def __init__(self, cfg, engine_name):
            del cfg, engine_name

        def act(self, board):
            del board
            return None

        def close(self):
            return None

    rollout = RolloutResult(
        token_sequences=torch.tensor([[1]], dtype=torch.long),
        seq_lens=torch.tensor([1], dtype=torch.long),
        log_probs_old=torch.zeros((1, 1), dtype=torch.float32),
        sampled_mask=torch.tensor([[True]], dtype=torch.bool),
        legal_token_mask=torch.ones((1, 1, 1971), dtype=torch.bool),
        end_think_positions=torch.tensor([0], dtype=torch.long),
        final_move_positions=torch.tensor([0], dtype=torch.long),
        imagined_leaf_fens=[chess.STARTING_FEN],
        root_fens_replayed=[chess.STARTING_FEN],
        think_tokens=[[]],
    )

    rollout.final_move_positions[0] = 0
    rollout.token_sequences[0, 0] = MOVE_TO_ACTION["e2e4"]

    monkeypatch.setattr(evaluator, "StockfishPlayer", DummyStockfish)
    monkeypatch.setattr("src.reasoning.sampler.rollout_batch", lambda **kwargs: rollout)

    score = evaluator._play_stockfish_match(
        model=object(),
        n_games=1,
        stockfish_level=0,
        seed=0,
        thinking=False,
    )

    assert score == pytest.approx(1.0)


def test_generate_endgame_position_returns_valid_nonterminal_endgame():
    from src.chess.boards_dataset import generate_endgame_position

    # Run 200 trials — the old buggy implementation (capture-biased random play)
    # produces game-over or high-material boards ~3.5% of the time, so a single
    # call would pass 96.5% of runs.  With 200 trials the false-pass rate drops
    # to 0.965^200 ≈ 0.08%.
    for _ in range(200):
        board = generate_endgame_position()
        non_king_material = sum(
            len(board.pieces(piece_type, color))
            for piece_type in [chess.PAWN, chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN]
            for color in [chess.WHITE, chess.BLACK]
        )

        assert board.is_valid()
        assert board.king(chess.WHITE) is not None
        assert board.king(chess.BLACK) is not None
        assert not board.is_game_over(claim_draw=True)
        assert non_king_material <= 12


def test_make_grouped_sample_rejects_singleton_sparse_label_group():
    from src.distill.convert_deepmind_data import make_grouped_sample

    sample = make_grouped_sample(
        chess.STARTING_FEN,
        [("e2e4", 0.9)],
        top_k=8,
        temperature=1.0,
    )

    assert sample is None


def test_make_grouped_sample_filters_illegal_moves():
    from src.distill.convert_deepmind_data import make_grouped_sample

    sample = make_grouped_sample(
        chess.STARTING_FEN,
        [("e2e5", 0.9), ("e2e4", 0.8), ("d2d4", 0.7)],
        top_k=8,
        temperature=1.0,
    )

    assert sample is not None
    assert set(sample["teacher_action_indices"].tolist()) == {
        MOVE_TO_ACTION["e2e4"],
        MOVE_TO_ACTION["d2d4"],
    }


def test_distill_loss_renormalizes_legal_teacher_mass():
    from src.distill.distill import DistillChessTransformer, DistillConfig

    module = DistillChessTransformer.__new__(DistillChessTransformer)
    module.distill_config = DistillConfig()
    module._module_device = lambda: torch.device("cpu")

    logits = torch.tensor([[2.0, 1.0, -1.0, -2.0, -3.0, -4.0]], dtype=torch.float32)
    legal_masks = torch.tensor([[True, False, True, False, False, False]], dtype=torch.bool)
    teacher_indices = torch.tensor([[0, 1]], dtype=torch.long)
    teacher_probs = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    k_mask = torch.tensor([[True, True]], dtype=torch.bool)

    loss, metrics = DistillChessTransformer._compute_loss(
        module,
        logits,
        legal_masks,
        teacher_indices,
        teacher_probs,
        k_mask,
    )

    expected = -torch.log_softmax(logits.masked_fill(~legal_masks, float("-inf")), dim=-1)[0, 0]
    assert loss.item() == pytest.approx(expected.item())
    assert metrics["teacher_legal_fraction"] == pytest.approx(0.5)


def test_training_step_passes_configured_rollout_temperature(monkeypatch):
    from src.grpo_logic.model import ReasoningGRPOLightningModule
    from src.grpo_logic.reasoning_loss import ReasoningLossOutput

    calls = {}

    rollout = RolloutResult(
        token_sequences=torch.tensor([[1, 2, 3]], dtype=torch.long),
        seq_lens=torch.tensor([3], dtype=torch.long),
        log_probs_old=torch.zeros((1, 3), dtype=torch.float32),
        sampled_mask=torch.tensor([[False, True, True]], dtype=torch.bool),
        legal_token_mask=torch.ones((1, 3, 1971), dtype=torch.bool),
        end_think_positions=torch.tensor([1], dtype=torch.long),
        final_move_positions=torch.tensor([2], dtype=torch.long),
        imagined_leaf_fens=[chess.STARTING_FEN],
        root_fens_replayed=[chess.STARTING_FEN],
        think_tokens=[[1]],
    )

    class DummyPolicy:
        def __call__(self, tokens, seq_lens):
            del tokens, seq_lens
            return SimpleNamespace(hidden=torch.ones((1, 3, 4), dtype=torch.float32))

        def value_at_end_think(self, hidden, end_think_positions):
            del hidden, end_think_positions
            return torch.tensor([0.0], dtype=torch.float32)

    fake = SimpleNamespace(
        cfg=SimpleNamespace(
            grpo=SimpleNamespace(k_samples_per_root=1, rollout_temperature=1.7, move_sampling_temperature=1.0),
            reasoning=SimpleNamespace(min_think_tokens=1, max_think_tokens=2),
            rival=SimpleNamespace(mode="self_play"),
            leaf_evaluator=SimpleNamespace(mode="stockfish"),
        ),
        device=torch.device("cpu"),
        loss_cfg=ReasoningLossConfig(),
        old_policy_model=object(),
        policy_model=DummyPolicy(),
        _post_fen_after_rival=lambda **kwargs: (kwargs["root_fen"], False),
        _recompute_log_probs=lambda *args, **kwargs: torch.zeros((1, 3), dtype=torch.float32),
        log_dict=lambda *args, **kwargs: None,
    )

    def fake_rollout_batch(**kwargs):
        calls["rollout_temperature"] = kwargs["rollout_temperature"]
        return rollout

    monkeypatch.setattr("src.grpo_logic.model.rollout_batch", fake_rollout_batch)
    monkeypatch.setattr("src.grpo_logic.model.evaluate_leaf", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(
        "src.grpo_logic.model.reasoning_loss",
        lambda **kwargs: ReasoningLossOutput(
            total=torch.tensor(0.0),
            ppo=torch.tensor(0.0),
            kl=torch.tensor(0.0),
            value=torch.tensor(0.0),
            clip_fraction=torch.tensor(0.0),
            ratio_mean=torch.tensor(1.0),
            advantage_mean=torch.tensor(0.0),
            advantage_std=torch.tensor(1.0),
        ),
    )

    ReasoningGRPOLightningModule.training_step(fake, [chess.STARTING_FEN], batch_idx=0)

    assert calls["rollout_temperature"] == pytest.approx(1.7)
