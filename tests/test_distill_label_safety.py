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


# ---------------------------------------------------------------------------
# Fixed FEN set covering starting position, castling, promotion, en passant,
# and endgames. Deterministic; no downloads required. Also used in
# test_distill_cloud_reliability.py — keep the two lists in sync.
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


def _sample_fens() -> list[str]:
    return _SAMPLE_FENS


def test_mapping_consistency_teacher_equals_eval() -> None:
    """Single canonical MOVE_TO_ACTION must be shared by teacher pipeline, legal mask
    builder, and eval-time policy decoder. A divergence would corrupt training data
    silently: the student learns the wrong labels for given board tokens.
    """
    from src.searchless_chess_imports import MOVE_TO_ACTION as canonical
    from src.searchless_chess_imports import ACTION_TO_MOVE as canonical_a2m
    from src.chess.chess_logic import MOVE_TO_ACTION as eval_m2a
    from src.chess.chess_logic import ACTION_TO_MOVE as eval_a2m
    from src.distill.teacher import ACTION_SPACE_SIZE
    from src.chess.chess_logic import MAX_ACTION

    # Teacher pipeline and eval decode path must use the exact same mapping.
    assert canonical == eval_m2a, (
        f"MOVE_TO_ACTION differs: canonical has {len(canonical)} entries, eval has {len(eval_m2a)}"
    )
    assert canonical_a2m == eval_a2m, "ACTION_TO_MOVE differs between canonical and eval path"

    # Action space dimensions must agree across both modules.
    assert ACTION_SPACE_SIZE == MAX_ACTION + 1, (
        f"teacher ACTION_SPACE_SIZE={ACTION_SPACE_SIZE} != chess_logic MAX_ACTION+1={MAX_ACTION + 1}"
    )

    # Mapping must be dense: values 0..N-1 are all assigned.
    action_ids = set(canonical.values())
    assert action_ids == set(range(ACTION_SPACE_SIZE)), (
        "MOVE_TO_ACTION values are not contiguous [0, ACTION_SPACE_SIZE): missing IDs detected"
    )


def test_legal_mask_covers_all_legal_moves() -> None:
    """build_legal_mask_from_fen must contain True for every legal move (no false
    negatives) and True only for legal moves (no false positives). Validates that
    the mask builder uses the canonical MOVE_TO_ACTION and that every legal move
    reachable by python-chess is representable in the action vocabulary.
    """
    import chess
    from src.chess.chess_logic import build_legal_mask_from_fen
    from src.searchless_chess_imports import MOVE_TO_ACTION, ACTION_TO_MOVE

    for fen in _SAMPLE_FENS:
        board = chess.Board(fen)
        legal_uci = {m.uci() for m in board.legal_moves}
        mask = build_legal_mask_from_fen(fen)  # bool tensor [1968]

        # All legal moves must be representable in MOVE_TO_ACTION (no gaps).
        missing_from_vocab = [u for u in legal_uci if u not in MOVE_TO_ACTION]
        assert not missing_from_vocab, (
            f"FEN={fen!r}: {len(missing_from_vocab)} legal moves absent from MOVE_TO_ACTION: "
            f"{missing_from_vocab}"
        )

        # No false negatives: every legal move must set its action index to True.
        for uci in legal_uci:
            idx = MOVE_TO_ACTION[uci]
            assert mask[idx].item(), (
                f"FEN={fen!r}: legal move {uci!r} (action {idx}) is missing from the mask"
            )

        # No false positives: every True position must correspond to a legal move.
        for idx in mask.nonzero(as_tuple=True)[0].tolist():
            uci = ACTION_TO_MOVE[idx]
            assert uci in legal_uci, (
                f"FEN={fen!r}: mask[{idx}]=True but {uci!r} is not a legal move"
            )


def test_collated_teacher_targets_are_legal() -> None:
    """After collation every valid (k_mask=True) teacher index must point to a True
    entry in its sample's legal mask. Catches data-generation bugs where the teacher
    records a move that is illegal for the given board position; such entries would
    be silently zeroed by the loss but could indicate a systematic labelling error.
    """
    from src.distill.distill_dataset import collate_distill_batch

    A = 1968  # full action space
    K = 4
    N = 6
    torch.manual_seed(7)

    batch = []
    for i in range(N):
        board_tokens = torch.zeros(77, dtype=torch.long)
        legal_mask = torch.zeros(A, dtype=torch.bool)
        # Give each sample a disjoint set of 8 legal actions to keep the test clean.
        legal_acts = list(range(i * 8, i * 8 + 8))
        legal_mask[legal_acts] = True

        # Teacher targets: point to the first K actions in the legal set.
        teacher_idx = torch.tensor(legal_acts[:K], dtype=torch.long)
        teacher_prob = torch.full((K,), 1.0 / K, dtype=torch.float32)
        k_mask_s = torch.ones(K, dtype=torch.bool)
        batch.append((board_tokens, legal_mask, teacher_idx, teacher_prob, k_mask_s))

    board_tokens, legal_masks, teacher_indices, teacher_probs, k_masks = collate_distill_batch(batch)

    # Every valid teacher index must be in the legal mask for that sample.
    is_legal = legal_masks.gather(1, teacher_indices)  # [N, K] bool
    valid_is_legal = is_legal[k_masks]
    assert valid_is_legal.all(), (
        f"Some valid teacher targets point to illegal actions: "
        f"{valid_is_legal.float().mean():.3f} legal fraction (expected 1.0)"
    )


def test_side_to_move_alignment_action_round_trip() -> None:
    """Every True entry in a legal mask must correspond to a move whose from-square
    carries a piece whose color matches the side-to-move in the FEN.

    A ply-shift bug (board encoded one move early/late) would make the mask contain
    moves for the wrong side; this test catches that without requiring a full game.
    """
    import chess
    from src.chess.chess_logic import build_legal_mask_from_fen
    from src.searchless_chess_imports import ACTION_TO_MOVE

    for fen in _sample_fens():
        board = chess.Board(fen)
        if not list(board.legal_moves):
            continue  # skip terminal positions

        mask = build_legal_mask_from_fen(fen)
        true_indices = mask.nonzero(as_tuple=True)[0].tolist()

        for idx in true_indices:
            uci = ACTION_TO_MOVE[idx]
            move = chess.Move.from_uci(uci)
            piece = board.piece_at(move.from_square)
            assert piece is not None, (
                f"FEN={fen!r}: action {idx} ({uci!r}) from-square {move.from_square} is empty"
            )
            assert piece.color == board.turn, (
                f"FEN={fen!r}: action {idx} ({uci!r}) moves a "
                f"{'white' if piece.color else 'black'} piece but it is "
                f"{'white' if board.turn else 'black'}'s turn — ply-shift detected"
            )
