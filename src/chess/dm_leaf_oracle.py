"""Frozen DM 136M value oracle for leaf evaluation."""

import chess

from src.checkpoint_inspector import inspect_checkpoint

_ENGINE = None
_BUCKET_VALUES = None
_CHECKPOINT_DIR = "searchless_chess/checkpoints"
_MODEL_NAME = "136M"


def _ensure_loaded() -> None:
    global _ENGINE, _BUCKET_VALUES
    if _ENGINE is not None:
        return
    from src.distill.teacher import build_teacher_engine

    checkpoint_path = f"{_CHECKPOINT_DIR}/{_MODEL_NAME}"
    checkpoint_info = inspect_checkpoint(checkpoint_path)
    if checkpoint_info.family != "dm_action_value":
        raise ValueError(
            f"Expected {_MODEL_NAME} oracle checkpoint to have family dm_action_value, "
            f"got {checkpoint_info.family} from {checkpoint_info.path}"
        )

    _ENGINE, _BUCKET_VALUES = build_teacher_engine(
        model_name=_MODEL_NAME,
        checkpoint_dir=_CHECKPOINT_DIR,
        checkpoint_step=6_400_000,
        batch_size=1,
        use_half=True,
    )


def dm_leaf_value(fen: str, pov_is_white: bool) -> float:
    """Return DM 136M's expected state value in [-1, 1], POV-normalized."""
    _ensure_loaded()

    import numpy as np

    board = chess.Board(fen)
    if board.is_game_over(claim_draw=True):
        if board.is_checkmate():
            pov_loses = board.turn == (chess.WHITE if pov_is_white else chess.BLACK)
            return -1.0 if pov_loses else 1.0
        return 0.0

    analysis = _ENGINE.analyse(board)
    action_log_probs = analysis["log_probs"]
    expected_per_action = (np.exp(action_log_probs) * _BUCKET_VALUES[None, :]).sum(axis=-1)
    value_side_to_move = float(expected_per_action.max()) * 2.0 - 1.0
    if pov_is_white != (board.turn == chess.WHITE):
        value_side_to_move = -value_side_to_move
    return max(-1.0, min(1.0, value_side_to_move))
