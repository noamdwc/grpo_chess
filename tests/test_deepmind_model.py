"""Tests for downloading, loading, and using the DeepMind searchless chess model.

Unit tests use mocks and work without JAX/Haiku.
Integration tests require JAX, Haiku, orbax-checkpoint, and the 270M checkpoint.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import chess
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.searchless_chess_imports import MOVE_TO_ACTION
from src.distill.generate_dataset import GenerateConfig, save_shard, count_existing_shards
from src.distill.teacher import (
    process_position,
    ACTION_SPACE_SIZE,
    FENPrepDataset,
    collate_positions,
    postprocess_teacher_batch,
    postprocess_teacher_output,
)
from src.distill.distill_dataset import DistillDataset, collate_distill_batch

# Check availability of JAX + Haiku for integration tests
try:
    import jax
    import haiku as hk

    _HAS_JAX_HAIKU = True
except ImportError:
    _HAS_JAX_HAIKU = False

requires_jax_haiku = pytest.mark.skipif(
    not _HAS_JAX_HAIKU, reason="JAX and/or Haiku not installed"
)

_CKPT_DIR = Path(__file__).resolve().parent.parent / "searchless_chess" / "checkpoints"
_270M_CKPT_DIR = _CKPT_DIR / "270M"

requires_270m_checkpoint = pytest.mark.skipif(
    not _270M_CKPT_DIR.exists(), reason="270M checkpoint not downloaded"
)

# A standard opening position after 1. e4 e5
_TEST_FEN = "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
_STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_engine(num_return_buckets: int = 128):
    """Create a mock engine that returns plausible log_probs."""
    engine = MagicMock()

    def mock_analyse(board: chess.Board):
        sorted_legal = sorted(board.legal_moves, key=lambda x: MOVE_TO_ACTION[x.uci()])
        n_legal = len(sorted_legal)
        # Random log-probs: [n_legal, num_return_buckets]
        raw = np.random.randn(n_legal, num_return_buckets).astype(np.float32)
        log_probs = raw - np.logaddexp.reduce(raw, axis=-1, keepdims=True)
        return {"log_probs": log_probs, "fen": board.fen()}

    engine.analyse = mock_analyse
    return engine


def _make_bucket_values(num_return_buckets: int = 128):
    """Reproduce get_uniform_buckets_edges_values for bucket values."""
    full = np.linspace(0.0, 1.0, num_return_buckets + 1)
    return (full[:-1] + full[1:]) / 2


# ---------------------------------------------------------------------------
# Unit tests (no JAX/Haiku needed)
# ---------------------------------------------------------------------------


class TestProcessPosition:
    """Test process_position with a mock engine."""

    def test_returns_valid_dict(self):
        engine = _make_mock_engine()
        bucket_values = _make_bucket_values()
        result = process_position(engine, bucket_values, _STARTING_FEN, top_k=8, temperature=1.0)

        assert result is not None
        assert "board_tokens" in result
        assert "legal_mask" in result
        assert "teacher_action_indices" in result
        assert "teacher_probs" in result

    def test_board_tokens_shape(self):
        engine = _make_mock_engine()
        bucket_values = _make_bucket_values()
        result = process_position(engine, bucket_values, _STARTING_FEN, top_k=8, temperature=1.0)

        # 77 tokens for a FEN
        assert result["board_tokens"].shape == (77,)
        assert result["board_tokens"].dtype == torch.long

    def test_legal_mask_shape_and_dtype(self):
        engine = _make_mock_engine()
        bucket_values = _make_bucket_values()
        result = process_position(engine, bucket_values, _STARTING_FEN, top_k=8, temperature=1.0)

        assert result["legal_mask"].shape == (ACTION_SPACE_SIZE,)
        assert result["legal_mask"].dtype == torch.bool

    def test_legal_mask_has_correct_count(self):
        engine = _make_mock_engine()
        bucket_values = _make_bucket_values()
        result = process_position(engine, bucket_values, _STARTING_FEN, top_k=8, temperature=1.0)

        board = chess.Board(_STARTING_FEN)
        expected_legal = sum(1 for m in board.legal_moves if m.uci() in MOVE_TO_ACTION)
        assert result["legal_mask"].sum().item() == expected_legal

    def test_top_k_limits_actions(self):
        engine = _make_mock_engine()
        bucket_values = _make_bucket_values()
        for k in [1, 3, 5, 8]:
            result = process_position(engine, bucket_values, _STARTING_FEN, top_k=k, temperature=1.0)
            assert len(result["teacher_action_indices"]) <= k
            assert len(result["teacher_probs"]) == len(result["teacher_action_indices"])

    def test_teacher_probs_sum_to_one(self):
        engine = _make_mock_engine()
        bucket_values = _make_bucket_values()
        result = process_position(engine, bucket_values, _STARTING_FEN, top_k=8, temperature=1.0)

        total = result["teacher_probs"].sum().item()
        assert abs(total - 1.0) < 1e-5, f"Probs sum to {total}, expected ~1.0"

    def test_teacher_action_indices_are_legal(self):
        engine = _make_mock_engine()
        bucket_values = _make_bucket_values()
        result = process_position(engine, bucket_values, _STARTING_FEN, top_k=8, temperature=1.0)

        legal_mask = result["legal_mask"]
        for idx in result["teacher_action_indices"]:
            assert legal_mask[idx.item()], f"Action index {idx.item()} is not legal"

    def test_invalid_fen_returns_none(self):
        engine = _make_mock_engine()
        bucket_values = _make_bucket_values()
        result = process_position(engine, bucket_values, "not a valid fen", top_k=8, temperature=1.0)
        assert result is None

    def test_checkmate_position_returns_none(self):
        """A position with <= 1 legal move should return None."""
        engine = _make_mock_engine()
        bucket_values = _make_bucket_values()
        # Fool's mate position — black is checkmated (0 legal moves)
        checkmate_fen = "rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        result = process_position(engine, bucket_values, checkmate_fen, top_k=8, temperature=1.0)
        assert result is None

    def test_temperature_affects_distribution(self):
        engine = _make_mock_engine()
        bucket_values = _make_bucket_values()
        result_t1 = process_position(engine, bucket_values, _STARTING_FEN, top_k=8, temperature=1.0)
        result_t01 = process_position(engine, bucket_values, _STARTING_FEN, top_k=8, temperature=0.1)

        # Lower temperature should produce a more peaked distribution
        max_prob_t1 = result_t1["teacher_probs"].max().item()
        max_prob_t01 = result_t01["teacher_probs"].max().item()
        assert max_prob_t01 >= max_prob_t1, "Lower temperature should produce higher max prob"


class TestSaveAndLoadShard:
    """Test shard save/load round-trip."""

    def test_save_load_roundtrip(self, tmp_path):
        engine = _make_mock_engine()
        bucket_values = _make_bucket_values()

        samples = []
        fens = [_STARTING_FEN, _TEST_FEN]
        for fen in fens:
            s = process_position(engine, bucket_values, fen, top_k=5, temperature=1.0)
            if s is not None:
                samples.append(s)

        assert len(samples) >= 1

        shard_path = str(tmp_path / "shard_0000.pt")
        save_shard(samples, shard_path)

        # Load it back
        loaded = torch.load(shard_path, weights_only=False)
        assert loaded["board_tokens"].shape[0] == len(samples)
        assert loaded["board_tokens"].shape[1] == 77
        assert loaded["legal_mask"].shape == (len(samples), ACTION_SPACE_SIZE)
        assert len(loaded["teacher_action_indices"]) == len(samples)
        assert len(loaded["teacher_probs"]) == len(samples)

    def test_shard_tensors_match_originals(self, tmp_path):
        engine = _make_mock_engine()
        bucket_values = _make_bucket_values()
        sample = process_position(engine, bucket_values, _STARTING_FEN, top_k=4, temperature=1.0)
        assert sample is not None

        shard_path = str(tmp_path / "shard_0000.pt")
        save_shard([sample], shard_path)
        loaded = torch.load(shard_path, weights_only=False)

        assert torch.equal(loaded["board_tokens"][0], sample["board_tokens"])
        assert torch.equal(loaded["legal_mask"][0], sample["legal_mask"])
        assert torch.equal(loaded["teacher_action_indices"][0], sample["teacher_action_indices"])
        assert torch.allclose(loaded["teacher_probs"][0], sample["teacher_probs"])


# ---------------------------------------------------------------------------
# Batched pipeline tests (FENPrepDataset → collate → postprocess)
# ---------------------------------------------------------------------------


class TestFENPrepDataset:
    """Test FENPrepDataset.__getitem__."""

    def test_valid_fen_returns_dict(self):
        ds = FENPrepDataset([_STARTING_FEN])
        item = ds[0]
        assert item is not None
        assert set(item.keys()) == {"sequences", "board_tokens", "legal_mask", "action_indices"}

    def test_sequences_shape(self):
        ds = FENPrepDataset([_STARTING_FEN])
        item = ds[0]
        board = chess.Board(_STARTING_FEN)
        n_legal = len(list(board.legal_moves))
        # sequences: [num_legal, SEQUENCE_LENGTH + 2]
        assert item["sequences"].shape[0] == n_legal
        assert item["sequences"].shape[1] == 77 + 2  # FEN tokens + action + return bucket

    def test_board_tokens_shape(self):
        ds = FENPrepDataset([_STARTING_FEN])
        item = ds[0]
        assert item["board_tokens"].shape == (77,)
        assert item["board_tokens"].dtype == torch.long

    def test_legal_mask_shape(self):
        ds = FENPrepDataset([_STARTING_FEN])
        item = ds[0]
        assert item["legal_mask"].shape == (ACTION_SPACE_SIZE,)
        assert item["legal_mask"].dtype == torch.bool

    def test_action_indices_match_legal_moves(self):
        ds = FENPrepDataset([_STARTING_FEN])
        item = ds[0]
        board = chess.Board(_STARTING_FEN)
        n_legal = len(list(board.legal_moves))
        assert item["action_indices"].shape == (n_legal,)
        # All action indices should be marked legal in the mask
        for idx in item["action_indices"]:
            assert item["legal_mask"][idx.item()]

    def test_invalid_fen_returns_none(self):
        ds = FENPrepDataset(["not a valid fen"])
        assert ds[0] is None

    def test_checkmate_returns_none(self):
        checkmate_fen = "rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        ds = FENPrepDataset([checkmate_fen])
        assert ds[0] is None

    def test_len(self):
        fens = [_STARTING_FEN, _TEST_FEN, "invalid"]
        ds = FENPrepDataset(fens)
        assert len(ds) == 3


class TestCollatePositions:
    """Test collate_positions collate function."""

    def test_filters_nones(self):
        ds = FENPrepDataset([_STARTING_FEN, "invalid", _TEST_FEN])
        batch = [ds[i] for i in range(3)]
        collated = collate_positions(batch)
        assert collated["valid"] is True
        assert collated["num_failed"] == 1
        assert len(collated["seq_counts"]) == 2

    def test_all_invalid_returns_not_valid(self):
        collated = collate_positions([None, None])
        assert collated["valid"] is False
        assert collated["num_failed"] == 2

    def test_sequences_concatenated(self):
        ds = FENPrepDataset([_STARTING_FEN, _TEST_FEN])
        batch = [ds[0], ds[1]]
        collated = collate_positions(batch)
        total_seqs = sum(collated["seq_counts"])
        assert collated["sequences"].shape[0] == total_seqs
        assert collated["sequences"].shape[1] == 77 + 2

    def test_lists_have_correct_length(self):
        ds = FENPrepDataset([_STARTING_FEN, _TEST_FEN])
        batch = [ds[0], ds[1]]
        collated = collate_positions(batch)
        n = len(collated["seq_counts"])
        assert len(collated["board_tokens"]) == n
        assert len(collated["legal_masks"]) == n
        assert len(collated["action_indices"]) == n


class TestPostprocessTeacherBatch:
    """Test vectorized postprocess_teacher_batch."""

    @staticmethod
    def _make_mock_batch_data(fens, num_buckets=128):
        """Build a collated batch and matching mock log_probs."""
        ds = FENPrepDataset(fens)
        items = [ds[i] for i in range(len(fens))]
        batch = collate_positions(items)
        total_seqs = batch["sequences"].shape[0]
        # Mock log-probs: [total_seqs, num_buckets]
        all_log_probs = np.random.randn(total_seqs, num_buckets).astype(np.float32)
        bucket_values = np.linspace(0.0, 1.0, num_buckets + 1)
        bucket_values = (bucket_values[:-1] + bucket_values[1:]) / 2
        return batch, all_log_probs, bucket_values

    def test_returns_correct_count(self):
        fens = [_STARTING_FEN, _TEST_FEN]
        batch, log_probs, bv = self._make_mock_batch_data(fens)
        samples = postprocess_teacher_batch(log_probs, bv, batch, top_k=5, temperature=1.0)
        assert len(samples) == 2

    def test_sample_keys(self):
        batch, log_probs, bv = self._make_mock_batch_data([_STARTING_FEN])
        samples = postprocess_teacher_batch(log_probs, bv, batch, top_k=5, temperature=1.0)
        assert set(samples[0].keys()) == {
            "board_tokens", "legal_mask", "teacher_action_indices", "teacher_probs",
        }

    def test_probs_sum_to_one(self):
        batch, log_probs, bv = self._make_mock_batch_data([_STARTING_FEN, _TEST_FEN])
        samples = postprocess_teacher_batch(log_probs, bv, batch, top_k=8, temperature=1.0)
        for s in samples:
            assert abs(s["teacher_probs"].sum().item() - 1.0) < 1e-5

    def test_top_k_limits(self):
        batch, log_probs, bv = self._make_mock_batch_data([_STARTING_FEN])
        for k in [1, 3, 5, 8]:
            samples = postprocess_teacher_batch(log_probs, bv, batch, top_k=k, temperature=1.0)
            assert len(samples[0]["teacher_action_indices"]) <= k
            assert len(samples[0]["teacher_probs"]) == len(samples[0]["teacher_action_indices"])

    def test_teacher_actions_are_legal(self):
        batch, log_probs, bv = self._make_mock_batch_data([_STARTING_FEN, _TEST_FEN])
        samples = postprocess_teacher_batch(log_probs, bv, batch, top_k=5, temperature=1.0)
        for s in samples:
            for idx in s["teacher_action_indices"]:
                assert s["legal_mask"][idx.item()], f"Action {idx.item()} not legal"

    def test_temperature_affects_distribution(self):
        batch, log_probs, bv = self._make_mock_batch_data([_STARTING_FEN])
        s_t1 = postprocess_teacher_batch(log_probs, bv, batch, top_k=8, temperature=1.0)
        s_t01 = postprocess_teacher_batch(log_probs, bv, batch, top_k=8, temperature=0.1)
        max_t1 = s_t1[0]["teacher_probs"].max().item()
        max_t01 = s_t01[0]["teacher_probs"].max().item()
        assert max_t01 >= max_t1

    def test_matches_per_position_output(self):
        """Vectorized batch should match per-position postprocess_teacher_output."""
        fens = [_STARTING_FEN, _TEST_FEN]
        batch, all_log_probs, bv = self._make_mock_batch_data(fens)

        # Batch result
        batch_samples = postprocess_teacher_batch(
            all_log_probs, bv, batch, top_k=5, temperature=1.0,
        )

        # Per-position result
        offset = 0
        for i, count in enumerate(batch["seq_counts"]):
            lp = all_log_probs[offset : offset + count]
            offset += count
            indices, probs = postprocess_teacher_output(
                lp, bv, batch["action_indices"][i], top_k=5, temperature=1.0,
            )
            assert torch.equal(batch_samples[i]["teacher_action_indices"], indices)
            assert torch.allclose(batch_samples[i]["teacher_probs"], probs, atol=1e-6)


class TestCountExistingShards:
    """Test count_existing_shards resume helper."""

    def test_empty_dir(self, tmp_path):
        shard_idx, samples = count_existing_shards(str(tmp_path))
        assert shard_idx == 0
        assert samples == 0

    def test_counts_samples(self, tmp_path):
        engine = _make_mock_engine()
        bv = _make_bucket_values()
        samples = []
        for fen in [_STARTING_FEN, _TEST_FEN]:
            s = process_position(engine, bv, fen, top_k=5, temperature=1.0)
            if s is not None:
                samples.append(s)
        save_shard(samples, str(tmp_path / "shard_0000.pt"))

        shard_idx, count = count_existing_shards(str(tmp_path))
        assert shard_idx == 1
        assert count == len(samples)

    def test_counts_multiple_shards(self, tmp_path):
        engine = _make_mock_engine()
        bv = _make_bucket_values()
        s = process_position(engine, bv, _STARTING_FEN, top_k=5, temperature=1.0)
        save_shard([s], str(tmp_path / "shard_0000.pt"))
        save_shard([s, s], str(tmp_path / "shard_0001.pt"))

        shard_idx, count = count_existing_shards(str(tmp_path))
        assert shard_idx == 2
        assert count == 3


class TestDistillDatasetAndCollate:
    """Test DistillDataset loading and collate_distill_batch."""

    @pytest.fixture
    def shard_dir(self, tmp_path):
        """Create a temp directory with a small shard file."""
        engine = _make_mock_engine()
        bucket_values = _make_bucket_values()

        samples = []
        # Use diverse positions to get a spread across train/eval split
        positions = [
            _STARTING_FEN,
            _TEST_FEN,
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
            "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
            "rnbqkbnr/ppp1pppp/3p4/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
            "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
        ]
        for fen in positions:
            s = process_position(engine, bucket_values, fen, top_k=5, temperature=1.0)
            if s is not None:
                samples.append(s)

        save_shard(samples, str(tmp_path / "shard_0000.pt"))
        return tmp_path

    def test_dataset_loads(self, shard_dir):
        ds = DistillDataset(str(shard_dir), is_eval=False, eval_fraction=0.0)
        assert len(ds) > 0

    def test_dataset_item_format(self, shard_dir):
        ds = DistillDataset(str(shard_dir), is_eval=False, eval_fraction=0.0)
        board_tokens, legal_mask, teacher_indices, teacher_probs, k_mask = ds[0]

        assert board_tokens.shape == (77,)
        assert legal_mask.shape == (ACTION_SPACE_SIZE,)
        assert teacher_indices.dim() == 1
        assert teacher_probs.dim() == 1
        assert len(teacher_indices) == len(teacher_probs)

    def test_train_eval_split(self, shard_dir):
        train_ds = DistillDataset(str(shard_dir), is_eval=False, eval_fraction=0.5)
        eval_ds = DistillDataset(str(shard_dir), is_eval=True, eval_fraction=0.5)

        # With 50% split and 8 samples, both should have some
        total = len(train_ds) + len(eval_ds)
        assert total > 0

    def test_collate_batch(self, shard_dir):
        ds = DistillDataset(str(shard_dir), is_eval=False, eval_fraction=0.0)
        batch = [ds[i] for i in range(min(4, len(ds)))]
        board_tokens, legal_masks, teacher_indices, teacher_probs, k_mask = collate_distill_batch(batch)

        B = len(batch)
        assert board_tokens.shape == (B, 77)
        assert legal_masks.shape == (B, ACTION_SPACE_SIZE)
        # teacher_indices and teacher_probs are padded to max_k
        assert teacher_indices.shape[0] == B
        assert teacher_probs.shape[0] == B
        assert k_mask.shape == teacher_indices.shape

    def test_collate_k_mask_matches_valid_entries(self, shard_dir):
        ds = DistillDataset(str(shard_dir), is_eval=False, eval_fraction=0.0)
        batch = [ds[i] for i in range(min(3, len(ds)))]
        _, _, teacher_indices, teacher_probs, k_mask = collate_distill_batch(batch)

        for i, (_, _, orig_indices, _, orig_k_mask) in enumerate(batch):
            k = orig_k_mask.sum().item()
            assert k_mask[i, :k].all()
            if k < k_mask.shape[1]:
                assert not k_mask[i, k:].any()

    def test_no_shards_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DistillDataset(str(tmp_path), is_eval=False)


class TestGenerateConfig:
    """Test GenerateConfig defaults."""

    def test_defaults(self):
        cfg = GenerateConfig()
        assert cfg.teacher_model == "270M"
        assert cfg.top_k == 8
        assert cfg.teacher_temperature == 1.0
        assert cfg.checkpoint_step == 6_400_000
        assert cfg.process_batch_size == 256
        assert cfg.num_workers == 0


# ---------------------------------------------------------------------------
# Integration tests (require JAX + Haiku + downloaded checkpoint)
# ---------------------------------------------------------------------------


@requires_jax_haiku
@requires_270m_checkpoint
class TestBuildAndUseTeacherEngine:
    """Integration tests: load the real 270M model and run inference."""

    @pytest.fixture(scope="class")
    def teacher(self):
        """Build the 270M teacher engine (shared across tests in this class)."""
        from src.distill.teacher import build_teacher_engine

        engine, bucket_values = build_teacher_engine(
            model_name="270M",
            checkpoint_dir=str(_CKPT_DIR),
            checkpoint_step=6_400_000,
            batch_size=32,
        )
        return engine, bucket_values

    def test_engine_loads(self, teacher):
        engine, bucket_values = teacher
        assert engine is not None
        assert bucket_values is not None
        assert len(bucket_values) == 128

    def test_analyse_starting_position(self, teacher):
        engine, _ = teacher
        board = chess.Board()
        result = engine.analyse(board)

        assert "log_probs" in result
        assert "fen" in result
        n_legal = len(list(board.legal_moves))
        assert result["log_probs"].shape == (n_legal, 128)

    def test_play_returns_legal_move(self, teacher):
        engine, _ = teacher
        engine._return_buckets_values = teacher[1]
        engine.temperature = None
        board = chess.Board()
        move = engine.play(board)
        assert move in board.legal_moves

    def test_process_position_integration(self, teacher):
        engine, bucket_values = teacher
        result = process_position(engine, bucket_values, _STARTING_FEN, top_k=8, temperature=1.0)

        assert result is not None
        assert result["teacher_probs"].sum().item() == pytest.approx(1.0, abs=1e-5)
        assert len(result["teacher_action_indices"]) <= 8
        # Teacher should produce a non-uniform distribution
        assert result["teacher_probs"].max().item() > result["teacher_probs"].min().item()

    def test_process_position_multiple_fens(self, teacher):
        """Verify the teacher can handle various positions."""
        engine, bucket_values = teacher
        fens = [
            _STARTING_FEN,
            _TEST_FEN,
            # Sicilian
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
            # Middlegame
            "r1bqkb1r/pppppppp/2n2n2/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 2 3",
        ]
        for fen in fens:
            result = process_position(engine, bucket_values, fen, top_k=5, temperature=1.0)
            assert result is not None, f"process_position returned None for {fen}"

    def test_batched_pipeline_integration(self, teacher):
        """Full batched pipeline: FENPrepDataset → collate → predict_fn → postprocess."""
        engine, bucket_values = teacher
        fens = [
            _STARTING_FEN,
            _TEST_FEN,
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
        ]
        ds = FENPrepDataset(fens)
        items = [ds[i] for i in range(len(fens))]
        batch = collate_positions(items)

        assert batch["valid"]
        all_log_probs = engine.predict_fn(batch["sequences"].numpy())[:, -1]
        samples = postprocess_teacher_batch(
            all_log_probs, bucket_values, batch, top_k=5, temperature=1.0,
        )

        assert len(samples) == 3
        for s in samples:
            assert s["teacher_probs"].sum().item() == pytest.approx(1.0, abs=1e-5)
            assert len(s["teacher_action_indices"]) <= 5
            for idx in s["teacher_action_indices"]:
                assert s["legal_mask"][idx.item()]

    def test_end_to_end_shard_pipeline(self, teacher, tmp_path):
        """Full pipeline: teacher inference -> save shard -> load dataset -> collate."""
        engine, bucket_values = teacher

        samples = []
        for fen in [_STARTING_FEN, _TEST_FEN]:
            s = process_position(engine, bucket_values, fen, top_k=5, temperature=1.0)
            assert s is not None
            samples.append(s)

        shard_path = str(tmp_path / "shard_0000.pt")
        save_shard(samples, shard_path)

        ds = DistillDataset(str(tmp_path), is_eval=False, eval_fraction=0.0)
        assert len(ds) == 2

        batch = collate_distill_batch([ds[0], ds[1]])
        board_tokens, legal_masks, teacher_indices, teacher_probs, k_mask = batch
        assert board_tokens.shape == (2, 77)
        assert k_mask.any()
