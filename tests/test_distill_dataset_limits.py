import pytest
import torch

from src.distill.distill_dataset import DistillDataset


def _write_shard(path, n_samples: int, token_value: int) -> None:
    board_tokens = torch.full((n_samples, 77), token_value, dtype=torch.long)
    legal_mask = torch.zeros((n_samples, 1968), dtype=torch.bool)
    legal_mask[:, 0] = True
    teacher_action_indices = [torch.tensor([0], dtype=torch.long) for _ in range(n_samples)]
    teacher_probs = [torch.tensor([1.0], dtype=torch.float32) for _ in range(n_samples)]
    torch.save(
        {
            "board_tokens": board_tokens,
            "legal_mask": legal_mask,
            "teacher_action_indices": teacher_action_indices,
            "teacher_probs": teacher_probs,
        },
        path,
    )


def _write_variable_k_shard(path, k_values: list[int], token_value: int) -> None:
    n_samples = len(k_values)
    board_tokens = torch.full((n_samples, 77), token_value, dtype=torch.long)
    legal_mask = torch.zeros((n_samples, 1968), dtype=torch.bool)
    teacher_action_indices = []
    teacher_probs = []

    for i, k in enumerate(k_values):
        legal_mask[i, :k] = True
        teacher_action_indices.append(torch.arange(k, dtype=torch.long))
        teacher_probs.append(torch.full((k,), 1.0 / k, dtype=torch.float32))

    torch.save(
        {
            "board_tokens": board_tokens,
            "legal_mask": legal_mask,
            "teacher_action_indices": teacher_action_indices,
            "teacher_probs": teacher_probs,
        },
        path,
    )


def test_load_train_eval_respects_max_shards(tmp_path):
    _write_shard(tmp_path / "shard_0000.pt", 10, 1)
    _write_shard(tmp_path / "shard_0001.pt", 10, 2)

    train_ds, eval_ds = DistillDataset.load_train_eval(
        str(tmp_path),
        eval_fraction=0.2,
        max_shards=1,
    )

    assert len(train_ds) + len(eval_ds) == 10


def test_max_shards_must_be_positive(tmp_path):
    _write_shard(tmp_path / "shard_0000.pt", 4, 1)

    with pytest.raises(ValueError, match="max_shards must be positive"):
        DistillDataset(str(tmp_path), max_shards=0)

    with pytest.raises(ValueError, match="max_shards must be positive"):
        DistillDataset.load_train_eval(str(tmp_path), max_shards=0)


def test_load_train_eval_handles_variable_teacher_width_across_shards(tmp_path):
    _write_variable_k_shard(tmp_path / "shard_0000.pt", [4, 4, 3], token_value=1)
    _write_variable_k_shard(tmp_path / "shard_0001.pt", [3, 2, 3], token_value=2)

    train_ds, eval_ds = DistillDataset.load_train_eval(str(tmp_path), eval_fraction=0.0)

    assert len(train_ds) == 6
    assert len(eval_ds) == 0
    assert train_ds.teacher_indices.shape[1] == 4
    assert train_ds.teacher_probs.shape[1] == 4
    assert train_ds.k_masks.shape[1] == 4


def test_padded_entries_have_zero_probs(tmp_path) -> None:
    """Padded k slots (k_mask=False) must carry teacher_probs == 0.0 and
    teacher_indices == 0. Prevents silent loss contamination if k_mask is ever
    inadvertently ignored and the raw padded values are used instead.
    """
    # k=[3, 2, 1]: after padding to width 3, samples 1 and 2 have padded slots.
    _write_variable_k_shard(tmp_path / "shard_0000.pt", [3, 2, 1], token_value=1)

    train_ds, _ = DistillDataset.load_train_eval(str(tmp_path), eval_fraction=0.0)

    k_masks = train_ds.k_masks          # [N, 3] bool
    teacher_probs = train_ds.teacher_probs  # [N, 3] float32
    teacher_indices = train_ds.teacher_indices  # [N, 3] long

    # All samples must have at least one padded slot for this test to be meaningful.
    assert (~k_masks).any(), "No padded entries found; check shard construction"

    padded_probs = teacher_probs[~k_masks]
    assert (padded_probs == 0.0).all(), (
        f"Padded entries carry non-zero teacher_probs; max abs = {padded_probs.abs().max():.6f}"
    )

    padded_indices = teacher_indices[~k_masks]
    assert (padded_indices == 0).all(), (
        f"Padded entries carry non-zero teacher_indices; max = {padded_indices.max()}"
    )
