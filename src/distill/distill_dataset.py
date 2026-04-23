"""Dataset for loading distillation shards and collating variable-length teacher labels."""

import hashlib
from pathlib import Path

import torch
from torch.utils.data import Dataset


def _split_mask(tokens, shard_path, eval_fraction):
    """Compute deterministic train/eval split mask for a shard."""
    n = len(tokens)
    sums = tokens.sum(dim=1).long()
    indices = torch.arange(n, dtype=torch.long)
    shard_hash = int(hashlib.md5(shard_path.name.encode()).hexdigest()[:8], 16)
    h = (sums * 2654435761 + indices * 40503 + shard_hash) % 10000
    return h < int(eval_fraction * 10000)


def _pad_teacher_data(raw_indices, raw_probs, keep_mask):
    """Pad variable-length teacher lists to fixed [N, top_k] tensors using boolean indexing.

    Returns (indices_padded, probs_padded, k_mask) for kept samples.
    """
    keep_idx = keep_mask.nonzero(as_tuple=True)[0]
    if len(keep_idx) == 0:
        return torch.empty(0, 1, dtype=torch.long), torch.empty(0, 1), torch.empty(0, 1, dtype=torch.bool)
    # Detect top_k from kept samples
    top_k = max(len(raw_indices[i.item()]) for i in keep_idx)
    n_keep = len(keep_idx)
    padded_indices = torch.zeros(n_keep, top_k, dtype=torch.long)
    padded_probs = torch.zeros(n_keep, top_k, dtype=torch.float32)
    k_mask = torch.zeros(n_keep, top_k, dtype=torch.bool)
    for j, i in enumerate(keep_idx):
        t_idx = raw_indices[i.item()]
        t_prob = raw_probs[i.item()]
        k = len(t_idx)
        padded_indices[j, :k] = t_idx
        padded_probs[j, :k] = t_prob
        k_mask[j, :k] = True
    return padded_indices, padded_probs, k_mask


def _align_teacher_width(tensors: list[torch.Tensor], pad_value) -> list[torch.Tensor]:
    """Pad [N, K] tensors to a common K so they can be concatenated across shards."""
    if not tensors:
        return tensors

    max_k = max(t.shape[1] for t in tensors)
    aligned = []
    for t in tensors:
        if t.shape[1] == max_k:
            aligned.append(t)
            continue
        pad_cols = max_k - t.shape[1]
        pad = torch.full((t.shape[0], pad_cols), pad_value, dtype=t.dtype)
        aligned.append(torch.cat([t, pad], dim=1))
    return aligned


class DistillDataset(Dataset):
    """Dataset loading .pt shard files with teacher soft labels.

    Each shard contains:
        board_tokens: [N, 77]
        legal_mask: [N, 1968]
        teacher_action_indices: list of Tensor[k_i]  (variable length)
        teacher_probs: list of Tensor[k_i]
    """

    def __init__(
        self,
        data_dir: str,
        is_eval: bool = False,
        eval_fraction: float = 0.05,
        max_shards: int | None = None,
    ):
        self.data_dir = Path(data_dir)
        shard_paths = sorted(self.data_dir.glob("shard_*.pt"))
        if max_shards is not None:
            if max_shards <= 0:
                raise ValueError(f"max_shards must be positive, got {max_shards}")
            shard_paths = shard_paths[:max_shards]
        if not shard_paths:
            raise FileNotFoundError(f"No shard files found in {data_dir}")

        all_board_tokens = []
        all_legal_masks = []
        all_teacher_indices = []
        all_teacher_probs = []
        all_k_masks = []
        total = 0
        n_shards = len(shard_paths)

        for si, sp in enumerate(shard_paths, 1):
            shard = torch.load(sp, weights_only=False)
            tokens = shard["board_tokens"]
            masks = shard["legal_mask"]

            is_eval_mask = _split_mask(tokens, sp, eval_fraction)
            keep_mask = is_eval_mask if is_eval else ~is_eval_mask
            n_keep = keep_mask.sum().item()

            all_board_tokens.append(tokens[keep_mask])
            all_legal_masks.append(masks[keep_mask])
            pi, pp, km = _pad_teacher_data(
                shard["teacher_action_indices"], shard["teacher_probs"], keep_mask
            )
            all_teacher_indices.append(pi)
            all_teacher_probs.append(pp)
            all_k_masks.append(km)
            total += n_keep
            print(f"Loading shard {si}/{n_shards}: {sp.name} ... {n_keep} samples ({total} total)")

        all_teacher_indices = _align_teacher_width(all_teacher_indices, pad_value=0)
        all_teacher_probs = _align_teacher_width(all_teacher_probs, pad_value=0.0)
        all_k_masks = _align_teacher_width(all_k_masks, pad_value=False)

        self.board_tokens = torch.cat(all_board_tokens) if all_board_tokens else torch.empty(0, 77, dtype=torch.long)
        self.legal_masks = torch.cat(all_legal_masks) if all_legal_masks else torch.empty(0, 1968, dtype=torch.bool)
        self.teacher_indices = torch.cat(all_teacher_indices) if all_teacher_indices else torch.empty(0, 1, dtype=torch.long)
        self.teacher_probs = torch.cat(all_teacher_probs) if all_teacher_probs else torch.empty(0, 1)
        self.k_masks = torch.cat(all_k_masks) if all_k_masks else torch.empty(0, 1, dtype=torch.bool)

        split_name = "eval" if is_eval else "train"
        print(f"Loaded {len(self)} {split_name} samples from {n_shards} shards")

    @classmethod
    def load_train_eval(
        cls,
        data_dir: str,
        eval_fraction: float = 0.05,
        max_shards: int | None = None,
    ) -> tuple["DistillDataset", "DistillDataset"]:
        """Load shards once and return (train_dataset, eval_dataset)."""
        data_dir = Path(data_dir)
        shard_paths = sorted(data_dir.glob("shard_*.pt"))
        if max_shards is not None:
            if max_shards <= 0:
                raise ValueError(f"max_shards must be positive, got {max_shards}")
            shard_paths = shard_paths[:max_shards]
        if not shard_paths:
            raise FileNotFoundError(f"No shard files found in {data_dir}")

        train_ds = cls.__new__(cls)
        eval_ds = cls.__new__(cls)
        train_ds.data_dir = data_dir
        eval_ds.data_dir = data_dir

        train_tokens, train_masks, train_ti, train_tp, train_km = [], [], [], [], []
        eval_tokens, eval_masks, eval_ti, eval_tp, eval_km = [], [], [], [], []
        train_total, eval_total = 0, 0
        n_shards = len(shard_paths)

        for si, sp in enumerate(shard_paths, 1):
            shard = torch.load(sp, weights_only=False)
            tokens = shard["board_tokens"]
            masks = shard["legal_mask"]
            raw_indices = shard["teacher_action_indices"]
            raw_probs = shard["teacher_probs"]

            is_eval_mask = _split_mask(tokens, sp, eval_fraction)

            # Train split
            train_mask = ~is_eval_mask
            n_train = train_mask.sum().item()
            train_tokens.append(tokens[train_mask])
            train_masks.append(masks[train_mask])
            pi, pp, km = _pad_teacher_data(raw_indices, raw_probs, train_mask)
            train_ti.append(pi)
            train_tp.append(pp)
            train_km.append(km)
            train_total += n_train

            # Eval split
            n_eval = is_eval_mask.sum().item()
            eval_tokens.append(tokens[is_eval_mask])
            eval_masks.append(masks[is_eval_mask])
            pi, pp, km = _pad_teacher_data(raw_indices, raw_probs, is_eval_mask)
            eval_ti.append(pi)
            eval_tp.append(pp)
            eval_km.append(km)
            eval_total += n_eval

            print(f"Loading shard {si}/{n_shards}: {sp.name} ... {n_train} train, {n_eval} eval ({train_total + eval_total} total)")

        train_ti = _align_teacher_width(train_ti, pad_value=0)
        train_tp = _align_teacher_width(train_tp, pad_value=0.0)
        train_km = _align_teacher_width(train_km, pad_value=False)
        eval_ti = _align_teacher_width(eval_ti, pad_value=0)
        eval_tp = _align_teacher_width(eval_tp, pad_value=0.0)
        eval_km = _align_teacher_width(eval_km, pad_value=False)

        train_ds.board_tokens = torch.cat(train_tokens) if train_tokens else torch.empty(0, 77, dtype=torch.long)
        train_ds.legal_masks = torch.cat(train_masks) if train_masks else torch.empty(0, 1968, dtype=torch.bool)
        train_ds.teacher_indices = torch.cat(train_ti) if train_ti else torch.empty(0, 1, dtype=torch.long)
        train_ds.teacher_probs = torch.cat(train_tp) if train_tp else torch.empty(0, 1)
        train_ds.k_masks = torch.cat(train_km) if train_km else torch.empty(0, 1, dtype=torch.bool)

        eval_ds.board_tokens = torch.cat(eval_tokens) if eval_tokens else torch.empty(0, 77, dtype=torch.long)
        eval_ds.legal_masks = torch.cat(eval_masks) if eval_masks else torch.empty(0, 1968, dtype=torch.bool)
        eval_ds.teacher_indices = torch.cat(eval_ti) if eval_ti else torch.empty(0, 1, dtype=torch.long)
        eval_ds.teacher_probs = torch.cat(eval_tp) if eval_tp else torch.empty(0, 1)
        eval_ds.k_masks = torch.cat(eval_km) if eval_km else torch.empty(0, 1, dtype=torch.bool)

        print(f"Loaded {len(train_ds)} train, {len(eval_ds)} eval samples from {n_shards} shards")
        return train_ds, eval_ds

    def __len__(self) -> int:
        return len(self.board_tokens)

    def __getitem__(self, idx: int):
        return (
            self.board_tokens[idx],
            self.legal_masks[idx],
            self.teacher_indices[idx],
            self.teacher_probs[idx],
            self.k_masks[idx],
        )


def collate_distill_batch(batch):
    """Collate pre-padded teacher labels into a batch.

    Returns:
        board_tokens: [B, 77]
        legal_masks: [B, 1968]
        teacher_indices: [B, top_k]
        teacher_probs: [B, top_k]
        k_mask: [B, top_k]
    """
    board_tokens, legal_masks, teacher_indices, teacher_probs, k_masks = zip(*batch)
    return (
        torch.stack(board_tokens),
        torch.stack(legal_masks),
        torch.stack(teacher_indices),
        torch.stack(teacher_probs),
        torch.stack(k_masks),
    )
