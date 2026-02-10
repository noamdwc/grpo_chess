"""Dataset for loading distillation shards and collating variable-length teacher labels."""

import hashlib
from pathlib import Path

import torch
from torch.utils.data import Dataset


class DistillDataset(Dataset):
    """Dataset loading .pt shard files with teacher soft labels.

    Each shard contains:
        board_tokens: [N, 77]
        legal_mask: [N, 1968]
        teacher_action_indices: list of Tensor[k_i]  (variable length)
        teacher_probs: list of Tensor[k_i]
    """

    def __init__(self, data_dir: str, is_eval: bool = False, eval_fraction: float = 0.05):
        self.data_dir = Path(data_dir)
        shard_paths = sorted(self.data_dir.glob("shard_*.pt"))
        if not shard_paths:
            raise FileNotFoundError(f"No shard files found in {data_dir}")

        # Load all shards into memory
        all_board_tokens = []
        all_legal_masks = []
        all_teacher_indices = []
        all_teacher_probs = []

        for sp in shard_paths:
            shard = torch.load(sp, weights_only=False)
            n = len(shard["board_tokens"])
            for i in range(n):
                # Hash-based split
                key = f"{shard['board_tokens'][i].sum().item()}-{i}-{sp.name}"
                h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 10000
                is_eval_sample = h < (eval_fraction * 10000)
                if is_eval_sample != is_eval:
                    continue

                all_board_tokens.append(shard["board_tokens"][i])
                all_legal_masks.append(shard["legal_mask"][i])
                all_teacher_indices.append(shard["teacher_action_indices"][i])
                all_teacher_probs.append(shard["teacher_probs"][i])

        self.board_tokens = torch.stack(all_board_tokens) if all_board_tokens else torch.empty(0, 77, dtype=torch.long)
        self.legal_masks = torch.stack(all_legal_masks) if all_legal_masks else torch.empty(0, 1968, dtype=torch.bool)
        self.teacher_indices = all_teacher_indices
        self.teacher_probs = all_teacher_probs

        split_name = "eval" if is_eval else "train"
        print(f"Loaded {len(self)} {split_name} samples from {len(shard_paths)} shards")

    def __len__(self) -> int:
        return len(self.board_tokens)

    def __getitem__(self, idx: int):
        return (
            self.board_tokens[idx],
            self.legal_masks[idx],
            self.teacher_indices[idx],
            self.teacher_probs[idx],
        )


def collate_distill_batch(batch):
    """Collate function that pads variable-length teacher labels to batch max-k.

    Returns:
        board_tokens: [B, 77]
        legal_masks: [B, 1968]
        teacher_indices: [B, max_k]  (padded with 0)
        teacher_probs: [B, max_k]  (padded with 0)
        k_mask: [B, max_k]  (True for valid entries)
    """
    board_tokens, legal_masks, indices_list, probs_list = zip(*batch)

    board_tokens = torch.stack(board_tokens)
    legal_masks = torch.stack(legal_masks)

    # Pad variable-length teacher labels
    max_k = max(len(idx) for idx in indices_list)
    B = len(indices_list)

    teacher_indices = torch.zeros(B, max_k, dtype=torch.long)
    teacher_probs = torch.zeros(B, max_k, dtype=torch.float32)
    k_mask = torch.zeros(B, max_k, dtype=torch.bool)

    for i, (idx, prob) in enumerate(zip(indices_list, probs_list)):
        k = len(idx)
        teacher_indices[i, :k] = idx
        teacher_probs[i, :k] = prob
        k_mask[i, :k] = True

    return board_tokens, legal_masks, teacher_indices, teacher_probs, k_mask
