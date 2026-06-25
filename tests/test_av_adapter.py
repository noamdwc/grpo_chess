import torch

from src.dm_port.av_adapter import AVAdapter, NUM_BUCKETS, PPOActorCritic
from src.searchless_chess_imports import ACTION_DIM, SEQUENCE_LENGTH, get_uniform_buckets_edges_values


class FakeAVModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[torch.Tensor] = []

    def forward(self, targets: torch.Tensor) -> torch.Tensor:
        self.calls.append(targets.detach().cpu())
        n, seq_len = targets.shape
        bucket_ids = (targets[:, -2] % NUM_BUCKETS).long()
        log_probs = torch.full(
            (n, seq_len, NUM_BUCKETS),
            -float("inf"),
            dtype=torch.float32,
            device=targets.device,
        )
        log_probs[torch.arange(n, device=targets.device), -1, bucket_ids] = 0.0
        return log_probs


def test_av_adapter_scores_only_legal_actions_and_scatters_to_dense_tensor():
    dm_model = FakeAVModel()
    adapter = AVAdapter(dm_model)
    board_tokens = torch.arange(2 * SEQUENCE_LENGTH, dtype=torch.long).view(2, SEQUENCE_LENGTH)
    legal_masks = torch.zeros((2, ACTION_DIM), dtype=torch.bool)
    legal_masks[0, [1, 7]] = True
    legal_masks[1, [3]] = True

    scores = adapter(board_tokens, legal_masks)

    _, bucket_values_np = get_uniform_buckets_edges_values(NUM_BUCKETS)
    bucket_values = torch.tensor(bucket_values_np, dtype=scores.dtype)
    assert scores.shape == (2, ACTION_DIM)
    assert dm_model.calls[0].shape == (3, SEQUENCE_LENGTH + 2)
    assert torch.isneginf(scores[0, 0])
    assert torch.isneginf(scores[1, 1])
    assert scores[0, 1] == bucket_values[1]
    assert scores[0, 7] == bucket_values[7]
    assert scores[1, 3] == bucket_values[3]


def test_av_adapter_leaves_rows_without_legal_actions_as_negative_infinity():
    adapter = AVAdapter(FakeAVModel())
    board_tokens = torch.zeros((2, SEQUENCE_LENGTH), dtype=torch.long)
    legal_masks = torch.zeros((2, ACTION_DIM), dtype=torch.bool)
    legal_masks[0, 4] = True

    scores = adapter(board_tokens, legal_masks)

    assert torch.isfinite(scores[0, 4])
    assert torch.isneginf(scores[1]).all()


def test_ppo_actor_critic_returns_masked_logits_and_scalar_values():
    module = PPOActorCritic(AVAdapter(FakeAVModel()))
    board_tokens = torch.zeros((2, SEQUENCE_LENGTH), dtype=torch.long)
    legal_masks = torch.zeros((2, ACTION_DIM), dtype=torch.bool)
    legal_masks[0, [1, 2]] = True
    legal_masks[1, [5]] = True

    out = module(board_tokens, legal_masks)

    assert set(out) == {"actor_logits", "value"}
    assert out["actor_logits"].shape == (2, ACTION_DIM)
    assert out["value"].shape == (2,)
    assert torch.isfinite(out["actor_logits"][legal_masks]).all()
    assert torch.isneginf(out["actor_logits"][~legal_masks]).all()
    assert torch.isfinite(out["value"]).all()
