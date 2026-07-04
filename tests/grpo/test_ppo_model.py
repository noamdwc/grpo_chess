import chess
import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

import src.grpo_logic.ppo_model as ppo_model
from src.configs.config_loader import (
    AVPPOExperimentConfig,
    AVPPOModelConfig,
    EvalConfig,
    LeafEvaluatorConfig,
    PPOConfig,
    TrainingConfig,
    load_av_ppo_config,
)
from src.chess.boards_dataset import ChessDatasetConfig
from src.chess.stockfish import StockfishConfig
from src.dm_port.av_adapter import NUM_BUCKETS

WHITE_TO_MOVE_FEN = chess.STARTING_FEN
BLACK_TO_MOVE_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"


class FakeAVModel(torch.nn.Module):
    """Deterministic stand-in for the DM AV transformer (see tests/test_av_adapter.py)."""

    def forward(self, targets: torch.Tensor) -> torch.Tensor:
        n, seq_len = targets.shape
        bucket_ids = (targets[:, -2] % NUM_BUCKETS).long()
        log_probs = torch.full(
            (n, seq_len, NUM_BUCKETS), -float("inf"), dtype=torch.float32, device=targets.device
        )
        log_probs[torch.arange(n, device=targets.device), -1, bucket_ids] = 0.0
        return log_probs


def _make_cfg(ppo_epochs: int = 3) -> AVPPOExperimentConfig:
    return AVPPOExperimentConfig(
        model=AVPPOModelConfig(base_checkpoint="unused-by-fake.pt"),
        ppo=PPOConfig(lr=1e-4, ppo_epochs=ppo_epochs),
        leaf_evaluator=LeafEvaluatorConfig(mode="stockfish"),
        training=TrainingConfig(),
        eval=EvalConfig(),
        stockfish=StockfishConfig(),
        dataset=ChessDatasetConfig(),
    )


@pytest.fixture()
def module_factory(monkeypatch):
    def factory(ppo_epochs: int = 3, reward_recorder: list | None = None):
        monkeypatch.setattr(ppo_model, "load_dm_port_model", lambda path: FakeAVModel())

        def fake_evaluate_leaf(fen, pov_is_white, mode, **kwargs):
            if reward_recorder is not None:
                reward_recorder.append({"fen": fen, "pov_is_white": pov_is_white, "mode": mode})
            return 0.5 if pov_is_white else -0.5

        monkeypatch.setattr(ppo_model, "evaluate_leaf", fake_evaluate_leaf)
        return ppo_model.AVPPOLightningModule(_make_cfg(ppo_epochs=ppo_epochs))

    return factory


def _fit(module: pl.LightningModule, fens: list[str], batch_size: int = 2) -> pl.Trainer:
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(module, DataLoader(fens, batch_size=batch_size, num_workers=0))
    return trainer


def test_load_av_ppo_config_preserves_ppo_epochs():
    cfg = load_av_ppo_config("ppo_av.yaml")
    assert cfg.ppo.ppo_epochs == 4
    assert cfg.leaf_evaluator.mode == "stockfish"
    assert cfg.model.base_checkpoint.endswith("9M.pt")


def test_training_step_runs_ppo_epochs_updates(module_factory, monkeypatch):
    module = module_factory(ppo_epochs=3)
    calls = {"n": 0}
    real_loss = ppo_model.safe_ppo_loss

    def counting_loss(*args, **kwargs):
        calls["n"] += 1
        return real_loss(*args, **kwargs)

    monkeypatch.setattr(ppo_model, "safe_ppo_loss", counting_loss)

    _fit(module, [WHITE_TO_MOVE_FEN, BLACK_TO_MOVE_FEN], batch_size=2)

    assert calls["n"] == 3


def test_first_ppo_epoch_ratio_is_one_and_metrics_logged(module_factory):
    module = module_factory(ppo_epochs=2)
    trainer = _fit(module, [WHITE_TO_MOVE_FEN, BLACK_TO_MOVE_FEN], batch_size=2)

    metrics = trainer.callback_metrics
    assert torch.isclose(metrics["train/ratio_first_epoch"], torch.tensor(1.0), atol=1e-4)
    for key in ("train_total_loss", "train/clip_fraction", "train/reward_mean", "train/kl_divergence"):
        assert key in metrics


def test_reward_uses_post_action_fen_and_root_pov(module_factory):
    rewards_seen: list[dict] = []
    module = module_factory(ppo_epochs=1, reward_recorder=rewards_seen)
    _fit(module, [WHITE_TO_MOVE_FEN, BLACK_TO_MOVE_FEN], batch_size=1)

    assert [record["pov_is_white"] for record in rewards_seen] == [True, False]
    for record, root_fen in zip(rewards_seen, [WHITE_TO_MOVE_FEN, BLACK_TO_MOVE_FEN]):
        root_board = chess.Board(root_fen)
        scored_board = chess.Board(record["fen"])
        assert scored_board.turn != root_board.turn  # exactly one (legal) move was pushed
        assert record["mode"] == "stockfish"


def test_old_policy_synced_after_training_step(module_factory):
    module = module_factory(ppo_epochs=2)
    _fit(module, [WHITE_TO_MOVE_FEN, BLACK_TO_MOVE_FEN], batch_size=2)

    new_state = module.policy_model.state_dict()
    old_state = module.old_policy_model.state_dict()
    assert all(torch.equal(new_state[key], old_state[key]) for key in new_state)
