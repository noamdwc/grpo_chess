"""Regression tests for review findings on the reasoning refactor."""
from types import SimpleNamespace

import chess
import torch


def test_config_loader_exports_legacy_helpers():
    from src.configs.config_loader import dict_to_dataclass, load_yaml_file

    raw = load_yaml_file("distill.yaml")
    cfg = dict_to_dataclass(SimpleNamespace, {})
    assert isinstance(raw, dict)
    assert cfg is not None


def test_load_experiment_config_accepts_legacy_grpo_yaml():
    from src.configs.config_loader import load_experiment_config

    cfg = load_experiment_config("grpo_9m_direct.yaml")
    assert cfg.model.base_checkpoint == "checkpoints/jax_9m_converted.pt"
    assert cfg.reasoning.max_think_tokens == 8
    assert cfg.grpo.k_samples_per_root == 4
    assert cfg.leaf_evaluator.mode == "stockfish"


def test_load_experiment_config_accepts_legacy_colab_yaml():
    from src.configs.config_loader import load_experiment_config

    cfg = load_experiment_config("grpo_colab_e2e_smoke.yaml")
    assert cfg.training.num_epochs == 1
    assert cfg.reasoning.max_think_tokens == 6
    assert cfg.grpo.k_samples_per_root == 4


def test_distill_generate_dataset_position_extraction_no_deleted_module_dependency():
    from src.distill.generate_dataset import get_positions_from_game

    positions = get_positions_from_game(
        ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"],
        skip_first_n=1,
        skip_last_n=1,
        sample_n=8,
    )
    assert len(positions) > 0
    fen, move, move_idx = positions[0]
    assert isinstance(fen, str)
    assert isinstance(move, str)
    assert isinstance(move_idx, int)


def test_recompute_log_probs_uses_move_temperature_for_final_move():
    from src.grpo_logic.model import ReasoningGRPOLightningModule

    class DummyPolicy:
        def logprob_sequence(self, tokens, seq_lens, temperature):
            del tokens, seq_lens
            return torch.full((2, 5), float(temperature))

    fake = SimpleNamespace(
        cfg=SimpleNamespace(
            grpo=SimpleNamespace(
                rollout_temperature=0.7,
                move_sampling_temperature=1.3,
            )
        ),
        policy_model=DummyPolicy(),
    )
    tokens = torch.zeros((2, 5), dtype=torch.long)
    seq_lens = torch.tensor([5, 5], dtype=torch.long)
    final_positions = torch.tensor([3, 4], dtype=torch.long)

    out = ReasoningGRPOLightningModule._recompute_log_probs(fake, tokens, seq_lens, final_positions)
    assert out[0, 2].item() == pytest_approx(0.7)
    assert out[0, 3].item() == pytest_approx(1.3)
    assert out[1, 4].item() == pytest_approx(1.3)


def test_post_fen_after_rival_honors_frozen_dm_mode(monkeypatch):
    from src.grpo_logic.model import ReasoningGRPOLightningModule

    calls = []

    def fake_play_final_and_respond(*, model, root_fen, m_final_token):
        calls.append((model, root_fen, m_final_token))
        return root_fen, False

    monkeypatch.setattr("src.grpo_logic.model.play_final_and_respond", fake_play_final_and_respond)

    fake = SimpleNamespace(
        cfg=SimpleNamespace(rival=SimpleNamespace(mode="frozen_dm_9m")),
        old_policy_model=object(),
        frozen_rival_model="frozen-model",
    )
    ReasoningGRPOLightningModule._post_fen_after_rival(fake, root_fen=chess.STARTING_FEN, final_token=0)
    assert calls == [("frozen-model", chess.STARTING_FEN, 0)]


def test_post_fen_after_rival_honors_stockfish_mode():
    from src.grpo_logic.model import ReasoningGRPOLightningModule
    from src.searchless_chess_imports import MOVE_TO_ACTION

    class DummyStockfish:
        def act(self, board):
            assert board.move_stack[-1] == chess.Move.from_uci("e2e4")
            return chess.Move.from_uci("e7e5")

    fake = SimpleNamespace(
        cfg=SimpleNamespace(rival=SimpleNamespace(mode="stockfish")),
    )
    post_fen, terminal = ReasoningGRPOLightningModule._post_fen_after_rival(
        fake,
        root_fen=chess.STARTING_FEN,
        final_token=MOVE_TO_ACTION["e2e4"],
        stockfish_player=DummyStockfish(),
    )
    assert not terminal
    board = chess.Board(post_fen)
    assert board.turn == chess.WHITE


def test_convert_jax_checkpoint_verification_skips_without_models_9m(monkeypatch):
    import importlib

    from src.distill.convert_jax_checkpoint import _verify_converted_state_dict

    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "src.models_9m":
            raise ModuleNotFoundError(name)
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    summary, extras = _verify_converted_state_dict({}, {"embed_dim": 256, "num_layers": 8, "num_heads": 8, "ffn_dim": 1024})
    assert summary["verification_skipped"] is True
    assert "verification_reason" in summary
    assert extras == {}


def pytest_approx(value):
    import pytest

    return pytest.approx(value)
