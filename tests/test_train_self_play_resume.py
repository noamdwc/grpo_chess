import os
from types import SimpleNamespace


def test_train_passes_ckpt_path_to_trainer_fit(monkeypatch):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import src.train_self_play as tsp

    captured = {}

    class DummyTrainer:
        def fit(self, module, dataloader, ckpt_path=None):
            captured["module"] = module
            captured["dataloader"] = dataloader
            captured["ckpt_path"] = ckpt_path

    monkeypatch.setattr(
        tsp,
        "load_experiment_config",
        lambda config_path, overrides=None: SimpleNamespace(
            dataset=SimpleNamespace(),
            training=SimpleNamespace(
                batch_size=2,
                num_epochs=1,
                checkpoint_dir="checkpoints/test",
                checkpoint_every_n_epochs=1,
                keep_n_checkpoints=1,
                use_wandb=False,
                wandb_project="Chess-GRPO-Bot",
            ),
        ),
    )
    monkeypatch.setattr(tsp, "ChessStartStatesDataset", lambda cfg: ["fen-1", "fen-2"])
    monkeypatch.setattr(
        tsp,
        "DataLoader",
        lambda dataset, batch_size, num_workers: ("loader", dataset, batch_size, num_workers),
    )
    monkeypatch.setattr(tsp, "ReasoningGRPOLightningModule", lambda cfg: ("module", cfg))
    monkeypatch.setattr(tsp, "get_trainer", lambda **kwargs: DummyTrainer())

    tsp.train(config_path="default.yaml", resume_from_checkpoint="checkpoints/run/last.ckpt")

    assert captured["ckpt_path"] == "checkpoints/run/last.ckpt"
    assert captured["dataloader"][0] == "loader"
