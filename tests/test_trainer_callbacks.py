from pytorch_lightning.callbacks import ModelCheckpoint

from src.trainer import get_trainer


def test_get_trainer_uses_unique_model_checkpoint_state_keys(tmp_path) -> None:
    trainer = get_trainer(
        num_epochs=1,
        checkpoint_dir=str(tmp_path),
        checkpoint_every_n_epochs=1,
        keep_n_checkpoints=1,
        use_wandb=False,
    )

    checkpoint_callbacks = [cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)]
    state_keys = [cb.state_key for cb in checkpoint_callbacks]

    assert len(checkpoint_callbacks) >= 2
    assert len(set(state_keys)) == len(state_keys)
