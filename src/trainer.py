import time
import random
import string
import pytorch_lightning as pl

from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger


class TaggedModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint with a unique state key tag.

    Newer Lightning versions require unique callback state keys when multiple
    instances of the same stateful callback class are attached to one Trainer.
    """

    def __init__(self, *, state_key_tag: str, **kwargs):
        super().__init__(**kwargs)
        self._state_key_tag = state_key_tag

    @property
    def state_key(self) -> str:
        return f"{super().state_key}-{self._state_key_tag}"


def generate_run_name(project: str = "chess-grpo") -> str:
    """Generate a unique run name with timestamp and random suffix.

    Args:
        project: Project name prefix

    Returns:
        Unique run name string
    """
    timestamp = time.strftime("%Y%m%d-%H%M")
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{project}-{timestamp}-{random_suffix}"


def get_trainer(num_epochs: int = 5000,
                checkpoint_dir: str = "checkpoints",
                checkpoint_every_n_epochs: int = 5,
                keep_n_checkpoints: int = 3,
                use_wandb: bool = True,
                wandb_project: str = "Chess-GRPO-Bot",
                wandb_log_model: bool = False,
                callbacks: list[Callback] | None = None) -> pl.Trainer:
    """Create a PyTorch Lightning trainer with WandB logging and checkpointing.

    Args:
        num_epochs: Maximum number of training epochs
        checkpoint_dir: Directory to save model checkpoints
        checkpoint_every_n_epochs: Save periodic checkpoint every N epochs
        keep_n_checkpoints: Keep last N periodic checkpoints per run
        wandb_log_model: Whether to upload model checkpoint artifacts to WandB

    Returns:
        Configured PyTorch Lightning trainer
    """
    run_name = generate_run_name()
    print(f"Generated run name: {run_name}")

    logger = (
        WandbLogger(project=wandb_project, log_model=wandb_log_model, name=run_name)
        if use_wandb
        else CSVLogger(save_dir=checkpoint_dir, name=run_name)
    )

    # Best checkpoint - saves top 2 based on loss
    best_checkpoint_cb = TaggedModelCheckpoint(
        state_key_tag="best",
        dirpath=checkpoint_dir,
        filename=run_name + "-best-{epoch:02d}-{train_total_loss:.4f}",
        save_top_k=2,
        monitor="train_total_loss",
        mode="min"
    )

    # Periodic checkpoint for crash recovery
    # Fixed filenames (periodic-0, periodic-1, etc.) that rotate within each run
    periodic_checkpoint_cb = TaggedModelCheckpoint(
        state_key_tag="periodic",
        dirpath=checkpoint_dir,
        filename=run_name + "-periodic",
        save_top_k=keep_n_checkpoints,
        monitor="train_total_loss",
        mode="min",
        every_n_epochs=checkpoint_every_n_epochs,
        save_last=True,  # Always keep the very last checkpoint
    )

    trainer_callbacks = [best_checkpoint_cb, periodic_checkpoint_cb]
    if callbacks:
        trainer_callbacks.extend(callbacks)

    return pl.Trainer(
        max_epochs=num_epochs,
        # Gradient clipping handled manually in GRPOChessTransformer.training_step
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=trainer_callbacks,
        log_every_n_steps=1  # Log every step for GRPO debug
    )
                      
