import time
import random
import string
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

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
                checkpoint_dir: str = "/content/drive/MyDrive/data/grpo-chess/checkpoints/",
                checkpoint_every_n_epochs: int = 5,
                keep_n_checkpoints: int = 3) -> pl.Trainer:
    """Create a PyTorch Lightning trainer with WandB logging and checkpointing.

    Args:
        num_epochs: Maximum number of training epochs
        checkpoint_dir: Directory to save model checkpoints
        checkpoint_every_n_epochs: Save periodic checkpoint every N epochs
        keep_n_checkpoints: Keep last N periodic checkpoints per run

    Returns:
        Configured PyTorch Lightning trainer
    """
    run_name = generate_run_name()
    print(f"Generated run name: {run_name}")

    wandb_logger = WandbLogger(project="Chess-GRPO-Bot", log_model=True, name=run_name)

    # Best checkpoint - saves top 2 based on loss
    best_checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=run_name + "-best-{epoch:02d}-{train_total_loss:.4f}",
        save_top_k=2,
        monitor="train_total_loss",
        mode="min"
    )

    # Periodic checkpoint for crash recovery
    # Fixed filenames (periodic-0, periodic-1, etc.) that rotate within each run
    periodic_checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=run_name + "-periodic",
        save_top_k=keep_n_checkpoints,
        every_n_epochs=checkpoint_every_n_epochs,
        save_last=True,  # Always keep the very last checkpoint
    )

    return pl.Trainer(
        max_epochs=num_epochs,
        # Gradient clipping handled manually in GRPOChessTransformer.training_step
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        callbacks=[best_checkpoint_cb, periodic_checkpoint_cb],
        log_every_n_steps=1  # Log every step for GRPO debug
    )
                      

