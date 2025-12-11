import random
from re import T
import time
import string
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from src.grpo_self_play.chess.chess_logic import ChessStartStatesDataset
from src.grpo_self_play.grpo_logic.model import GRPOChessTransformer, GRPOConfig
from src.grpo_self_play.models import ChessTransformerConfig


GRPO_CONFIG = GRPOConfig(lr=1e-6, num_trajectories=8, trajectory_depth=32)
TRANSFORMER_CONFIG = ChessTransformerConfig()
NUM_EPOCHS = 5000
BATCH_SIZE = 32
STEPS_PER_EPOCH = 1024

def generate_run_name(project="chess-grpo"):
    timestamp = time.strftime("%Y%m%d-%H%M")
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{project}-{timestamp}-{random_suffix}"



def train(): 
    run_name = generate_run_name()
    print(f"Generated run name: {run_name}")

    wandb_logger = WandbLogger(project="Chess-GRPO-Bot", log_model=True, name=run_name)
    dataset = ChessStartStatesDataset(max_steps=STEPS_PER_EPOCH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=2)
    checkpoint_cb = ModelCheckpoint(
        dirpath="drive/MyDrive/data/grpo-chess/checkpoints/",
        filename=run_name + "-{epoch:02d}-{train_total_loss:.4f}",
        save_top_k=2,                # keep best
        monitor="train_total_loss",  # metric to track
        mode="min"
    )

    model = GRPOChessTransformer(TRANSFORMER_CONFIG, GRPO_CONFIG)#, num_trajectories=16, trajectory_depth=32))
    trainer = pl.Trainer(
            max_epochs=NUM_EPOCHS,
            accelerator="auto",
            devices=1,
            logger=wandb_logger,
            callbacks=[checkpoint_cb],
            log_every_n_steps=1 # Log every step for GRPO debug
        )

    print("Starting Training with WandB Tracking...")
    trainer.fit(model, dataloader)