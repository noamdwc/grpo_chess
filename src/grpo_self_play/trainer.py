import time
import random
import string
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def generate_run_name(project="chess-grpo"):
    timestamp = time.strftime("%Y%m%d-%H%M")
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{project}-{timestamp}-{random_suffix}"




def get_trainer(num_epochs=5000, checkpoint_dir="/content/drive/MyDrive/data/grpo-chess/checkpoints/"): 
    run_name = generate_run_name()
    print(f"Generated run name: {run_name}")

    wandb_logger = WandbLogger(project="Chess-GRPO-Bot", log_model=True, name=run_name)
    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=run_name + "-{epoch:02d}-{train_total_loss:.4f}",
        save_top_k=2,                # keep best
        monitor="train_total_loss",  # metric to track
        mode="min"
    )
    return pl.Trainer(max_epochs=num_epochs,
                      accelerator="auto",
                      devices=1,
                      logger=wandb_logger,
                      callbacks=[checkpoint_cb],
                      log_every_n_steps=1) # Log every step for GRPO debug
                      

