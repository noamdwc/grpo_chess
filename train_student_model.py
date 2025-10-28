import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import lightning as L

from src.data_pytorch import get_dataloaders

# a simple student model for chess
# for demonstration, let's assume input is a flattened board state (e.g., 8x8 board * 12 piece types = 768 features)
# and output is a single value (e.g., board evaluation) or move probabilities (e.g., 73-dimensional for all possible moves)
# we'll start with a simple evaluation output.

class StudentChessModel(nn.Module):
    def __init__(self, input_size=768, hidden_size=256, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # print(x.shape)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class LitChessModel(L.LightningModule):
    def __init__(self, student_model):
        super().__init__()
        self.student_model = student_model
        # for now, using a simple MSE loss for demonstration
        # later, this could be cross-entropy for move prediction or a more complex loss for distillation
        self.criterion = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.student_model(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.student_model(x)
        loss = self.criterion(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.student_model.parameters(), lr=1e-3)
        return optimizer

def create_dummy_data(num_samples=1000, input_size=768, output_size=1):
    # generates random board states and evaluation targets for demonstration
    inputs = torch.randn(num_samples, input_size)
    targets = torch.randn(num_samples, output_size) * 10 # scale targets a bit
    return TensorDataset(inputs, targets)

if __name__ == "__main__":
    import torch
    import lightning as L

    input_dim = 77
    output_dim = 1

    student_model = StudentChessModel(input_size=input_dim, output_size=output_dim)
    lit_model = LitChessModel(student_model)

    # Pick the right accelerator
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
    else:
        accelerator = "cpu"
        devices = 1

    # Dataloaders (consider smaller batch if you hit OOM on GPU)
    train_dataloader, val_dataloader = get_dataloaders('', '', batch_size=1024)

    # Let Lightning manage devices; mixed precision helps on GPU
    trainer = L.Trainer(
        max_epochs=5,
        enable_progress_bar=True,
        accelerator=accelerator,
        devices=devices,
        precision="16-mixed" if accelerator in ("gpu", "mps") else "32-true",
        log_every_n_steps=1,
    )

    # Optional: log the actual device during training
    class PrintDeviceCallback(L.Callback):
        def on_train_start(self, trainer, pl_module):
            print(f"Lightning placed the model on: {pl_module.device}")

    trainer.callbacks.append(PrintDeviceCallback())

    trainer.fit(lit_model, train_dataloader, val_dataloader)

    print("student model dummy training complete!")
    print("next, we'll look into integrating the actual deepmind data.")
