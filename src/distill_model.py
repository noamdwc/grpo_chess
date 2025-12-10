from cgi import test
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

# set the PYTHONPATH correctly for imports
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, path)

# import our new pytorch model and data components
from transformer_pytorch import PytorchStateValueModel
from data_numpy import NumpyChessDataset


class LitDistillationModule(L.LightningModule):
    def __init__(self, teacher_input_dim: int, student_input_dim: int, student_hidden_dim: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        # teacher model (placeholder: in a real scenario, this would load converted 9M deepmind weights)
        self.teacher_model = PytorchStateValueModel(input_dim=teacher_input_dim, output_dim=1)
        # for a true teacher, you'd load the checkpoint:
        # self.teacher_model.load_state_dict(torch.load('path/to/converted/9M_state_value_pytorch_weights.pth'))
        self.teacher_model.eval() # teacher is frozen, not trained
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # student model
        self.student_model = PytorchStateValueModel(
            input_dim=student_input_dim,
            hidden_dim=student_hidden_dim,
            output_dim=1 # state value output is a scalar
        )

    def forward(self, x):
        return self.student_model(x)

    def training_step(self, batch, batch_idx):
        device = next(self.parameters()).device
        features, true_state_values = batch
        features, true_state_values = features.to(device, non_blocking=True), true_state_values.to(device, non_blocking=True)
        # teacher's prediction (no gradient updates for teacher)
        with torch.no_grad():
            teacher_predictions = self.teacher_model(features)

        # student's prediction
        student_predictions = self.student_model(features)

        # 1. distillation loss (e.g., MSE between student and teacher outputs)
        distillation_loss = F.mse_loss(student_predictions, teacher_predictions)

        # 2. task-specific loss (e.g., MSE between student and true state values)
        task_loss = F.mse_loss(student_predictions, true_state_values)

        # combine losses (you can adjust the weights as needed)
        total_loss = distillation_loss + task_loss

        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_distillation_loss", distillation_loss, on_step=True, on_epoch=True)
        self.log("train_task_loss", task_loss, on_step=True, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        features, true_state_values = batch
        student_predictions = self.student_model(features)
        val_loss = F.mse_loss(student_predictions, true_state_values)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.student_model.parameters(), lr=self.hparams.lr)
        return optimizer


if __name__ == "__main__":

    # define where your downloaded data is
    # adjust these paths to where your 'data/train' and 'data/test' directories are
    data_path = os.path.join('..', 'data')
    
    # build datasets
    train_dataset = NumpyChessDataset(os.path.join(data_path, 'numpy_arrays', 'state_value_train_sequences.npy'))
    test_dataset = NumpyChessDataset(os.path.join(data_path, 'numpy_arrays', 'state_value_train_sequences.npy'))


    # hyperparameters for distillation
    teacher_input_dim = 77 # assuming this from the deepmind models
    student_input_dim = 77 # student processes same input format
    student_hidden_dim = 64 # significantly smaller than teacher
    learning_rate = 1e-3
    batch_size = 4096
    num_epochs = 5

    # build dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )


    # instantiate the distillation module
    model = LitDistillationModule(
        teacher_input_dim=teacher_input_dim,
        student_input_dim=student_input_dim,
        student_hidden_dim=student_hidden_dim,
        lr=learning_rate
    )

    # initialize a lightning trainer
    # if you've switched to a GPU machine, lightning will automatically use it
    trainer = L.Trainer(
        max_epochs=num_epochs,
        devices=1,
        accelerator="gpu",
        precision="16-mixed",
        accumulate_grad_batches=8, 
        num_sanity_val_steps=0,
        )

    # train the model
    trainer.fit(model, train_loader, val_loader)

    print("distillation training complete!")
    # you can save your student model here
    torch.save(model.student_model.state_dict(), "distilled_student_model.pth")