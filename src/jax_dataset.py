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

from src.model_wrapper import create_predict_func
from huggingface_hub import snapshot_download

model_path = snapshot_download(
    repo_id="dbest-isi/searchless-chess-9M-selfplay",
    local_dir="./searchless_chess_model"
)
print(f"Model downloaded to: {model_path}")

# Add bundled code to Python path
sys.path.insert(0, f"{model_path}/searchless_chess_code")

import hf_model
import utils
import tokenizer

# Load the model
model = hf_model.SearchlessChessModel.from_pretrained(model_path)
model.predict = create_predict_func(model, utils, tokenizer)



data_path = os.path.join('..', 'data')

# build datasets
train_dataset = NumpyChessDataset(os.path.join(data_path, 'numpy_arrays', 'state_value_train_sequences.npy'))
test_dataset = NumpyChessDataset(os.path.join(data_path, 'numpy_arrays', 'state_value_train_sequences.npy'))


# hyperparameters for distillation
teacher_input_dim = 77 # assuming this from the deepmind models

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


