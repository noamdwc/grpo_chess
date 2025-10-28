import os
import torch
from torch.cpu import is_available
from torch.utils.data import DataLoader
import numpy as np
import grain.python as pygrain
# import build_data_loader and config_lib
from searchless_chess.src.data_loader import build_data_loader
from searchless_chess.src import config as config_lib
from searchless_chess.src import tokenizer as searchless_chess_tokenizer


# the _extract_features_from_proto, _extract_value_from_proto, and StateValueDataset
# are no longer needed as build_data_loader from searchless_chess handles data processing.
class PyGrainToTorchIterator:
    def __init__(self, pygrain_loader, num_records, batch_size, drop_remainder=True, max_size=None):
        self.pygrain_loader = pygrain_loader
        self.num_records = num_records
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder

        self.size = 0
        if self.drop_remainder:
            self.size = self.num_records // self.batch_size
        self.size = (self.num_records + self.batch_size - 1) // self.batch_size
        if max_size is not None:
            self.size = min(self.size, max_size)


    def __iter__(self):
        for batch_np, loss_mask_np in self.pygrain_loader:
            features_tensor = torch.from_numpy(batch_np).float().pin_memory()
            inputs = features_tensor[:, :searchless_chess_tokenizer.SEQUENCE_LENGTH]
            targets = features_tensor[:, searchless_chess_tokenizer.SEQUENCE_LENGTH:]
            yield inputs, targets

    def __len__(self):
        return self.size
        

def get_num_records(pygrain_loader: pygrain.DataLoader) -> int:
    # Access the private attribute _sampler to get num_records
    # This is a workaround since pygrain.DataLoader does not expose num_records directly
    try:
        return pygrain_loader._sampler._num_records
    except AttributeError:
        raise ValueError("Could not access num_records or sampler from pygrain DataLoader.")


def get_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    policy: str = 'state_value',
    num_return_buckets: int = 1,
    max_size: int = None,
):
    """
    builds train and validation dataloaders using pygrain's build_data_loader.

    args:
        train_data_dir: expected path for training data (e.g., used to infer "train" split).
        test_data_dir: expected path for testing data (e.g., used to infer "test" split).
        batch_size: batch size for the dataloaders.
        num_workers: number of worker processes for data loading.
        policy: the data policy to use (e.g., 'state_value', 'action_value').
        num_return_buckets: number of return buckets for value data.

    returns:
        a tuple of (train_loader, val_loader) wrapped to yield torch tensors.
    """
    # dataconfig expects 'split' like "train" or "test", not full paths.
    # the build_data_loader internally constructs the path '../data/{split}/{policy}_data.bag'.

    train_config = config_lib.DataConfig(
        split="train", # assumes data for training is in '../data/train/'
        policy=policy,
        num_return_buckets=num_return_buckets,
        batch_size=batch_size,
        shuffle=True,
        seed=0, # fixed seed for reproducibility, can be parameterized
        worker_count=num_workers,
        num_records=None, # use all records in the bag file
    )
    train_loader_pygrain = build_data_loader(train_config)

    val_config = config_lib.DataConfig(
        split="test", # assumes data for testing is in '../data/test/'
        policy=policy,
        num_return_buckets=num_return_buckets,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        worker_count=num_workers,
        num_records=None,
    )
    val_loader_pygrain = build_data_loader(val_config)

    # wrap pygrain dataloaders to yield torch tensors
    train_loader = PyGrainToTorchIterator(train_loader_pygrain,
                                          get_num_records(train_loader_pygrain),
                                          batch_size,
                                          max_size=max_size)
    val_loader = PyGrainToTorchIterator(val_loader_pygrain,
                                        get_num_records(val_loader_pygrain),
                                        batch_size,
                                        max_size=max_size)

    return train_loader, val_loader