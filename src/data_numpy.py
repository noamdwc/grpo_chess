import numpy as np
import torch
from torch.utils.data import Dataset


class NumpyChessDataset(Dataset):
    """Dataset for chess data stored in numpy .npy files."""
    def __init__(self, sequences_path):
        super().__init__()
        self.sequences = np.load(sequences_path)


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        board, state_value = sequence[:-1], sequence[-1]

        # Convert to torch tensors
        board_tensor = torch.from_numpy(board).float()
        state_value_tensor = torch.tensor([state_value], dtype=torch.float32)

        return board_tensor, state_value_tensor
