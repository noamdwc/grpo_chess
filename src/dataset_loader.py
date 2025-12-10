import torch
from torch.utils.data import Dataset
import numpy as np
import os

class DistillationDataset(Dataset):
    def __init__(self, npz_path):
        """
        Loads a compressed .npz file containing 'tokens' and 'logits'.
        """
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Dataset not found at {npz_path}")
            
        print(f"Loading dataset from {npz_path}...")
        data = np.load(npz_path)
        
        # Load tokens (Inputs)
        # Assuming they were saved as int16 or int32, convert to int64 for PyTorch Embedding
        self.tokens = torch.from_numpy(data['tokens'].astype(np.int64))
        
        # Load Logits (Targets)
        # We keep them as float16 in memory to save RAM, convert to float32 on the fly
        self.logits = torch.from_numpy(data['logits'])
        
        print(f"Loaded {len(self.tokens)} samples.")
        print(f"Input Shape: {self.tokens.shape}")
        print(f"Target Shape: {self.logits.shape}")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        # Returns: (Input Tokens, Teacher Logits)
        # Convert logits to float32 here for numerical stability in Loss calculation
        return self.tokens[idx], self.logits[idx].float()