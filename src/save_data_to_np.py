import os
import sys

import numpy as np

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, path)

from searchless_chess.src import config as config_lib
from searchless_chess.src.data_loader import build_data_loader

def save_dataset_as_numpy(config: config_lib.DataConfig, output_path: str, max_size: int = None):
    """Save chess dataset as uint8 numpy arrays."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
 # Save as separate .npy files for fastest loading
    base_path = output_path.replace('.npz', '')
    sequences_path = f"{base_path}_sequences.npy"
    masks_path = f"{base_path}_masks.npy"
    
    # Build the data loader
    data_loader = build_data_loader(config)
    
    sequences_list = []
    loss_masks_list = []
    
    print(f"Processing {config.policy} dataset...")
    last_batch_idx = 0
    if_finished_loading = False
    while not if_finished_loading:
        try:
            for batch_idx, (sequences_batch, loss_masks_batch) in enumerate(data_loader):
                if max_size is not None and batch_idx >= max_size:
                    if_finished_loading = True
                    break
                if batch_idx < last_batch_idx:
                    continue
                sequences_list.append(sequences_batch)
                loss_masks_list.append(loss_masks_batch)
                last_batch_idx = batch_idx
                if batch_idx % 100 == 0:
                    print(f"Processed {batch_idx} batches")
            if_finished_loading = True
        except Exception as e:
            print(f"Exception occurred: {e}. Restarting data loader...")
            data_loader = build_data_loader(config)

    print("Finished loading all batches.")
    # Concatenate all sequences
    all_sequences = np.concatenate(sequences_list, axis=0)
    print(f"Total samples: {all_sequences.shape[0]}")

    # Convert to uint8 (sequences are already int32, so we need to check range)
    # Check if values fit in uint8 range
    seq_min, seq_max = all_sequences.min(), all_sequences.max()
    print(f"Sequence value range: {seq_min} to {seq_max}")
    
    if seq_min >= 0 and seq_max <= 255:
        sequences_uint8 = all_sequences.astype(np.uint8)
        print("Converted sequences to uint8")
    else:
        print(f"Warning: Values don't fit in uint8 range. Keeping as int32.")
        sequences_uint8 = all_sequences
    
    # Save the arrays
    np.save(sequences_path, sequences_uint8)
    print(f"Saved sequences to {sequences_path}")
    print(f"Sequences dtype: {sequences_uint8.dtype}")

    # Concatenate all loss masks
    all_loss_masks = np.concatenate(loss_masks_list, axis=0)
    print(f"Sequence shape: {all_sequences.shape}")
    np.save(masks_path, all_loss_masks.astype(np.uint8))
    
    

    print(f"Saved masks to {masks_path}")
    print(f"Loss masks dtype: {all_loss_masks.dtype}")
    print('----------------------------------')
    print(f"Saved dataset to {output_path}")

# Usage example:
if __name__ == "__main__":
    # Create config for state_value training data
    config = config_lib.DataConfig(
        policy='state_value',
        split='train',
        batch_size=4096,  # adjust as needed
        shuffle=False,  # set to False to maintain order
        worker_count=0,
        seed=42,
        num_return_buckets=128  # adjust based on your config
    )
    
    # Save as numpy array
    save_dataset_as_numpy(
        config=config,
        output_path='../data/numpy_arrays/state_value_train.npy',
        max_size=56e2  # limit to first 1 million batches for testing
    )