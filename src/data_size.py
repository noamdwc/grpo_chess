import os
import sys
import numpy as np
from tqdm import tqdm

# set the PYTHONPATH correctly for imports
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, path)

from data_pytorch import get_dataloaders


train_data_path = os.path.join(os.getcwd(), 'data', 'train')
test_data_path = os.path.join(os.getcwd(), 'data', 'test')

# hyperparameters for distillation
teacher_input_dim = 77 # assuming this from the deepmind models
student_input_dim = 77 # student processes same input format
student_hidden_dim = 64 # significantly smaller than teacher
learning_rate = 1e-3
batch_size = 67072
num_epochs = 5

def main():
    train_loader, val_loader = get_dataloaders(batch_size=batch_size)
    # print(sum(x.shape[0] for (x, _) in tqdm(train_loader, desc="Training samples")))
    vals = [batch for batch in tqdm(val_loader, desc="Validation samples")]
    print(len(vals))
    print(vals)
    print(sum(vals))

if __name__ == "__main__":
    main()
# Example output:
# Training samples: 1000000
# Validation samples: 200000