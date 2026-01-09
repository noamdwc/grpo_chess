"""Training script for GRPO chess self-play."""
from torch.utils.data import DataLoader

from src.grpo_self_play.trainer import get_trainer
from src.grpo_self_play.models import ChessTransformerConfig
from src.grpo_self_play.chess.boards_dataset import ChessStartStatesDataset, ChessDatasetConfig
from src.grpo_self_play.grpo_logic.model import GRPOChessTransformer, GRPOConfig


# Training configuration
GRPO_CONFIG = GRPOConfig(lr=1e-6, num_trajectories=8, trajectory_depth=32)
TRANSFORMER_CONFIG = ChessTransformerConfig()
NUM_EPOCHS = 5000
BATCH_SIZE = 32
STEPS_PER_EPOCH = 1024


def train() -> None:
    """Main training function for GRPO chess self-play."""
    trainer = get_trainer(num_epochs=NUM_EPOCHS)
    dataset_config = ChessDatasetConfig(max_steps=STEPS_PER_EPOCH)
    dataset = ChessStartStatesDataset(config=dataset_config)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=2)
    model = GRPOChessTransformer(TRANSFORMER_CONFIG, GRPO_CONFIG)

    print("Starting Training with WandB Tracking...")
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    train()