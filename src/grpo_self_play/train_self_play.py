from torch.utils.data import DataLoader

from src.grpo_self_play.trainer import get_trainer
from src.grpo_self_play.models import ChessTransformerConfig
from src.grpo_self_play.chess.chess_logic import ChessStartStatesDataset
from src.grpo_self_play.grpo_logic.model import GRPOChessTransformer, GRPOConfig


GRPO_CONFIG = GRPOConfig(lr=1e-6, num_trajectories=8, trajectory_depth=32)
TRANSFORMER_CONFIG = ChessTransformerConfig()
NUM_EPOCHS = 5000
BATCH_SIZE = 32
STEPS_PER_EPOCH = 1024


def train():
    trainer = get_trainer(num_epochs=NUM_EPOCHS)
    dataset = ChessStartStatesDataset(max_steps=STEPS_PER_EPOCH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=2)
    model = GRPOChessTransformer(TRANSFORMER_CONFIG, GRPO_CONFIG)#, num_trajectories=16, trajectory_depth=32))

    print("Starting Training with WandB Tracking...")
    trainer.fit(model, dataloader)