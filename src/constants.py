"""Constants used across the GRPO self-play module."""

# Sequence length for tokenized FEN strings
SEQUENCE_LENGTH = 77

# Default training hyperparameters
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_NUM_TRAJECTORIES = 4
DEFAULT_TRAJECTORY_DEPTH = 5
DEFAULT_CLIP_RATIO = 0.2
DEFAULT_KL_COEF = 0.01

# Default evaluation settings
DEFAULT_EVAL_GAMES = 50
DEFAULT_EVAL_MAX_PLIES = 400