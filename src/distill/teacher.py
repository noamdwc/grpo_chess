from typing import List, Tuple
from src.model_wrapper import download_model, load_model
from huggingface_hub import snapshot_download
# ----------------------------------------------------------------------
# Teacher interface (DeepMind Searchless Chess)
# ----------------------------------------------------------------------


class TeacherEngine:
    """Abstract teacher interface."""

    def topk(self, fen: str, k: int) -> Tuple[List[str], List[float]]:
        """
        Given a FEN string, return:
            moves: list[str]  - UCI moves, length <= k
            probs: list[float] - same length, normalized to sum to 1
        """
        raise NotImplementedError


class SearchlessActionValueTeacher(TeacherEngine):
    """
    Placeholder wrapper around DeepMind's action_value model from:
    https://github.com/google-deepmind/searchless_chess :contentReference[oaicite:3]{index=3}

    You need to implement this class for your environment:

    - Clone the repo and install requirements:
        git clone https://github.com/google-deepmind/searchless_chess.git
        cd searchless_chess
        conda create --name searchless_chess python=3.10
        conda activate searchless_chess
        pip install -r requirements.txt

    - Download checkpoints:
        cd checkpoints
        ./download.sh
        cd ..

    - Make sure PYTHONPATH includes the repo root, e.g.:
        export PYTHONPATH=/path/to/searchless_chess:$PYTHONPATH

    Then modify __init__ and topk below to call the action_value model.
    """

    def __init__(self, agent: str = "270M", device: str = "gpu"):
        """
        agent: "9M", "136M", or "270M" (see README for available agents).
        device: "cpu" or "gpu" depending on your JAX install.
        """
        raise NotImplementedError(
            "TODO: Implement SearchlessActionValueTeacher using "
            "google-deepmind/searchless_chess (see comments in the class)."
        )

        # Example skeleton (you need to adjust to the real API):
        # from engines import neural_engines  # searchless_chess/src on PYTHONPATH
        # self.engine = neural_engines.NeuralEngine(
        #     agent=agent,
        #     device=device,
        #     head="action_value",  # if such a flag exists
        # )

    def topk(self, fen: str, k: int) -> Tuple[List[str], List[float]]:
        """
        Example of what you should implement, conceptually:

            1. Convert FEN to the model's tokenization (if needed).
            2. Ask the model for action-values for all legal moves.
            3. Convert action-values to probabilities (softmax).
            4. Return the top-K moves and probs.

        Pseudocode (replace with the actual engine call):

            action_values: Dict[str, float] = self.engine.get_action_values(fen)
            # action_values maps UCI move -> value (e.g. win prob or logit)
        """
        raise NotImplementedError("Implement topk(fen, k) using the DeepMind engine.")

        # Example logic (once you have a dict move -> value):
        # import numpy as np
        # if not action_values:
        #     return [], []
        # items = sorted(action_values.items(), key=lambda kv: kv[1], reverse=True)
        # top_items = items[:k]
        # moves, values = zip(*top_items)
        # values_arr = np.asarray(values, dtype=np.float32)
        # # Softmax over values (if they are not already probabilities)
        # exp = np.exp(values_arr - values_arr.max())
        # probs = exp / exp.sum()
        # return list(moves), probs.tolist()



class SearchlessChessHF(TeacherEngine):
    def __init__(self):
        model_path = download_model()
        self.model = load_model(model_path)

    def topk(self, fen: str, k: int) -> Tuple[List[str], List[float]]:
        pass