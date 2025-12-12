import random
from typing import Dict, Optional, Tuple
import torch

from src.grpo_self_play.chess.policy_player import PolicyPlayer, PolicyConfig
from src.grpo_self_play.chess.searcher import TrajectorySearcher, SearchConfig
from src.grpo_self_play.chess.chess_engine import StockfishPlayer, StockfishConfig
from src.grpo_self_play.eval_utils import EvalConfig, play_one_game, estimate_elo_diff, evaluate_policy_vs_stockfish



class Evaluator:
    '''
    Evalue a chess model by playing against Stockfish.
    '''
    def __init__(self,
                 model: torch.nn.Module,
                 device=None,
                 eval_cfg: EvalConfig = EvalConfig(),
                 policy_cfg: PolicyConfig = PolicyConfig(),
                 searcher_cfg: Optional[SearchConfig] = None,
                 stockfish_cfg: StockfishConfig = StockfishConfig()):
        self.model = model
        self.device = device
        self.eval_cfg = eval_cfg
        self.policy = PolicyPlayer(model, device=device, cfg=policy_cfg)
        if searcher_cfg is not None:
            self.policy = TrajectorySearcher(self.policy, cfg=searcher_cfg)
        self.defualt_stockfish = StockfishPlayer(stockfish_cfg)



    def _single_evaluation(self) -> Tuple[Dict, PolicyPlayer | TrajectorySearcher]:
        '''
        Evaluate the model by playing games against Stockfish.
        Returns a tuple of (results_dict, policy_or_searcher).
        '''
        results, policy_or_searcher = evaluate_policy_vs_stockfish(
            self.policy,
            self.defualt_stockfish,
            self.eval_cfg,
        )
        return results, policy_or_searcher


    def eval_ladder(self):
        results = {}
        for skill in [1, 3, 5, 8, 10]:
            stockfish_cfg = StockfishConfig(
                path=self.defualt_stockfish.cfg.path,
                skill_level=skill,
                movetime_ms=self.defualt_stockfish.cfg.movetime_ms,
            )
            eval_cfg = EvalConfig(
                games=50,
                seed=self.eval_cfg.seed,
                randomize_opening=False
            )
            policy_cfg = PolicyConfig(temperature=0.3, greedy=False)

            r, policy = evaluate_policy_vs_stockfish(
                self.policy,
                self.defualt_stockfish,
                self.eval_cfg,
            )
            results[skill] = r["score"]
            print("skill", skill, r)
            print('policy stats', policy.stats)
        return results




