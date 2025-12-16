from typing import Dict, Optional, Tuple
import torch

from src.grpo_self_play.chess.policy_player import PolicyPlayer, PolicyConfig
from src.grpo_self_play.chess.searcher import TrajectorySearcher, SearchConfig
from src.grpo_self_play.chess.stockfish import StockfishPlayer, StockfishConfig
from src.grpo_self_play.eval_utils import EvalConfig, evaluate_policy_vs_stockfish



class Evaluator:
    '''
    Evaluate a chess model by playing against Stockfish.
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
        self.default_stockfish = StockfishPlayer(stockfish_cfg)



    def _single_evaluation(self) -> Tuple[Dict, PolicyPlayer | TrajectorySearcher]:
        '''
        Evaluate the model by playing games against Stockfish.
        Returns a tuple of (results_dict, policy_or_searcher).
        '''
        results, policy_or_searcher = evaluate_policy_vs_stockfish(
            self.policy,
            self.default_stockfish,
            self.eval_cfg,
        )
        return results, policy_or_searcher


    def eval_ladder(self):
        results = {}
        for skill in [1, 3, 5, 8, 10]:
            stockfish_cfg = StockfishConfig(
                path=self.default_stockfish.cfg.path,
                skill_level=skill,
                movetime_ms=self.default_stockfish.cfg.movetime_ms,
            )
            stockfish_player = StockfishPlayer(stockfish_cfg)

            r, policy = evaluate_policy_vs_stockfish(
                self.policy,
                stockfish_player,
                self.eval_cfg,
            )
            results[skill] = r["score"]
            print("skill", skill, r)
            print('policy stats', policy.stats)
        return results




