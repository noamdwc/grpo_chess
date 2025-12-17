from typing import Dict, Optional, Tuple
import torch.nn as nn

from src.grpo_self_play.chess.policy_player import PolicyPlayer, PolicyConfig
from src.grpo_self_play.chess.searcher import TrajectorySearcher, SearchConfig
from src.grpo_self_play.chess.stockfish import StockfishPlayer, StockfishConfig
from src.grpo_self_play.eval_utils import EvalConfig, evaluate_policy_vs_stockfish



class Evaluator:
    '''
    Evaluate a chess model by playing against Stockfish.
    '''
    def __init__(self,
                 eval_cfg: EvalConfig = EvalConfig(),
                 policy_cfg: PolicyConfig = PolicyConfig(),
                 searcher_cfg: Optional[SearchConfig] = None,
                 stockfish_cfg: StockfishConfig = StockfishConfig()):
        self.eval_cfg = eval_cfg
        self.policy_cfg = policy_cfg
        self.searcher_cfg = searcher_cfg
        self.default_stockfish_cfg = stockfish_cfg

    def _make_policy(self, model: nn.Module) -> PolicyPlayer | TrajectorySearcher:
        policy = PolicyPlayer(model, cfg=self.policy_cfg)
        if self.searcher_cfg is not None:
            policy = TrajectorySearcher(policy, cfg=self.searcher_cfg)
        return policy

    def _make_stockfish(self) -> StockfishPlayer:
        return StockfishPlayer(self.default_stockfish_cfg)

    def single_evaluation(self, model: nn.Module) -> Tuple[Dict, PolicyPlayer | TrajectorySearcher]:
        '''
        Evaluate the model by playing games against Stockfish.
        Returns a tuple of (results_dict, policy_or_searcher).
        '''
        stockfish_player = self._make_stockfish()
        policy = self._make_policy(model)
        results, policy_or_searcher = evaluate_policy_vs_stockfish(
            policy,
            stockfish_player,
            self.eval_cfg,
        )
        return results, policy_or_searcher

    def eval_ladder(self, model: nn.Module) -> Dict[int, float]:
        policy = self._make_policy(model)
        results = {}
        for skill in [1, 3, 5, 8, 10]:
            stockfish_cfg = StockfishConfig(
                path=self.default_stockfish_cfg.path,
                skill_level=skill,
                movetime_ms=self.default_stockfish_cfg.movetime_ms,
            )
            stockfish_player = StockfishPlayer(stockfish_cfg)

            r, policy_wrapper  = evaluate_policy_vs_stockfish(
                policy,
                stockfish_player,
                self.eval_cfg,
            )
            results[skill] = r["score"]
            print("skill", skill, r)
            print('policy stats', policy_wrapper.stats)
        return results




