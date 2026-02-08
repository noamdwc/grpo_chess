from typing import Dict, List, Optional, Tuple
from chess import engine
import torch.nn as nn

from src.chess.policy_player import PolicyPlayer, PolicyConfig
from src.chess.searcher import TrajectorySearcher, SearchConfig
from src.chess.stockfish import StockfishPlayer, StockfishConfig, StockfishManager
from src.eval_utils import EvalConfig, evaluate_policy_vs_stockfish



class Evaluator:
    """Evaluate a chess model by playing against Stockfish.
    
    Handles evaluation of chess policies against Stockfish at various skill levels.
    Supports both single evaluations and skill ladder evaluations.
    """
    def __init__(self,
                 eval_cfg: EvalConfig = EvalConfig(),
                 policy_cfg: PolicyConfig = PolicyConfig(),
                 searcher_cfg: Optional[SearchConfig] = None,
                 stockfish_cfg: StockfishConfig = StockfishConfig()):
        """
        Initialize evaluator.
        
        Args:
            eval_cfg: Evaluation configuration (number of games, etc.)
            policy_cfg: Policy player configuration
            searcher_cfg: Optional search configuration for tree search
            stockfish_cfg: Stockfish engine configuration
        """
        self.eval_cfg = eval_cfg
        self.policy_cfg = policy_cfg
        self.searcher_cfg = searcher_cfg
        self.default_stockfish_cfg = stockfish_cfg

    def _make_policy(self, model: nn.Module) -> PolicyPlayer | TrajectorySearcher:
        """Create a policy player (optionally wrapped with search).
        
        Args:
            model: Neural network model
            
        Returns:
            Policy player, optionally wrapped with trajectory search
        """
        policy = PolicyPlayer(model, cfg=self.policy_cfg)
        if self.searcher_cfg is not None:
            policy = TrajectorySearcher(policy, cfg=self.searcher_cfg)
        return policy

    def _make_stockfish(self) -> StockfishPlayer:
        """Create a Stockfish player with default configuration.
        
        Returns:
            Stockfish player instance
        """
        return StockfishPlayer(self.default_stockfish_cfg)

    def single_evaluation(self, model: nn.Module) -> Tuple[Dict, PolicyPlayer | TrajectorySearcher, List[str]]:
        """Evaluate the model by playing games against Stockfish.

        Args:
            model: Neural network model to evaluate

        Returns:
            Tuple of (results_dict, policy_or_searcher, pgns)
            pgns is a list of PGN strings for all games played
        """
        stockfish_player = self._make_stockfish()
        policy = self._make_policy(model)
        results, policy_or_searcher, pgns = evaluate_policy_vs_stockfish(
            policy,
            stockfish_player,
            self.eval_cfg,
        )
        return results, policy_or_searcher, pgns

    def eval_ladder(self, model: nn.Module) -> Dict[int, float]:
        """Evaluate model against Stockfish at multiple skill levels.
        
        Args:
            model: Neural network model to evaluate
            
        Returns:
            Dictionary mapping skill level to win rate score
        """
        policy = self._make_policy(model)
        results = {}
        skill_levels = [1, 3, 5, 8, 10]
        for skill in skill_levels:
            stockfish_cfg = StockfishConfig(
                path=self.default_stockfish_cfg.path,
                skill_level=skill,
                movetime_ms=self.default_stockfish_cfg.movetime_ms,
            )
            engine_name = f"stockfish_skill_{skill}"
            stockfish_player = StockfishPlayer(stockfish_cfg, engine_name=engine_name)

            try:
                r, policy_wrapper, _ = evaluate_policy_vs_stockfish(
                    policy,
                    stockfish_player,
                    self.eval_cfg,
                )
                results[skill] = r["score"]
                print(f"Skill {skill}: {r}")
                if hasattr(policy_wrapper, 'stats'):
                    print(f'Policy stats: {policy_wrapper.stats}')
            except Exception as e:
                print(f"Error evaluating at skill {skill}: {e}")
                results[skill] = 0.0
            finally:
                StockfishManager.close(engine_name)  # Close engine to free resources
        return results




