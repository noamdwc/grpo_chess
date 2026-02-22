from typing import Dict, List, Optional, Tuple
import os
import traceback
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

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
        # Keep eval engine separate from reward/teacher-forcing engines.
        return StockfishPlayer(self.default_stockfish_cfg, engine_name=f"eval_engine_{os.getpid()}")

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

    def evaluate_and_log(self, pl_module: pl.LightningModule, model: nn.Module) -> Optional[Dict]:
        """Run evaluation and log results to the Lightning module's logger.

        Returns results dict or None if evaluation failed.
        """
        was_training = pl_module.training
        pl_module.eval()
        results = None
        pgns: List[str] = []
        max_attempts = 3
        try:
            for attempt in range(1, max_attempts + 1):
                try:
                    with torch.no_grad():
                        results, _, pgns = self.single_evaluation(model)
                    break
                except Exception as e:
                    print(f"Stockfish eval attempt {attempt}/{max_attempts} failed: {e}")
                    print(traceback.format_exc())
                    # Reset eval engine and retry on transient engine failures.
                    StockfishManager.close(f"eval_engine_{os.getpid()}")
            if results is None:
                return None
        finally:
            if was_training:
                pl_module.train()

        games = max(int(results["games"]), 1)
        wins = float(results["wins"])
        draws = float(results["draws"])
        losses = float(results["losses"])

        pl_module.log("eval_stockfish/score", results["score"], prog_bar=True)
        pl_module.log("eval_stockfish/elo_diff", results["elo_diff_vs_stockfish_approx"], prog_bar=True)
        pl_module.log("eval_stockfish/games", float(results["games"]))
        pl_module.log("eval_stockfish/wins", wins)
        pl_module.log("eval_stockfish/draws", draws)
        pl_module.log("eval_stockfish/losses", losses)
        pl_module.log("eval_stockfish/win_rate", wins / games)
        pl_module.log("eval_stockfish/draw_rate", draws / games)
        pl_module.log("eval_stockfish/loss_rate", losses / games)
        print(
            "[StockfishEvalCallback] Results: "
            f"score={results['score']:.4f}, elo_diff={results['elo_diff_vs_stockfish_approx']:.1f}, "
            f"W/D/L={int(wins)}/{int(draws)}/{int(losses)} ({results['games']} games)"
        )

        for reason, cnt in results["termination_reasons"].items():
            pl_module.log(f"eval_stockfish/term_{reason}", cnt / games)

        if pgns and pl_module.logger and hasattr(pl_module.logger, "experiment"):
            try:
                import wandb
                combined_pgn = "\n\n".join(pgns)
                pl_module.logger.experiment.log({
                    "eval_stockfish/pgns": wandb.Html(f"<pre>{combined_pgn}</pre>"),
                    "eval_stockfish/pgn_text": combined_pgn,
                })
            except Exception:
                pass

        return results

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


class StockfishEvalCallback(Callback):
    """Lightning callback that evaluates the model against Stockfish periodically."""

    def __init__(self, evaluator: Evaluator, every_n_epochs: int = 1, model_attr: str = "model"):
        if every_n_epochs < 1:
            raise ValueError(f"every_n_epochs must be >= 1, got {every_n_epochs}")
        self.evaluator = evaluator
        self.every_n_epochs = every_n_epochs
        self.model_attr = model_attr
        self._attempts = 0
        self._successes = 0
        self._failures = 0

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        self._attempts += 1
        print(
            f"[StockfishEvalCallback] Running evaluation at epoch {trainer.current_epoch + 1}",
            flush=True,
        )
        model = getattr(pl_module, self.model_attr)
        results = self.evaluator.evaluate_and_log(pl_module, model)
        if results is None:
            self._failures += 1
        else:
            self._successes += 1

        # Always log callback health so we can distinguish "not called" vs "called but failed".
        pl_module.log("eval_stockfish/callback_attempts", float(self._attempts), on_step=False, on_epoch=True)
        pl_module.log("eval_stockfish/callback_successes", float(self._successes), on_step=False, on_epoch=True)
        pl_module.log("eval_stockfish/callback_failures", float(self._failures), on_step=False, on_epoch=True)
