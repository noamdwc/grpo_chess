from typing import Any, Dict, List, Optional, Tuple
import os
import traceback
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from src.chess.policy_player import PolicyPlayer, PolicyConfig
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
                 searcher_cfg: Optional[Any] = None,
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

    def _make_policy(self, model: nn.Module) -> PolicyPlayer:
        """Create a policy player.
        
        Args:
            model: Neural network model
            
        Returns:
            Policy player, optionally wrapped with trajectory search
        """
        return PolicyPlayer(model, cfg=self.policy_cfg)

    def _make_stockfish(self) -> StockfishPlayer:
        """Create a Stockfish player with default configuration.
        
        Returns:
            Stockfish player instance
        """
        # Keep eval engine separate from reward/teacher-forcing engines.
        return StockfishPlayer(self.default_stockfish_cfg, engine_name=f"eval_engine_{os.getpid()}")

    @staticmethod
    def _safe_exception_message(exc: BaseException) -> str:
        """Best-effort stringification that won't raise during error handling."""
        try:
            return str(exc)
        except Exception:
            return f"<unprintable {type(exc).__name__}>"

    @staticmethod
    def _safe_traceback_text(exc: BaseException) -> str:
        """Best-effort traceback rendering that avoids recursive crash loops."""
        if isinstance(exc, RecursionError):
            return "RecursionError during Stockfish evaluation; skipping traceback rendering."
        try:
            return traceback.format_exc()
        except Exception as traceback_exc:
            traceback_msg = Evaluator._safe_exception_message(traceback_exc)
            return f"Failed to render traceback safely: {traceback_msg}"

    def single_evaluation(self, model: nn.Module) -> Tuple[Dict, PolicyPlayer, List[str]]:
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

    def evaluate_and_log(
        self,
        pl_module: pl.LightningModule,
        model: nn.Module,
        metric_prefix: str = "eval_stockfish",
    ) -> Optional[Dict]:
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
                    err_msg = self._safe_exception_message(e)
                    print(f"Stockfish eval attempt {attempt}/{max_attempts} failed: {err_msg}")
                    print(self._safe_traceback_text(e))
                    # Reset eval engine and retry on transient engine failures.
                    StockfishManager.close(f"eval_engine_{os.getpid()}")
                    # RecursionError tends to cascade in traceback/rendering stacks.
                    # Treat it as non-transient and fail this eval cycle quickly.
                    if isinstance(e, RecursionError):
                        break
            if results is None:
                return None
        finally:
            if was_training:
                pl_module.train()

        games = max(int(results["games"]), 1)
        wins = float(results["wins"])
        draws = float(results["draws"])
        losses = float(results["losses"])

        pl_module.log(f"{metric_prefix}/score", results["score"], prog_bar=True)
        pl_module.log(f"{metric_prefix}/elo_diff", results["elo_diff_vs_stockfish_approx"], prog_bar=True)
        pl_module.log(f"{metric_prefix}/games", float(results["games"]))
        pl_module.log(f"{metric_prefix}/wins", wins)
        pl_module.log(f"{metric_prefix}/draws", draws)
        pl_module.log(f"{metric_prefix}/losses", losses)
        pl_module.log(f"{metric_prefix}/win_rate", wins / games)
        pl_module.log(f"{metric_prefix}/draw_rate", draws / games)
        pl_module.log(f"{metric_prefix}/loss_rate", losses / games)
        print(
            f"[StockfishEvalCallback:{metric_prefix}] Results: "
            f"score={results['score']:.4f}, elo_diff={results['elo_diff_vs_stockfish_approx']:.1f}, "
            f"W/D/L={int(wins)}/{int(draws)}/{int(losses)} ({results['games']} games)"
        )

        for reason, cnt in results["termination_reasons"].items():
            pl_module.log(f"{metric_prefix}/term_{reason}", cnt / games)

        if pgns and pl_module.logger and hasattr(pl_module.logger, "experiment"):
            try:
                import wandb
                combined_pgn = "\n\n".join(pgns)
                pl_module.logger.experiment.log({
                    f"{metric_prefix}/pgns": wandb.Html(f"<pre>{combined_pgn}</pre>"),
                    f"{metric_prefix}/pgn_text": combined_pgn,
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

    def __init__(
        self,
        evaluator: Evaluator,
        every_n_epochs: int = 1,
        model_attr: str = "model",
        metric_prefix: str = "eval_stockfish",
        run_on_fit_start: bool = False,
        init_metric_prefix: str = "eval_stockfish_init",
    ):
        if every_n_epochs < 1:
            raise ValueError(f"every_n_epochs must be >= 1, got {every_n_epochs}")
        self.evaluator = evaluator
        self.every_n_epochs = every_n_epochs
        self.model_attr = model_attr
        self.metric_prefix = metric_prefix
        self.run_on_fit_start = run_on_fit_start
        self.init_metric_prefix = init_metric_prefix
        self._attempts = 0
        self._successes = 0
        self._failures = 0

    def _log_failure_fallback(self, pl_module: pl.LightningModule, metric_prefix: str) -> None:
        """Log fallback values so monitored metrics always exist."""
        pl_module.log(f"{metric_prefix}/score", -1.0, prog_bar=True)
        pl_module.log(f"{metric_prefix}/elo_diff", -10_000.0, prog_bar=False)
        pl_module.log(f"{metric_prefix}/games", 0.0)
        pl_module.log(f"{metric_prefix}/wins", 0.0)
        pl_module.log(f"{metric_prefix}/draws", 0.0)
        pl_module.log(f"{metric_prefix}/losses", 0.0)
        pl_module.log(f"{metric_prefix}/win_rate", 0.0)
        pl_module.log(f"{metric_prefix}/draw_rate", 0.0)
        pl_module.log(f"{metric_prefix}/loss_rate", 1.0)

    def _log_metrics_direct(
        self,
        trainer: pl.Trainer,
        metric_prefix: str,
        results: Optional[Dict],
    ) -> None:
        """Log metrics directly via trainer.logger for hooks where pl_module.log is disallowed."""
        if results is None:
            metrics = {
                f"{metric_prefix}/score": -1.0,
                f"{metric_prefix}/elo_diff": -10_000.0,
                f"{metric_prefix}/games": 0.0,
                f"{metric_prefix}/wins": 0.0,
                f"{metric_prefix}/draws": 0.0,
                f"{metric_prefix}/losses": 0.0,
                f"{metric_prefix}/win_rate": 0.0,
                f"{metric_prefix}/draw_rate": 0.0,
                f"{metric_prefix}/loss_rate": 1.0,
            }
        else:
            games = max(int(results["games"]), 1)
            wins = float(results["wins"])
            draws = float(results["draws"])
            losses = float(results["losses"])
            metrics = {
                f"{metric_prefix}/score": float(results["score"]),
                f"{metric_prefix}/elo_diff": float(results["elo_diff_vs_stockfish_approx"]),
                f"{metric_prefix}/games": float(results["games"]),
                f"{metric_prefix}/wins": wins,
                f"{metric_prefix}/draws": draws,
                f"{metric_prefix}/losses": losses,
                f"{metric_prefix}/win_rate": wins / games,
                f"{metric_prefix}/draw_rate": draws / games,
                f"{metric_prefix}/loss_rate": losses / games,
            }
            for reason, cnt in results["termination_reasons"].items():
                metrics[f"{metric_prefix}/term_{reason}"] = float(cnt) / games

        if trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=trainer.global_step)
        else:
            print(f"[StockfishEvalCallback:{metric_prefix}] No trainer.logger; metrics={metrics}")

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.run_on_fit_start:
            return
        print(
            f"[StockfishEvalCallback:{self.init_metric_prefix}] Running baseline evaluation at fit start",
            flush=True,
        )
        model = getattr(pl_module, self.model_attr)
        results = None
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                with torch.no_grad():
                    results, _, _ = self.evaluator.single_evaluation(model)
                break
            except Exception as e:
                err_msg = Evaluator._safe_exception_message(e)
                print(f"Stockfish baseline eval attempt {attempt}/{max_attempts} failed: {err_msg}")
                print(Evaluator._safe_traceback_text(e))
                StockfishManager.close(f"eval_engine_{os.getpid()}")
                if isinstance(e, RecursionError):
                    break

        self._log_metrics_direct(trainer, self.init_metric_prefix, results)
        if results is None:
            print(
                f"[StockfishEvalCallback:{self.init_metric_prefix}] Baseline evaluation failed; "
                "logged fallback metrics via trainer.logger.",
                flush=True,
            )

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        self._attempts += 1
        print(
            f"[StockfishEvalCallback:{self.metric_prefix}] Running evaluation at epoch {trainer.current_epoch + 1}",
            flush=True,
        )
        model = getattr(pl_module, self.model_attr)
        results = self.evaluator.evaluate_and_log(pl_module, model, metric_prefix=self.metric_prefix)
        if results is None:
            self._failures += 1
            # Keep monitored metrics present even when eval fails, so
            # ModelCheckpoint(monitor=...) does not crash on missing keys.
            self._log_failure_fallback(pl_module, self.metric_prefix)
            print(
                f"[StockfishEvalCallback:{self.metric_prefix}] Evaluation failed; "
                "logged fallback metrics for checkpoint compatibility.",
                flush=True,
            )
        else:
            self._successes += 1

        # Always log callback health so we can distinguish "not called" vs "called but failed".
        pl_module.log(f"{self.metric_prefix}/callback_attempts", float(self._attempts), on_step=False, on_epoch=True)
        pl_module.log(f"{self.metric_prefix}/callback_successes", float(self._successes), on_step=False, on_epoch=True)
        pl_module.log(f"{self.metric_prefix}/callback_failures", float(self._failures), on_step=False, on_epoch=True)


def evaluate_with_and_without_thinking(
    *,
    model,
    n_games: int,
    stockfish_level: int,
    seed: int,
) -> dict[str, float]:
    """Run eval with thinking enabled and disabled, and return think_delta."""
    score_with = _play_stockfish_match(model, n_games, stockfish_level, seed, thinking=True)
    score_without = _play_stockfish_match(model, n_games, stockfish_level, seed, thinking=False)
    return {
        "score_with_think": score_with,
        "score_no_think": score_without,
        "think_delta": score_with - score_without,
    }


def _play_stockfish_match(model, n_games: int, stockfish_level: int, seed: int, thinking: bool) -> float:
    """Play n_games vs Stockfish and return the model score in [0, 1]."""
    import random
    import chess

    from src.reasoning.sampler import rollout_batch
    from src.searchless_chess_imports import ACTION_TO_MOVE

    total = 0.0
    random.seed(seed)
    torch.manual_seed(seed)
    stockfish_cfg = StockfishConfig(skill_level=stockfish_level)
    stockfish = StockfishPlayer(stockfish_cfg, engine_name=f"think_delta_eval_{os.getpid()}_{int(thinking)}")
    try:
        for game_idx in range(n_games):
            board = chess.Board()
            model_color = chess.WHITE if game_idx % 2 == 0 else chess.BLACK
            stockfish_failed = False
            while not board.is_game_over(claim_draw=True):
                if board.turn == model_color:
                    result = rollout_batch(
                        model=model,
                        root_fens=[board.fen()],
                        k_samples=1,
                        min_think_tokens=0 if not thinking else 2,
                        max_think_tokens=0 if not thinking else 12,
                        rollout_temperature=0.0001,
                    )
                    final_token = int(
                        result.token_sequences[0, int(result.final_move_positions[0].item())].item()
                    )
                    board.push(chess.Move.from_uci(ACTION_TO_MOVE[final_token]))
                else:
                    reply = stockfish.act(board)
                    if reply is None:
                        stockfish_failed = True
                        break
                    board.push(reply)
            if stockfish_failed:
                total += 1.0
                continue
            outcome = board.outcome(claim_draw=True)
            if outcome is None or outcome.winner is None:
                total += 0.5
            elif outcome.winner == model_color:
                total += 1.0
    finally:
        stockfish.close()
    return total / n_games
