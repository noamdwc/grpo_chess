import chess
import chess.engine
import warnings
from dataclasses import dataclass
from src.grpo_self_play.chess.chess_logic import ChessPlayer

DEFAULT_STOCKFISH_PATH = "/usr/games/stockfish"


@dataclass(frozen=True)
class StockfishConfig:
    path: str = DEFAULT_STOCKFISH_PATH
    skill_level: int = 20
    use_elo_limit: bool = False
    elo: int = 2500
    movetime_ms: int = 50
    threads: int = 1
    hash_mb: int = 128


class StockfishManager:
  '''
  Manage stockfish engine instances by name for player, eval and reward engines.
  For example, We will use several enignes at diffrenet levels for evaluation,
  or for reward we will limit by time.
  '''
  _engines: dict[str, chess.engine.SimpleEngine] = {}
  _cfgs: dict[str, StockfishConfig] = {}


  @classmethod
  def _configure_engine(cls, engine: chess.engine.SimpleEngine, cfg: StockfishConfig) -> None:
      try:
          engine.configure({"Threads": cfg.threads})
      except Exception:
          warnings.warn("Failed to set Stockfish threads", RuntimeWarning)

      try:
          engine.configure({"Hash": cfg.hash_mb})
      except Exception:
          warnings.warn("Failed to set Stockfish hash size", RuntimeWarning)

      try:
          engine.configure({"Skill Level": cfg.skill_level})
      except Exception:
          warnings.warn("Failed to set Stockfish skill level", RuntimeWarning)

      if cfg.use_elo_limit:
          try:
              engine.configure({
                  "UCI_LimitStrength": True,
                  "UCI_Elo": cfg.elo,
              })
          except Exception:
              warnings.warn("Failed to set Stockfish ELO limit", RuntimeWarning)


  @classmethod
  def get_engine(cls, name: str, cfg: StockfishConfig | None = None) -> chess.engine.SimpleEngine:
      """
      Get (or create) a named engine instance.
      - name: e.g. "reward", "player"
      - cfg: config to use when creating it (ignored later calls).
      """
      if name not in cls._engines:
          if cfg is None:
              cfg = StockfishConfig()
          engine = chess.engine.SimpleEngine.popen_uci(cfg.path)
          cls._configure_engine(engine, cfg)
          cls._engines[name] = engine
          cls._cfgs[name] = cfg
      return cls._engines[name]


  @classmethod
  def close(cls, name: str) -> None:
      engine = cls._engines.get(name)
      if engine is not None:
          try:
              engine.quit()
          except Exception:
              warnings.warn(f"Failed to close Stockfish engine '{name}' in StockfishManager", RuntimeWarning)
          finally:
              cls._engines.pop(name, None)
              cls._cfgs.pop(name, None)


  @classmethod
  def close_all(cls) -> None:
      for name in list(cls._engines.keys()):
          cls.close(name)



class StockfishPlayer(ChessPlayer):
    '''
    A chess player that uses Stockfish engine to select moves.
    '''
    def __init__(self, cfg: StockfishConfig):
        self.cfg = cfg
        self.engine = StockfishManager.get_engine("player_engine", cfg)


    def close(self):
        try:
            self.engine.quit()
        except Exception:
            warnings.warn("Failed to close Stockfish engine in StockfishPlayer", RuntimeWarning)

    def act(self, board: chess.Board) -> chess.Move | None:
        limit = chess.engine.Limit(time=self.cfg.movetime_ms / 1000.0)
        result = self.engine.play(board, limit)
        return result.move
