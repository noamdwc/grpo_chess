
import chess
import chess.engine
from dataclasses import dataclass

STOCKFISH_PATH = "/usr/games/stockfish"


@dataclass
class StockfishConfig:
    path: str = "stockfish"
    skill_level: int = 20       # 0..20 typical
    use_elo_limit: bool = False
    elo: int = 2500
    movetime_ms: int = 50       # per move
    threads: int = 1
    hash_mb: int = 128


class StockfishPlayer:
    def __init__(self, cfg: StockfishConfig):
        self.cfg = cfg
        self.engine = chess.engine.SimpleEngine.popen_uci(cfg.path)

        # Configure engine options if supported
        try:
            self.engine.configure({"Threads": cfg.threads})
        except Exception:
            pass
        try:
            self.engine.configure({"Hash": cfg.hash_mb})
        except Exception:
            pass

        # Strength controls
        try:
            self.engine.configure({"Skill Level": cfg.skill_level})
        except Exception:
            pass

        if cfg.use_elo_limit:
            # Some builds support these options
            try:
                self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": cfg.elo})
            except Exception:
                pass

    def close(self):
        try:
            self.engine.quit()
        except Exception:
            pass

    def choose_move(self, board: chess.Board) -> chess.Move | None:
        limit = chess.engine.Limit(time=self.cfg.movetime_ms / 1000.0)
        result = self.engine.play(board, limit)
        return result.move
