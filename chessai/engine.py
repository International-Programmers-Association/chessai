from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import chess
import chess.engine

DEFAULT_ENGINE_PATH = (Path(sys._MEIPASS) / "engines" / "stockfish.exe" if getattr(sys, 'frozen', False)
                        else Path(__file__).resolve().parent.parent / "engines" / "stockfish.exe")
DEFAULT_NODE_PATH = r"D:\NodeJS\node.exe"
TORCH_WRAPPER_PATH = Path(__file__).resolve().parent.parent / "torch" / "js" / "torch_wrapper.js"


@dataclass
class AnalysisLine:
    rank: int
    score_cp: Optional[int]
    score_mate: Optional[int]
    depth: int
    nodes: int
    pv: list[chess.Move] = field(default_factory=list)

    @property
    def score_text(self) -> str:
        if self.score_mate is not None:
            sign = "+" if self.score_mate > 0 else ""
            return f"#{sign}{self.score_mate}"
        if self.score_cp is not None:
            pawns = self.score_cp / 100
            sign = "+" if pawns > 0 else ""
            return f"{sign}{pawns:.2f}"
        return "—"

    def pv_san(self, board: chess.Board) -> str:
        if not self.pv:
            return ""
        copy = board.copy(stack=False)
        moves: list[str] = []
        for move in self.pv:
            try:
                moves.append(copy.san(move))
                copy.push(move)
            except ValueError:
                break
        return " ".join(moves)


@dataclass
class AnalysisResult:
    fen: str
    lines: list[AnalysisLine]
    finished: bool = True


def _parse_info_line(board: chess.Board, info: dict, rank: int) -> AnalysisLine:
    score = info.get("score")
    score_cp: Optional[int] = None
    score_mate: Optional[int] = None
    if score is not None:
        pov = score.pov(board.turn)
        if pov.is_mate():
            score_mate = pov.mate()
        else:
            score_cp = pov.score()

    return AnalysisLine(
        rank=rank,
        score_cp=score_cp,
        score_mate=score_mate,
        depth=info.get("depth", 0),
        nodes=info.get("nodes", 0),
        pv=info.get("pv", []),
    )


class StockfishEngine:
    def __init__(
        self,
        engine_path: Path | str = DEFAULT_ENGINE_PATH,
        threads: Optional[int] = None,
        hash_mb: int = 512,
    ) -> None:
        self.engine_path = Path(engine_path)
        self.threads = threads or os.cpu_count() or 4
        self.hash_mb = hash_mb
        self._engine: Optional[chess.engine.SimpleEngine] = None
        self._busy = threading.Lock()
        self._analysis_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._current_fen: Optional[str] = None

    @property
    def is_running(self) -> bool:
        return self._engine is not None

    def _configure(self) -> None:
        self._engine.configure({"Threads": self.threads, "Hash": self.hash_mb})

    def _popen(self) -> chess.engine.SimpleEngine:
        eng = chess.engine.SimpleEngine.popen_uci(str(self.engine_path))
        eng.configure({"Threads": self.threads, "Hash": self.hash_mb})
        return eng

    def set_threads(self, n: int) -> None:
        self.threads = n
        if self._engine is not None:
            try:
                self._engine.configure({"Threads": n})
            except Exception:
                pass

    def set_hash(self, mb: int) -> None:
        self.hash_mb = mb
        if self._engine is not None:
            try:
                self._engine.configure({"Hash": mb})
            except Exception:
                pass

    def start(self) -> None:
        if self._engine is not None:
            return
        if not self.engine_path.is_file():
            raise FileNotFoundError(f"Stockfish not found: {self.engine_path}")
        self._engine = self._popen()

    def stop(self) -> None:
        self.stop_continuous_analysis()
        eng = self._engine
        self._engine = None
        if eng is not None:
            try:
                eng.quit()
            except Exception:
                pass

    def set_engine_path(self, path: Path | str) -> None:
        self.engine_path = Path(path)
        self.stop()
        self.start()

    def analyze_position(
        self,
        board: chess.Board,
        *,
        time_limit: float = 1.0,
        depth: Optional[int] = None,
        nodes: Optional[int] = None,
        multipv: int = 3,
    ) -> AnalysisResult:
        eng = self._engine
        if eng is None:
            raise RuntimeError("Engine is not started")

        limit = chess.engine.Limit(time=time_limit, depth=depth, nodes=nodes)
        with self._busy:
            try:
                infos = eng.analyse(board, limit, multipv=multipv, info=chess.engine.INFO_ALL)
            except Exception:
                # Engine crashed — don't call quit() on it (can hang).
                # Spawn a replacement silently.
                self._engine = self._popen()
                infos = self._engine.analyse(board, limit, multipv=multipv, info=chess.engine.INFO_ALL)

        if not isinstance(infos, list):
            infos = [infos]

        lines = [_parse_info_line(board, info, i + 1) for i, info in enumerate(infos)]
        return AnalysisResult(fen=board.fen(), lines=lines)

    def stop_continuous_analysis(self) -> None:
        self._stop_event.set()
        if self._analysis_thread and self._analysis_thread.is_alive():
            self._analysis_thread.join(timeout=2.0)
        self._analysis_thread = None
        self._current_fen = None

    def start_continuous_analysis(
        self,
        get_board: Callable[[], chess.Board],
        on_update: Callable[[AnalysisResult], None],
        *,
        time_limit: float = 0.5,
        depth: Optional[int] = None,
        nodes: Optional[int] = None,
        multipv: int = 3,
        poll_interval: float = 0.15,
    ) -> None:
        self.stop_continuous_analysis()

        def worker() -> None:
            while not self._stop_event.is_set():
                board = get_board()
                fen = board.fen()
                if fen != self._current_fen:
                    self._current_fen = fen
                try:
                    result = self.analyze_position(
                        board, time_limit=time_limit, depth=depth, nodes=nodes, multipv=multipv
                    )
                    if not self._stop_event.is_set() and get_board().fen() == result.fen:
                        on_update(result)
                except Exception:
                    pass
                if self._stop_event.wait(poll_interval):
                    break

        self._stop_event.clear()
        self._analysis_thread = threading.Thread(target=worker, daemon=True)
        self._analysis_thread.start()


class TorchEngine(StockfishEngine):
    def __init__(
        self,
        node_path: Path | str = DEFAULT_NODE_PATH,
        threads: Optional[int] = None,
        hash_mb: int = 16,
    ) -> None:
        self.node_path = Path(node_path)
        self._torch_wrapper = TORCH_WRAPPER_PATH
        self.engine_path = self._torch_wrapper
        self.threads = 1
        self.hash_mb = min(hash_mb, 128)
        self._engine: Optional[chess.engine.SimpleEngine] = None
        self._busy = threading.Lock()
        self._analysis_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._current_fen: Optional[str] = None

    def _popen(self) -> chess.engine.SimpleEngine:
        eng = chess.engine.SimpleEngine.popen_uci([str(self.node_path), str(self._torch_wrapper)])
        eng.configure({"Threads": self.threads, "Hash": self.hash_mb})
        return eng

    def set_hash(self, mb: int) -> None:
        super().set_hash(min(mb, 128))

    def set_threads(self, n: int) -> None:
        super().set_threads(n)

    def set_node_path(self, path: Path | str) -> None:
        self.node_path = Path(path)
        self.stop()
        self.start()
