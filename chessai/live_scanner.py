from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import chess
import cv2
import numpy as np

from chessai.capture import ContinuousCapture, FrameEvent
from chessai.classifier import PieceClassifier, TemplateMatcher
from chessai.local_detector import board_map_to_fen
from chessai.position_tracker import PositionTracker, detect_move


@dataclass
class LivePosition:
    fen: str
    board_fen: str
    turn: chess.Color
    last_move: Optional[chess.Move] = None
    last_move_san: str = ""
    ply: int = 0
    confidence: float = 0.0
    timestamp: float = 0.0
    changed: bool = True


@dataclass
class LiveScannerConfig:
    region: tuple[int, int, int, int] = (0, 0, 480, 480)
    fps: float = 20.0
    change_threshold: float = 0.92
    cooldown_sec: float = 0.3
    flipped: bool = False


class LiveScanner:
    """Real-time chess board scanner with continuous capture + classification.

    Combines ContinuousCapture (OBS-style background capture thread) with
    a PieceClassifier to detect board positions in real-time.

    The capture thread runs at configurable FPS, detects changes via SSIM,
    and only runs classification when the board actually changes.

    Usage:
        def on_position(pos: LivePosition):
            print(pos.fen, pos.last_move_san)

        scanner = LiveScanner(region=(100, 200, 480, 480), on_position=on_position)
        scanner.start()
        # ...
        scanner.stop()
    """

    def __init__(
        self,
        config: Optional[LiveScannerConfig] = None,
        *,
        classifier: Optional[PieceClassifier] = None,
        on_position: Optional[Callable[[LivePosition], None]] = None,
        region: Optional[tuple[int, int, int, int]] = None,
    ):
        self.config = config or LiveScannerConfig()
        if region is not None:
            self.config.region = region

        self.classifier = classifier or TemplateMatcher()
        self.on_position = on_position

        self.tracker = PositionTracker()
        self._capture = ContinuousCapture(
            region=self.config.region,
            fps=self.config.fps,
            change_threshold=self.config.change_threshold,
            on_frame=self._on_frame,
        )
        self._last_emit_time = 0.0
        self._last_fen: Optional[str] = None
        self._classify_lock = threading.Lock()
        self._latest_position: Optional[LivePosition] = None

    @property
    def latest_position(self) -> Optional[LivePosition]:
        return self._latest_position

    @property
    def is_running(self) -> bool:
        return self._capture.is_running

    def update_region(self, region: tuple[int, int, int, int]) -> None:
        self.config.region = region
        self._capture.update_region(region)
        self.tracker.reset()

    def _on_frame(self, event: FrameEvent) -> None:
        now = time.perf_counter()

        if now - self._last_emit_time < self.config.cooldown_sec:
            return

        if not self._classify_lock.acquire(blocking=False):
            return

        try:
            board_map = self.classifier.classify_board(event.frame)
            # If board is flipped (black at bottom), remap ranks without rotating image
            if self.config.flipped:
                remapped: dict[str, Optional[str]] = {}
                for sq, sym in board_map.items():
                    new_rank = 9 - int(sq[1])
                    remapped[f"{sq[0]}{new_rank}"] = sym
                board_map = remapped
            fen = board_map_to_fen(board_map)
            board_fen = fen.split()[0]

            confidence = sum(1 for v in board_map.values() if v is not None) / 64.0

            tracked = self.tracker.ingest_scan(board_fen, turn_hint=None, auto_turn=True)

            if tracked is not None:
                changed = tracked.board_fen != self._last_fen
                self._last_fen = tracked.board_fen
                pos = LivePosition(
                    fen=f"{tracked.board_fen} {'w' if tracked.turn == chess.WHITE else 'b'} - - 0 1",
                    board_fen=tracked.board_fen,
                    turn=tracked.turn,
                    last_move=tracked.last_move,
                    last_move_san=tracked.last_move_san,
                    ply=tracked.ply,
                    confidence=confidence,
                    timestamp=event.timestamp,
                    changed=changed,
                )
                self._latest_position = pos
                self._last_emit_time = now

                if changed and self.on_position is not None:
                    self.on_position(pos)
        finally:
            self._classify_lock.release()

    def start(self) -> None:
        self._capture.start()

    def stop(self) -> None:
        self._capture.stop()
        self.tracker.reset()

    def capture_preview(self) -> np.ndarray:
        return self._capture.capture_once()



