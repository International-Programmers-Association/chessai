from __future__ import annotations

import threading
import time
import tkinter as tk
from dataclasses import dataclass
from typing import Callable, Optional

import chess
import cv2
import mss
import numpy as np

from chessai.cv_scanner import ScanResult, frame_changed, scan_board
from chessai.position_tracker import PositionTracker, infer_turn_from_diff


@dataclass
class BoardRegion:
    left: int
    top: int
    width: int
    height: int


@dataclass
class ScreenReadResult:
    board_fen: str
    turn: chess.Color
    flipped: bool
    changed: bool
    confidence: float
    last_move: Optional[chess.Move] = None
    last_move_san: str = ""
    ply: int = 0
    source: str = "scan"


class ScreenBoardReader:
    """Reads chess positions from screen via fast local scan (+ optional API fallback)."""

    def __init__(self) -> None:
        self.region: Optional[BoardRegion] = None
        self.screen_flipped = False
        self.auto_turn = True
        self.manual_turn: chess.Color = chess.WHITE
        self.tracker = PositionTracker()
        self._sct = mss.mss()
        self._last_frame: Optional[np.ndarray] = None
        self._emitted_fen: Optional[str] = None
        self._scan_lock = threading.Lock()
        self._scanning = False
        self._scan_start_time = 0.0
        self._latest_frame: Optional[np.ndarray] = None

    @property
    def is_calibrated(self) -> bool:
        return self.region is not None

    def close(self) -> None:
        try:
            self._sct.close()
        except Exception:
            pass

    def capture(self) -> np.ndarray:
        if self.region is None:
            raise RuntimeError("region not set")
        monitor = {
            "left": self.region.left,
            "top": self.region.top,
            "width": self.region.width,
            "height": self.region.height,
        }
        shot = self._sct.grab(monitor)
        return cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)

    def set_region(self, region: BoardRegion) -> None:
        self.region = region
        self.tracker.reset()
        self._last_frame = None
        self._emitted_fen = None
        self._latest_frame = None

    def reset_region(self) -> None:
        self.region = None
        self.tracker.reset()
        self._last_frame = None
        self._emitted_fen = None
        self._latest_frame = None

    def set_turn_mode(self, *, auto_turn: bool, manual_turn: chess.Color) -> None:
        self.auto_turn = auto_turn
        self.manual_turn = manual_turn

    def apply_manual_turn(self) -> None:
        if self.tracker.last is None:
            return
        last = self.tracker.last
        if last.turn == self.manual_turn:
            return
        self.tracker.force(last.board_fen, self.manual_turn)

    def set_orientation_for_player(self, player_color: chess.Color) -> None:
        self.screen_flipped = player_color == chess.BLACK
        self.tracker.reset()
        self._last_frame = None
        self._emitted_fen = None

    def scan_once(self, player_color: chess.Color = chess.WHITE) -> Optional[ScreenReadResult]:
        if self.region is None:
            return None
        frame = self.capture()
        scan = self._scan_frame(frame, player_color, allow_api=True)
        if not scan.success or not scan.board_fen:
            return None
        turn = self._resolve_scan_turn(scan)
        tracked = self.tracker.force(scan.board_fen, turn)
        self._emitted_fen = tracked.board_fen
        self._last_frame = frame
        return self._to_result(tracked, changed=True, confidence=0.95, source=scan.source)

    def read_position(
        self,
        *,
        force_sync: bool = False,
        player_color: chess.Color = chess.WHITE,
    ) -> Optional[ScreenReadResult]:
        if self.region is None:
            return None
        frame = self.capture()
        scan = self._scan_frame(frame, player_color, allow_api=force_sync)
        if not scan.success:
            return None
        return self._accept_scan(scan, force=True)

    def read_position_async(
        self,
        player_color: chess.Color,
        callback: Callable[[Optional[ScreenReadResult], Optional[str]], None],
        *,
        force: bool = False,
    ) -> None:
        if self.region is None:
            callback(None)
            return

        with self._scan_lock:
            if self._scanning:
                callback(None, "busy")
                return
            self._scanning = True
            self._scan_start_time = time.monotonic()

        def worker() -> None:
            result: Optional[ScreenReadResult] = None
            error: Optional[str] = None
            try:
                frame = self.capture()
                if frame is None:
                    error = "capture failed"
                else:
                    scan = self._scan_frame(frame, player_color, allow_api=False)
                    if scan.success:
                        result = self._accept_scan(scan, force=force)
                    else:
                        error = scan.error
            except Exception as exc:
                error = str(exc)
            finally:
                with self._scan_lock:
                    self._scanning = False
                    self._scan_start_time = 0.0
                callback(result, error)

        threading.Thread(target=worker, daemon=True).start()

    def _scan_frame(
        self,
        frame: np.ndarray,
        player_color: chess.Color,
        *,
        allow_api: bool,
    ) -> ScanResult:
        player = "black" if player_color == chess.BLACK else "white"
        return scan_board(
            frame,
            current_player=player,
            flipped=self.screen_flipped,
            prefer_local=True,
            allow_api=allow_api,
        )

    def _resolve_scan_turn(self, scan: ScanResult) -> chess.Color:
        if not scan.board_fen:
            return chess.WHITE
        if not self.auto_turn:
            return self.manual_turn
        if scan.turn is not None:
            return scan.turn
        from chessai.position_tracker import infer_turn_from_position

        return infer_turn_from_position(scan.board_fen)

    def _accept_scan(self, scan: ScanResult, *, force: bool) -> Optional[ScreenReadResult]:
        if not scan.board_fen:
            return None

        tracked = self.tracker.ingest_scan(
            scan.board_fen,
            scan.turn,
            manual_turn=self.manual_turn,
            auto_turn=self.auto_turn,
        )

        if tracked is None:
            resolved_turn = self._resolve_scan_turn(scan)
            tracked = self.tracker.force(scan.board_fen, resolved_turn)

        changed = tracked.board_fen != self._emitted_fen or force
        self._emitted_fen = tracked.board_fen
        return self._to_result(
            tracked,
            changed=changed,
            confidence=0.92,
            source=scan.source,
        )

    def _to_result(
        self,
        tracked,
        *,
        changed: bool,
        confidence: float,
        source: str,
    ) -> ScreenReadResult:
        return ScreenReadResult(
            board_fen=tracked.board_fen,
            turn=tracked.turn,
            flipped=self.screen_flipped,
            changed=changed,
            confidence=confidence,
            last_move=tracked.last_move,
            last_move_san=tracked.last_move_san,
            ply=tracked.ply,
            source=source,
        )


def select_screen_region(root: tk.Tk, on_done: Callable[[Optional[BoardRegion]], None]) -> None:
    overlay = tk.Toplevel(root)
    overlay.attributes("-fullscreen", True)
    overlay.attributes("-alpha", 0.25)
    overlay.configure(bg="black")
    overlay.attributes("-topmost", True)

    canvas = tk.Canvas(overlay, cursor="cross", bg="black", highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)

    hint = tk.Label(
        overlay,
        text="Select the board (like Chessvision: only the cells). Esc — cancel.",
        bg="#000000",
        fg="#00ff88",
        font=("Segoe UI", 12),
    )
    hint.place(relx=0.5, rely=0.04, anchor=tk.N)

    start: dict[str, int] = {}
    rect_id: Optional[int] = None

    def on_press(event: tk.Event) -> None:
        nonlocal rect_id
        start["x"], start["y"] = event.x_root, event.y_root
        if rect_id is not None:
            canvas.delete(rect_id)
        rect_id = canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline="#00ff88", width=2
        )

    def on_drag(event: tk.Event) -> None:
        if rect_id is None:
            return
        x0 = min(start["x"], event.x_root) - overlay.winfo_rootx()
        y0 = min(start["y"], event.y_root) - overlay.winfo_rooty()
        x1 = max(start["x"], event.x_root) - overlay.winfo_rootx()
        y1 = max(start["y"], event.y_root) - overlay.winfo_rooty()
        canvas.coords(rect_id, x0, y0, x1, y1)

    def finish(event: tk.Event) -> None:
        overlay.destroy()
        if "x" not in start:
            on_done(None)
            return
        left = min(start["x"], event.x_root)
        top = min(start["y"], event.y_root)
        width = abs(event.x_root - start["x"])
        height = abs(event.y_root - start["y"])
        if width < 80 or height < 80:
            on_done(None)
            return
        side = min(width, height)
        on_done(BoardRegion(left=left, top=top, width=side, height=side))

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", finish)
    overlay.bind("<Escape>", lambda _e: (overlay.destroy(), on_done(None)))


def infer_turn(
    prev_board_fen: Optional[str],
    prev_turn: chess.Color,
    new_board_fen: str,
) -> chess.Color:
    if prev_board_fen is None:
        return chess.WHITE
    if prev_board_fen == new_board_fen:
        return prev_turn
    turn = infer_turn_from_diff(prev_board_fen, prev_turn, new_board_fen)
    return turn if turn is not None else prev_turn
