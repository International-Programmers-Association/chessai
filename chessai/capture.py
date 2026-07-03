from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import cv2
import numpy as np


class ChangeLevel(Enum):
    UNKNOWN = 0
    SAME = 1
    BOARD_CHANGED = 2
    BOARD_RESET = 3


@dataclass
class FrameEvent:
    frame: np.ndarray
    timestamp: float
    change: ChangeLevel
    score: float


class FrameDiffDetector:
    """Detects meaningful changes between consecutive screen captures."""

    def __init__(self, ssim_threshold: float = 0.92, grid: int = 8):
        self._prev: Optional[np.ndarray] = None
        self._ssim_threshold = ssim_threshold
        self._grid = grid

    def reset(self) -> None:
        self._prev = None

    def _ssim(self, a: np.ndarray, b: np.ndarray) -> float:
        a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.ndim == 3 else a
        b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) if b.ndim == 3 else b

        h, w = a_gray.shape
        cell_h, cell_w = h // self._grid, w // self._grid

        scores = []
        for r in range(self._grid):
            for c in range(self._grid):
                y1, y2 = r * cell_h, (r + 1) * cell_h
                x1, x2 = c * cell_w, (c + 1) * cell_w
                tile_a = a_gray[y1:y2, x1:x2].astype(np.float32)
                tile_b = b_gray[y1:y2, x1:x2].astype(np.float32)
                ma, mb = tile_a.mean(), tile_b.mean()
                sa, sb = tile_a.std(), tile_b.std()
                if sa < 1e-6 or sb < 1e-6:
                    scores.append(1.0 if abs(ma - mb) < 5 else 0.0)
                    continue
                cov = ((tile_a - ma) * (tile_b - mb)).mean()
                denom = sa * sb
                if denom < 1e-8:
                    scores.append(1.0)
                    continue
                scores.append(float(cov / denom))
        return float(np.mean(scores))

    def classify(self, frame: np.ndarray) -> tuple[ChangeLevel, float]:
        if self._prev is None or self._prev.shape != frame.shape:
            self._prev = frame.copy()
            return ChangeLevel.UNKNOWN, 0.0

        score = self._ssim(self._prev, frame)
        self._prev = frame.copy()

        if score >= self._ssim_threshold:
            return ChangeLevel.SAME, score

        return ChangeLevel.BOARD_CHANGED, score


class ContinuousCapture:
    """Captures a screen region at target FPS in a background thread.

    Mimics OBS-style capture: runs a dedicated thread that grabs frames
    at a configurable rate, detects changes, and fires a callback only
    when the board meaningfully changes.

    Args:
        region: (left, top, width, height) of the capture area on screen.
        fps: Target capture rate (capped by actual screen refesh / mss speed).
        change_threshold: SSIM threshold below which a change is reported.
        on_frame: Called on the capture thread with each FrameEvent.
                  Only fires when change=BOARD_CHANGED (or BOARD_RESET).

    Example:
        def on_change(event: FrameEvent):
            print(f"Board changed! SSIM={event.score:.3f}")

        cap = ContinuousCapture((0, 0, 480, 480), fps=30, on_frame=on_change)
        cap.start()
        # ... later ...
        cap.stop()
    """

    def __init__(
        self,
        region: tuple[int, int, int, int],
        *,
        fps: float = 30.0,
        change_threshold: float = 0.92,
        on_frame: Optional[Callable[[FrameEvent], None]] = None,
    ):
        self.region = region
        self.interval = 1.0 / max(fps, 1.0)
        self.change_threshold = change_threshold
        self.on_frame = on_frame

        self._diff = FrameDiffDetector(ssim_threshold=change_threshold)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

    @property
    def latest_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    @property
    def is_running(self) -> bool:
        return self._running

    def update_region(self, region: tuple[int, int, int, int]) -> None:
        self.region = region
        self._diff.reset()

    def _capture(self) -> np.ndarray:
        import mss

        left, top, width, height = self.region
        monitor = {"left": left, "top": top, "width": width, "height": height}
        sct_cls = mss.MSS if hasattr(mss, "MSS") else mss.mss
        with sct_cls() as sct:
            shot = sct.grab(monitor)
            return cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)

    def _loop(self) -> None:
        while self._running:
            t0 = time.perf_counter()

            try:
                frame = self._capture()
            except Exception:
                time.sleep(self.interval)
                continue

            with self._frame_lock:
                self._latest_frame = frame

            change, score = self._diff.classify(frame)
            if change != ChangeLevel.SAME and self.on_frame is not None:
                event = FrameEvent(
                    frame=frame,
                    timestamp=t0,
                    change=change,
                    score=score,
                )
                try:
                    self.on_frame(event)
                except Exception:
                    pass

            elapsed = time.perf_counter() - t0
            remaining = self.interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._diff.reset()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="capture")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def capture_once(self) -> np.ndarray:
        return self._capture()
