"""Standalone live chess board scanner.

Usage:
    python scripts/live_scan.py

Select a region on screen (just the 8×8 board) and watch real-time
position detection via template matching.

Keys:
    q / ESC    — quit
    f          — toggle flipped (black at bottom)
    r          — re-select region
    p          — print current FEN
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
from typing import Optional

import chess
import cv2
import mss
import numpy as np

from chessai.live_scanner import LiveScanner, LiveScannerConfig, LivePosition
from chessai.capture import ContinuousCapture


def select_region() -> Optional[tuple[int, int, int, int]]:
    print("Select board region: drag a rectangle, then press ENTER in terminal.")
    with mss.mss() as sct:
        screen = sct.grab(sct.monitors[1])
        img = cv2.cvtColor(np.array(screen), cv2.COLOR_BGRA2BGR)

    clone = img.copy()
    roi: list[int] = []

    def mouse_cb(event, x, y, flags, param):
        nonlocal roi
        if event == cv2.EVENT_LBUTTONDOWN:
            roi = [x, y, 0, 0]
        elif event == cv2.EVENT_LBUTTONUP:
            roi[2] = x - roi[0]
            roi[3] = y - roi[1]

    win = "Select board region — drag, then press any key"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, mouse_cb)
    cv2.imshow(win, img)

    while True:
        display = img.copy()
        if len(roi) == 4 and roi[2] > 0:
            cv2.rectangle(display, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0, 255, 0), 2)
            cv2.putText(display, f"{roi[2]}x{roi[3]}", (roi[0], roi[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(win, display)
        key = cv2.waitKey(30) & 0xFF
        if key != 255:
            break

    cv2.destroyWindow(win)

    if len(roi) == 4 and roi[2] > 80 and roi[3] > 80:
        side = min(roi[2], roi[3])
        print(f"Region selected: ({roi[0]}, {roi[1]}) {side}×{side}")
        return (roi[0], roi[1], side, side)
    print("No region selected, using full screen center 480×480")
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    return (cx - 240, cy - 240, 480, 480)


def main() -> None:
    region = select_region()
    if region is None:
        return

    flipped = False
    config = LiveScannerConfig(region=region, fps=20)
    last_print = 0.0

    def on_position(pos: LivePosition) -> None:
        nonlocal last_print
        now = time.time()
        if now - last_print > 0.5:
            move_info = f" | {pos.last_move_san}" if pos.last_move_san else ""
            turn = "white" if pos.turn == chess.WHITE else "black"
            print(f"\rMove {pos.ply:3d}  {turn:6s}  FEN: {pos.board_fen}{move_info}  conf={pos.confidence:.0%}   ", end="")
            last_print = now

    scanner = LiveScanner(
        config=config,
        on_position=on_position,
    )

    scanner.start()
    print(f"\nLive scanner running at {config.fps} fps. Press 'q' to quit, 'f' to flip, 'p' to print FEN.")
    print(f"Region: {region}")
    print()

    preview_win = "Live Scan — press q/ESC to quit"
    cv2.namedWindow(preview_win, cv2.WINDOW_NORMAL)

    try:
        while scanner.is_running:
            display = scanner.capture_preview()
            pos = scanner.latest_position
            if pos is not None:
                turn = "white" if pos.turn == chess.WHITE else "black"
                info = [
                    f"FEN: {pos.board_fen}",
                    f"Turn: {turn} (ply {pos.ply})",
                    f"Last move: {pos.last_move_san or '-'}",
                    f"Conf: {pos.confidence:.0%}",
                ]
                for i, line in enumerate(info):
                    cv2.putText(display, line, (10, 30 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(preview_win, display)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('f'):
                flipped = not flipped
                scanner.config.flipped = flipped
                scanner.tracker.reset()
                print(f"\nFlipped = {flipped}")
            elif key == ord('p') and pos:
                print(f"\nFEN: {pos.fen}  (ply {pos.ply}, conf {pos.confidence:.0%})")
    finally:
        scanner.stop()
        cv2.destroyAllWindows()
    print("\nDone.")


if __name__ == "__main__":
    main()
