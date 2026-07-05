"""Dataset collection: select the board once, then change positions + enter FEN.

Launch:  python scripts\collect_dataset.py

  1. Select the board area (once)
  2. For each position:
     - Open on the website
     - Enter FEN
     Enter without FEN = exit
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import re
import time
from typing import Optional

import cv2
import mss
import numpy as np

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
FEN_RE = re.compile(r"^([rnbqkpRNBQKP1-8]+/){7}[rnbqkpRNBQKP1-8]+$")


def select_region() -> Optional[tuple[int, int, int, int]]:
    with mss.mss() as sct:
        screen = sct.grab(sct.monitors[1])
        img = cv2.cvtColor(np.array(screen), cv2.COLOR_BGRA2BGR)

    roi: list[int] = []
    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            roi[:] = [x, y, 0, 0]
        elif event == cv2.EVENT_LBUTTONUP:
            roi[2] = x - roi[0]
            roi[3] = y - roi[1]

    win = "Select board — drag, then any key"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, mouse_cb)

    while True:
        display = img.copy()
        if len(roi) == 4 and roi[2] > 0:
            cv2.rectangle(display, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0, 255, 0), 2)
        cv2.imshow(win, display)
        if cv2.waitKey(30) & 0xFF != 255:
            break

    cv2.destroyWindow(win)
    if len(roi) == 4 and roi[2] > 80 and roi[3] > 80:
        side = min(roi[2], roi[3])
        return (roi[0], roi[1], side, side)
    return None


def capture(region) -> np.ndarray:
    with mss.mss() as sct:
        mon = {"left": region[0], "top": region[1], "width": region[2], "height": region[3]}
        return cv2.cvtColor(np.array(sct.grab(mon)), cv2.COLOR_BGRA2BGR)


def make_labels(board_fen: str) -> dict[str, str]:
    """Convert FEN to {square: piece_symbol} dict."""
    files = "abcdefgh"
    labels = {}
    for rank_idx, row in enumerate(board_fen.split("/")):
        rank = 8 - rank_idx
        file_idx = 0
        for ch in row:
            if ch.isdigit():
                file_idx += int(ch)
            else:
                labels[f"{files[file_idx]}{rank}"] = ch
                file_idx += 1
    return labels


def main():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = DATASET_DIR / "metadata.json"

    samples = []
    if meta_path.exists():
        with open(meta_path) as f:
            samples = json.load(f)
        print(f"Loaded {len(samples)} existing samples ({len(samples)*64} cells)")

    print("=" * 60)
    print("DATASET COLLECTION")
    print("=" * 60)
    print("1. Select the board area (once)")
    print("2. For each position on the website:")
    print("   - the program takes a screenshot")
    print("   - you enter the FEN")
    print("   Enter without FEN = exit")
    print()

    region = select_region()
    if region is None:
        print("Region not selected")
        return
    print(f"Region: {region}")
    print()

    count = 0
    while True:
        input("Press Enter when the position is ready on screen...")
        img = capture(region)

        cv2.imshow("Current screenshot — close and enter FEN", cv2.resize(img, (480, 480)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        board_fen = input("FEN (or Enter to exit): ").strip()
        if not board_fen:
            break
        if not FEN_RE.match(board_fen):
            print("  Invalid FEN. Must be like: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
            continue

        ts = int(time.time())
        img_path = DATASET_DIR / f"board_{ts}.png"
        cv2.imwrite(str(img_path), img)

        labels = make_labels(board_fen)
        sample = {"image": img_path.name, "fen": board_fen, "cells": labels}
        samples.append(sample)

        with open(meta_path, "w") as f:
            json.dump(samples, f, indent=1)

        count += 1
        pieces = sum(1 for v in labels.values())
        empty = 64 - pieces
        print(f"  [{count}] Saved: {board_fen[:30]}...  ({pieces} pieces, {empty} empty)")
        print(f"  Total: {len(samples)} boards ({len(samples)*64} cells)")

    print()
    print(f"Done! Collected {len(samples)} boards ({len(samples)*64} cells)")
    print(f"Dataset: {DATASET_DIR}")

    if len(samples) >= 2:
        print()
        print("Now start training:")
        print("  python scripts/train_model.py")


if __name__ == "__main__":
    main()
