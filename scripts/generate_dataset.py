"""Generate synthetic chess board dataset for training.

Uses exact piece images from piece_templates/ and board colors
provided by the user.  Renders legal chess positions, saves both
full-board images and per-cell crops, producing metadata.json
compatible with train_model.py.
"""

import json
import random
import sys
from pathlib import Path

import chess
import chess.pgn
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chessai.local_detector import PIECES

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "piece_templates"

# Board colours — user corrected
LIGHT_HEX = "#E9E5D4"
DARK_HEX = "#6A994F"

# Internal SVG coordinates: 480x480 board, 60x60 cells
# Website displays at 688x688 but that's just up-scaling
BOARD_PX = 480
CELL_PX = BOARD_PX // 8  # 60

# Piece SVGs are 60x60 viewBox; templates (120x120 at 2x) resized to 60x60
PIECE_SCALE = 1.0

FILES = "abcdefgh"

# Sym -> filename mapping
PIECE_FILES = {
    "K": "wk.png", "Q": "wq.png", "R": "wr.png",
    "B": "wb.png", "N": "wn.png", "P": "wp.png",
    "k": "bk.png", "q": "bq.png", "r": "br.png",
    "b": "bb.png", "n": "bn.png", "p": "bp.png",
}


def hex_to_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


LIGHT_BGR = hex_to_bgr(LIGHT_HEX)  # light squares
DARK_BGR = hex_to_bgr(DARK_HEX)    # dark squares


def _load_pieces() -> dict[str, np.ndarray]:
    pieces: dict[str, np.ndarray] = {}
    for sym, filename in PIECE_FILES.items():
        path = TEMPLATE_DIR / filename
        if not path.exists():
            print(f"  WARNING: missing {path}")
            continue
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  WARNING: could not read {path}")
            continue
        pieces[sym] = img
    return pieces


def render_board(
    fen: str,
    pieces: dict[str, np.ndarray],
    *,
    board_px: int = BOARD_PX,
) -> np.ndarray:
    """Render a chess position as a 480x480 BGR image. Always white at bottom."""
    board = chess.Board(fen)
    cell = board_px // 8
    img = np.zeros((board_px, board_px, 3), dtype=np.uint8)

    for row in range(8):
        for col in range(8):
            y1, y2 = row * cell, (row + 1) * cell
            x1, x2 = col * cell, (col + 1) * cell
            is_light = (row + col) % 2 == 0
            colour = LIGHT_BGR if is_light else DARK_BGR
            img[y1:y2, x1:x2] = colour

            sq = chess.square(col, 7 - row)
            piece = board.piece_at(sq)
            if piece is None:
                continue

            sym = piece.symbol()
            tmpl = pieces.get(sym)
            if tmpl is None:
                continue

            target = int(cell * PIECE_SCALE)
            if target % 2:
                target += 1
            resized = cv2.resize(tmpl, (target, target), interpolation=cv2.INTER_AREA)

            off_x = (cell - target) // 2
            off_y = (cell - target) // 2

            if resized.shape[2] == 4:
                alpha = resized[:, :, 3].astype(np.float32) / 255.0
                bgr = resized[:, :, :3].astype(np.float32)
                region = img[y1 + off_y : y1 + off_y + target,
                             x1 + off_x : x1 + off_x + target].astype(np.float32)
                blended = bgr * alpha[:, :, None] + region * (1.0 - alpha[:, :, None])
                img[y1 + off_y : y1 + off_y + target,
                    x1 + off_x : x1 + off_x + target] = blended.clip(0, 255).astype(np.uint8)
            else:
                img[y1 + off_y : y1 + off_y + target,
                    x1 + off_x : x1 + off_x + target] = resized[:, :, :3]

    return img


def generate_boards(
    num_boards: int,
    pieces: dict[str, np.ndarray],
) -> list[dict]:
    """Generate synthetic boards with metadata compatible with train_model.py."""
    samples: list[dict] = []
    rng = random.Random(42)

    for i in range(num_boards):
        board = chess.Board()
        num_moves = rng.randint(10, 60)
        for _ in range(num_moves):
            legal = list(board.legal_moves)
            if not legal:
                break
            move = rng.choice(legal)
            board.push(move)

        fen = board.fen()
        img = render_board(fen, pieces)

        # Save full board image
        filename = f"synthetic_{i:05d}.png"
        cv2.imwrite(str(DATASET_DIR / filename), img)

        # Build cells dict
        cells: dict[str, str] = {}
        for row in range(8):
            for col in range(8):
                sq = f"{FILES[col]}{8 - row}"
                piece = board.piece_at(chess.square(col, 7 - row))
                cells[sq] = piece.symbol() if piece else "."

        samples.append({"image": filename, "cells": cells, "fen": fen})

        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{num_boards}")

    return samples


def main():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading piece images from piece_templates/ ...")
    pieces = _load_pieces()
    if len(pieces) < 12:
        print(f"ERROR: only {len(pieces)} piece types loaded (need 12).")
        print("Make sure piece_templates/ has all 12 PNGs (wk.png, bk.png, etc.)")
        sys.exit(1)
    print(f"  Loaded {len(pieces)} piece types")

    num = 2000
    print(f"Generating {num} synthetic boards ...")
    samples = generate_boards(num, pieces)

    # Save metadata — only synthetic boards
    meta_path = DATASET_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"\nDone! Saved {len(samples)} synthetic boards to {DATASET_DIR}/")
    print(f"Metadata: {meta_path}")

    # Quick stats
    total_cells = len(samples) * 64
    total_pieces = sum(
        1 for s in samples for v in s["cells"].values() if v != "."
    )
    print(f"Total cells: {total_cells}")
    print(f"Total pieces: {total_pieces}")
    print(f"Total empty:  {total_cells - total_pieces}")


if __name__ == "__main__":
    main()
