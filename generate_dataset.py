"""Generate synthetic boards for all 3 sites into D:\chessbot\dataset."""
import json
import random
import sys
from pathlib import Path

import chess
import cv2
import numpy as np

DATASET_DIR = Path("D:/chessbot/dataset")

PIECE_FILES = {
    "K": "wk.png", "Q": "wq.png", "R": "wr.png",
    "B": "wb.png", "N": "wn.png", "P": "wp.png",
    "k": "bk.png", "q": "bq.png", "r": "br.png",
    "b": "bb.png", "n": "bn.png", "p": "bp.png",
}

SITES = [
    {
        "name": "chesscom",
        "light_hex": "#EBECD0",
        "dark_hex": "#739552",
        "pieces_dir": Path("D:/chessbot/data/chesscom"),
    },
    {
        "name": "lichess",
        "light_hex": "#F0D9B5",
        "dark_hex": "#B58863",
        "pieces_dir": Path("D:/chessbot/data/lichess"),
    },
    {
        "name": "worldchess",
        "light_hex": "#E9E5D4",
        "dark_hex": "#6A994F",
        "pieces_dir": Path("D:/chessbot/data/worldchess"),
    },
]


def hex_to_bgr(h: str):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def load_pieces(pieces_dir: Path) -> dict[str, np.ndarray]:
    pieces: dict[str, np.ndarray] = {}
    for sym, fn in PIECE_FILES.items():
        path = pieces_dir / fn
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is not None:
            pieces[sym] = img
        else:
            print(f"  WARNING: missing {path}")
    return pieces


def render_board(fen: str, pieces: dict[str, np.ndarray],
                 light_bgr: tuple, dark_bgr: tuple,
                 rng: random.Random) -> np.ndarray:
    board = chess.Board(fen)
    img = np.zeros((480, 480, 3), dtype=np.uint8)

    for row in range(8):
        for col in range(8):
            y1, y2 = row * 60, (row + 1) * 60
            x1, x2 = col * 60, (col + 1) * 60
            is_light = (row + col) % 2 == 0
            img[y1:y2, x1:x2] = light_bgr if is_light else dark_bgr

            sq = chess.square(col, 7 - row)
            piece = board.piece_at(sq)
            if piece is None:
                continue

            sym = piece.symbol()
            tmpl = pieces.get(sym)
            if tmpl is None:
                continue

            target = rng.randint(30, 56)
            if target % 2:
                target += 1
            ox = rng.randint(-2, 2)
            oy = rng.randint(-2, 2)
            resized = cv2.resize(tmpl, (target, target), interpolation=cv2.INTER_AREA)
            off_x = max(0, min(60 - target, (60 - target) // 2 + ox))
            off_y = max(0, min(60 - target, (60 - target) // 2 + oy))

            if resized.shape[2] == 4:
                alpha = resized[:, :, 3].astype(np.float32) / 255.0
                bgr = resized[:, :, :3].astype(np.float32)
                region = img[y1+off_y:y1+off_y+target, x1+off_x:x1+off_x+target].astype(np.float32)
                blended = bgr * alpha[:, :, None] + region * (1.0 - alpha[:, :, None])
                img[y1+off_y:y1+off_y+target, x1+off_x:x1+off_x+target] = blended.clip(0, 255).astype(np.uint8)
            else:
                img[y1+off_y:y1+off_y+target, x1+off_x:x1+off_x+target] = resized[:, :, :3]

    return img


def main():
    FILES = "abcdefgh"
    BOARDS_PER_SITE = 2000

    for site in SITES:
        site_dir = DATASET_DIR / site["name"]
        site_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*50}")
        print(f"Generating {BOARDS_PER_SITE} boards for {site['name']}...")
        print(f"  Colors: light={site['light_hex']} dark={site['dark_hex']}")
        print(f"  Pieces: {site['pieces_dir']}")

        pieces = load_pieces(site["pieces_dir"])
        if len(pieces) < 12:
            print(f"  SKIP: only {len(pieces)}/12 pieces loaded")
            continue

        light_bgr = hex_to_bgr(site["light_hex"])
        dark_bgr = hex_to_bgr(site["dark_hex"])
        rng = random.Random(42)
        samples = []

        for i in range(BOARDS_PER_SITE):
            board = chess.Board()
            for _ in range(rng.randint(10, 60)):
                legal = list(board.legal_moves)
                if not legal:
                    break
                board.push(rng.choice(legal))

            fen = board.fen()
            img = render_board(fen, pieces, light_bgr, dark_bgr, rng)

            filename = f"board_{i:05d}.png"
            cv2.imwrite(str(site_dir / filename), img)

            cells = {}
            for row in range(8):
                for col in range(8):
                    sq = f"{FILES[col]}{8 - row}"
                    piece = board.piece_at(chess.square(col, 7 - row))
                    cells[sq] = piece.symbol() if piece else "."

            samples.append({"image": filename, "cells": cells, "fen": fen, "site": site["name"]})

            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{BOARDS_PER_SITE}")

        meta_path = site_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(samples, f, indent=1)
        print(f"  Saved {len(samples)} boards + metadata to {site_dir}")

    print(f"\n{'='*50}")
    print(f"Done! All datasets in {DATASET_DIR}/")
    total = BOARDS_PER_SITE * len(SITES)
    print(f"Total boards: {total}")
    print(f"Disk space: ~{total * 480 * 480 * 3 // (1024*1024)} MB")


if __name__ == "__main__":
    main()
