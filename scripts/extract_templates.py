"""Automatic extraction of piece templates from screen.

How to use:
  1. Open your chess board website (any position)
  2. Launch:  python scripts/extract_templates.py
  3. Select the board area (8×8 cells only)
  4. Enter the FEN of this position
  5. The program will crop all pieces and save them as templates

If one screenshot is not enough (some piece is missing),
just repeat with another position — new templates will be added.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re
from typing import Optional

import cv2
import mss
import numpy as np

from chessai.local_detector import PIECES, TEMPLATE_DIR


FEN_RE = re.compile(r"^([rnbqkpRNBQKP1-8]+/){7}[rnbqkpRNBQKP1-8]+$")


def select_board_region() -> Optional[tuple[int, int, int, int]]:
    with mss.mss() as sct:
        screen = sct.grab(sct.monitors[1])
        img = cv2.cvtColor(np.array(screen), cv2.COLOR_BGRA2BGR)

    clone = img.copy()
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
            cv2.putText(display, f"{roi[2]}x{roi[3]}", (roi[0], roi[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(win, display)
        if cv2.waitKey(30) & 0xFF != 255:
            break

    cv2.destroyWindow(win)

    if len(roi) == 4 and roi[2] > 80 and roi[3] > 80:
        side = min(roi[2], roi[3])
        return (roi[0], roi[1], side, side)
    print("Invalid region")
    return None


def board_fen_to_map(board_fen: str) -> dict[str, str]:
    """Convert 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR' to {'a1': 'R', ...}."""
    files = "abcdefgh"
    result = {}
    for rank_idx, row in enumerate(board_fen.split("/")):
        rank = 8 - rank_idx
        file_idx = 0
        for ch in row:
            if ch.isdigit():
                file_idx += int(ch)
            else:
                sq = f"{files[file_idx]}{rank}"
                piece = ch.upper() if ch.isupper() else ch.lower()
                result[sq] = ch
                file_idx += 1
    return result


def extract_templates_from_fen(
    image_bgr: np.ndarray,
    board_fen: str,
    *,
    flipped: bool = False,
) -> int:
    """Extract piece images from a board screenshot given its FEN."""
    h, w = image_bgr.shape[:2]
    cell_h, cell_w = h // 8, w // 8
    board_map = board_fen_to_map(board_fen)
    files = "hgfedcba" if flipped else "abcdefgh"

    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0

    for sq, piece_sym in board_map.items():
        file_char, rank_str = sq[0], int(sq[1])
        if flipped:
            col = 7 - files.index(file_char)
            row = rank_str - 1
        else:
            col = files.index(file_char)
            row = 8 - rank_str

        y1, y2 = row * cell_h, (row + 1) * cell_h
        x1, x2 = col * cell_w, (col + 1) * cell_w
        cell = image_bgr[y1:y2, x1:x2]

        if cell.size == 0:
            continue

        prefix = "w" if piece_sym.isupper() else "b"
        sym_upper = piece_sym.upper()
        out_name = f"{prefix}_{sym_upper}.png"
        out_path = TEMPLATE_DIR / out_name

        cv2.imwrite(str(out_path), cell)
        saved += 1
        print(f"  [{piece_sym}] {sq} -> {out_name}")

    return saved


def main() -> None:
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Chess Template Extractor ===")
    print("1. Select board region on screen")
    print("2. Enter the FEN of the current position")
    print()

    region = select_board_region()
    if region is None:
        return

    with mss.mss() as sct:
        mon = {"left": region[0], "top": region[1], "width": region[2], "height": region[3]}
        shot = sct.grab(mon)
        img = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)

    cv2.imshow("Captured board — close to continue", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    while True:
        board_fen = input("Enter board FEN (e.g. rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR): ").strip()
        if FEN_RE.match(board_fen):
            break
        print("Invalid FEN format. Use only the position part (8 slash-separated rows).")

    flipped_input = input("Board flipped (black at bottom)? [y/N]: ").strip().lower()
    flipped = flipped_input == "y"

    print(f"\nExtracting templates from FEN: {board_fen}")
    print(f"Flipped: {flipped}")
    print()

    saved = extract_templates_from_fen(img, board_fen, flipped=flipped)

    existing = list(TEMPLATE_DIR.glob("*.png"))
    print(f"\nSaved {saved} templates this run.")
    print(f"Total templates in {TEMPLATE_DIR}: {len(existing)}")

    # Check which pieces are still missing
    all_symbols = set(PIECES.keys())
    have_symbols = set()
    for p in existing:
        name = p.stem
        for sym in all_symbols:
            expected = f"{'w' if sym.isupper() else 'b'}_{sym.upper()}"
            if name == expected:
                have_symbols.add(sym)

    missing = all_symbols - have_symbols
    if missing:
        missing_names = [f"{'w' if s.isupper() else 'b'}_{s.upper()}" for s in missing]
        print(f"\nStill missing: {', '.join(sorted(missing_names))}")
        print("Run again with a position containing these pieces.")
    else:
        print("\nAll 12 piece types saved! Template matching will now work.")


if __name__ == "__main__":
    main()
