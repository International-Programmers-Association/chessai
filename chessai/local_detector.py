from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import chess
import cv2
import numpy as np

from chessai.fast_matcher import batch_match_from_gray, match_backend_name, prepare_templates

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "piece_templates"

PIECES = {
    "P": ("white", "pawn"),
    "N": ("white", "knight"),
    "B": ("white", "bishop"),
    "R": ("white", "rook"),
    "Q": ("white", "queen"),
    "K": ("white", "king"),
    "p": ("black", "pawn"),
    "n": ("black", "knight"),
    "b": ("black", "bishop"),
    "r": ("black", "rook"),
    "q": ("black", "queen"),
    "k": ("black", "king"),
}

FILES_NORMAL = "abcdefgh"
FILES_FLIPPED = "hgfedcba"
DETECT_BOARD_PX = 480
CELL_PX = DETECT_BOARD_PX // 8


def detect_board_flip(gray_board: np.ndarray) -> bool:
    """Detect if board is flipped 180° by comparing mean brightness.

    White pieces are brighter than black pieces on average.
    In normal orientation (white at bottom), bottom half is brighter.
    Returns True when flipped (black at bottom) is detected.
    """
    h, w = gray_board.shape[:2]
    cell_h, cell_w = h // 8, w // 8
    top_means, bot_means = [], []
    for row in range(8):
        for col in range(8):
            cell = gray_board[row*cell_h:(row+1)*cell_h, col*cell_w:(col+1)*cell_w]
            std = float(cell.std())
            if std < 25:
                continue
            mean = float(cell.mean())
            if row < 4:
                top_means.append(mean)
            else:
                bot_means.append(mean)
    if not top_means or not bot_means:
        return False
    top_avg = sum(top_means) / len(top_means)
    bot_avg = sum(bot_means) / len(bot_means)
    return top_avg > bot_avg


def has_trained_templates() -> bool:
    return any(_template_path(sym).exists() for sym in PIECES)


def _template_path(sym: str) -> Path:
    prefixed = f"{'w' if sym.isupper() else 'b'}_{sym.upper()}.png"
    lowercased = f"{'w' if sym.isupper() else 'b'}{sym.lower()}.png"
    candidates = [
        TEMPLATE_DIR / prefixed,
        TEMPLATE_DIR / lowercased,
        TEMPLATE_DIR / f"{sym}.png",
    ]
    alt_dir = TEMPLATE_DIR.parent / "chess-vision-ai" / "piece_templates"
    candidates.extend(
        [
            alt_dir / prefixed,
            alt_dir / lowercased,
            alt_dir / f"{sym}.png",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def sanitize_board_map(bmap: dict[str, Optional[str]]) -> tuple[dict[str, Optional[str]], list[str]]:
    max_pieces = {
        "K": 1, "k": 1, "Q": 9, "q": 9, "R": 10, "r": 10,
        "B": 10, "b": 10, "N": 10, "n": 10, "P": 8, "p": 8,
    }
    tracker: dict[str, int] = {}
    clean: dict[str, Optional[str]] = {}
    warnings: list[str] = []
    for sq in sorted(bmap.keys()):
        piece = bmap[sq]
        if not piece:
            clean[sq] = None
            continue
        # Remove pawns on illegal ranks
        rank = int(sq[1])
        if piece == "P" and rank == 1:
            warnings.append(f"White pawn on rank 1 at {sq} removed")
            clean[sq] = None
            continue
        if piece == "p" and rank == 8:
            warnings.append(f"Black pawn on rank 8 at {sq} removed")
            clean[sq] = None
            continue
        count = tracker.get(piece, 0)
        if count < max_pieces.get(piece, 1):
            tracker[piece] = count + 1
            clean[sq] = piece
        else:
            warnings.append(f"Extra {piece} at {sq} removed")
            clean[sq] = None
    if tracker.get("K", 0) == 0:
        clean["e1"] = "K"
        warnings.append("White King added at e1")
    if tracker.get("k", 0) == 0:
        clean["e8"] = "k"
        warnings.append("Black King added at e8")
    return clean, warnings


def _infer_castling(bmap: dict[str, Optional[str]]) -> str:
    rights = ""
    if bmap.get("e1") == "K":
        if bmap.get("h1") == "R":
            rights += "K"
        if bmap.get("a1") == "R":
            rights += "Q"
    if bmap.get("e8") == "k":
        if bmap.get("h8") == "r":
            rights += "k"
        if bmap.get("a8") == "r":
            rights += "q"
    return rights or "-"


def board_map_to_fen(
    bmap: dict[str, Optional[str]],
    *,
    active: chess.Color = chess.WHITE,
    ep: str = "-",
    halfmove: int = 0,
    fullmove: int = 1,
) -> str:
    castling = _infer_castling(bmap)
    rows: list[str] = []
    for rank in range(8, 0, -1):
        empty = 0
        row = ""
        for file_char in FILES_NORMAL:
            piece = bmap.get(f"{file_char}{rank}")
            if piece:
                if empty:
                    row += str(empty)
                    empty = 0
                row += piece
            else:
                empty += 1
        if empty:
            row += str(empty)
        rows.append(row)
    turn = "w" if active == chess.WHITE else "b"
    return f"{'/'.join(rows)} {turn} {castling} {ep} {halfmove} {fullmove}"


def generate_synthetic_templates(cell_size: int = 60) -> None:
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    size = cell_size
    half = cell_size // 2

    def base(color: str) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int, int, int]]:
        img = np.zeros((size, size, 4), dtype=np.uint8)
        fill = (255, 255, 255, 230) if color == "white" else (30, 30, 30, 230)
        outline = (40, 40, 40, 255) if color == "white" else (200, 200, 200, 255)
        return img, fill, outline

    def pawn(color: str) -> np.ndarray:
        img, fill, outline = base(color)
        cx = half
        cv2.circle(img, (cx, size // 4), size // 6, fill, -1)
        cv2.circle(img, (cx, size // 4), size // 6, outline, 1)
        cv2.rectangle(img, (cx - size // 8, size // 4), (cx + size // 8, size * 3 // 4), fill, -1)
        cv2.rectangle(img, (cx - size // 8, size // 4), (cx + size // 8, size * 3 // 4), outline, 1)
        cv2.rectangle(img, (cx - size // 5, size * 3 // 4), (cx + size // 5, size - 4), fill, -1)
        cv2.rectangle(img, (cx - size // 5, size * 3 // 4), (cx + size // 5, size - 4), outline, 1)
        return img

    def rook(color: str) -> np.ndarray:
        img, fill, outline = base(color)
        cx = half
        pts = np.array(
            [
                [cx - size // 4, 4],
                [cx - size // 5, 4],
                [cx - size // 5, size // 5],
                [cx - size // 8, size // 5],
                [cx - size // 8, 4],
                [cx + size // 8, 4],
                [cx + size // 8, size // 5],
                [cx + size // 5, size // 5],
                [cx + size // 5, 4],
                [cx + size // 4, 4],
                [cx + size // 4, size * 3 // 4],
                [cx - size // 4, size * 3 // 4],
            ],
            np.int32,
        )
        cv2.fillPoly(img, [pts], fill)
        cv2.polylines(img, [pts], True, outline, 1)
        cv2.rectangle(img, (cx - size // 4, size * 3 // 4), (cx + size // 4, size - 4), fill, -1)
        cv2.rectangle(img, (cx - size // 4, size * 3 // 4), (cx + size // 4, size - 4), outline, 1)
        return img

    def knight(color: str) -> np.ndarray:
        img, fill, outline = base(color)
        cx = half
        pts = np.array(
            [
                [cx - size // 5, size - 4],
                [cx + size // 4, size - 4],
                [cx + size // 4, size // 3],
                [cx + size // 8, size // 5],
                [cx - size // 8, size // 6],
                [cx - size // 5, size // 3],
            ],
            np.int32,
        )
        cv2.fillPoly(img, [pts], fill)
        cv2.polylines(img, [pts], True, outline, 1)
        cv2.circle(img, (cx - size // 10, size // 5), size // 8, fill, -1)
        cv2.circle(img, (cx - size // 10, size // 5), size // 8, outline, 1)
        return img

    def bishop(color: str) -> np.ndarray:
        img, fill, outline = base(color)
        cx = half
        cv2.circle(img, (cx, size // 4), size // 6, fill, -1)
        cv2.circle(img, (cx, size // 4), size // 6, outline, 1)
        pts = np.array([[cx, size // 8], [cx + size // 5, size * 2 // 3], [cx - size // 5, size * 2 // 3]], np.int32)
        cv2.fillPoly(img, [pts], fill)
        cv2.polylines(img, [pts], True, outline, 1)
        cv2.rectangle(img, (cx - size // 4, size * 2 // 3), (cx + size // 4, size - 4), fill, -1)
        cv2.rectangle(img, (cx - size // 4, size * 2 // 3), (cx + size // 4, size - 4), outline, 1)
        return img

    def queen(color: str) -> np.ndarray:
        img, fill, outline = base(color)
        cx = half
        crown = np.array(
            [
                [cx - size // 3, size // 2],
                [cx - size // 4, size // 5],
                [cx, size // 3],
                [cx + size // 4, size // 5],
                [cx + size // 3, size // 2],
            ],
            np.int32,
        )
        cv2.fillPoly(img, [crown], fill)
        cv2.polylines(img, [crown], False, outline, 1)
        for px in (cx - size // 3, cx - size // 4, cx, cx + size // 4, cx + size // 3):
            cv2.circle(img, (px, size // 5), 3, fill, -1)
            cv2.circle(img, (px, size // 5), 3, outline, 1)
        cv2.ellipse(img, (cx, size * 2 // 3), (size // 3, size // 6), 0, 0, 360, fill, -1)
        cv2.ellipse(img, (cx, size * 2 // 3), (size // 3, size // 6), 0, 0, 360, outline, 1)
        cv2.rectangle(img, (cx - size // 3, size * 3 // 4), (cx + size // 3, size - 4), fill, -1)
        cv2.rectangle(img, (cx - size // 3, size * 3 // 4), (cx + size // 3, size - 4), outline, 1)
        return img

    def king(color: str) -> np.ndarray:
        img, fill, outline = base(color)
        cx = half
        cv2.rectangle(img, (cx - 2, 4), (cx + 2, size // 5), fill, -1)
        cv2.rectangle(img, (cx - 2, 4), (cx + 2, size // 5), outline, 1)
        cv2.rectangle(img, (cx - size // 8, size // 8), (cx + size // 8, size // 6), fill, -1)
        cv2.rectangle(img, (cx - size // 8, size // 8), (cx + size // 8, size // 6), outline, 1)
        cv2.ellipse(img, (cx, size * 5 // 12), (size // 4, size // 5), 0, 0, 360, fill, -1)
        cv2.ellipse(img, (cx, size * 5 // 12), (size // 4, size // 5), 0, 0, 360, outline, 1)
        cv2.rectangle(img, (cx - size // 3, size * 3 // 5), (cx + size // 3, size - 4), fill, -1)
        cv2.rectangle(img, (cx - size // 3, size * 3 // 5), (cx + size // 3, size - 4), outline, 1)
        return img

    makers = {
        "pawn": pawn,
        "rook": rook,
        "knight": knight,
        "bishop": bishop,
        "queen": queen,
        "king": king,
    }
    for sym, (color, ptype) in PIECES.items():
        cv2.imwrite(str(TEMPLATE_DIR / f"{sym}.png"), makers[ptype](color))


def load_templates(cell_size: int) -> dict[str, np.ndarray]:
    if not all(_template_path(sym).exists() for sym in PIECES):
        generate_synthetic_templates(cell_size)
    templates: dict[str, np.ndarray] = {}
    for sym in PIECES:
        path = _template_path(sym)
        if not path.exists():
            continue
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is not None:
            templates[sym] = cv2.resize(img, (cell_size, cell_size))
    return templates


class BoardDetector:
    """Local board recognition via batched template matching (CPU/CUDA/OpenCL)."""

    def __init__(self) -> None:
        self._cached: dict[int, PreparedTemplates] = {}

    def backend_name(self) -> str:
        return match_backend_name()

    def _get_prepared(self, cell_px: int) -> PreparedTemplates:
        if cell_px not in self._cached:
            templates = load_templates(cell_px)
            self._cached[cell_px] = prepare_templates(templates, cell_px)
        return self._cached[cell_px]

    def detect(self, image_bgr: np.ndarray, *, flipped: bool = False, board_px: int = DETECT_BOARD_PX) -> tuple[dict[str, Optional[str]], list[str]]:
        cell_px = board_px // 8
        board = cv2.resize(image_bgr, (board_px, board_px), interpolation=cv2.INTER_AREA)
        prepared = self._get_prepared(cell_px)
        gray_board = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        cells_gray = (
            gray_board.reshape(8, cell_px, 8, cell_px)
            .transpose(0, 2, 1, 3)
            .reshape(64, cell_px, cell_px)
            .astype(np.float32)
        )

        symbols = batch_match_from_gray(cells_gray, prepared)
        raw: dict[str, Optional[str]] = {}
        for index, sym in enumerate(symbols):
            row, col = divmod(index, 8)
            if flipped:
                rank = row + 1
                file_char = FILES_FLIPPED[col]
            else:
                rank = 8 - row
                file_char = FILES_NORMAL[col]
            raw[f"{file_char}{rank}"] = sym
        return sanitize_board_map(raw)

    def detect_multi_scale(self, image_bgr: np.ndarray, *, flipped: bool = False) -> tuple[dict[str, Optional[str]], list[str]]:
        best: Optional[dict[str, Optional[str]]] = None
        best_warnings: list[str] = []
        best_count = -1
        for scale in [480, 688]:
            try:
                result, warnings = self.detect(image_bgr, flipped=flipped, board_px=scale)
                cnt = sum(1 for v in result.values() if v is not None)
                if cnt > best_count:
                    best = result
                    best_warnings = warnings
                    best_count = cnt
            except Exception:
                continue
        if best is None:
            return {}, ["no scale produced output"]
        return best, best_warnings

    def _split_and_identify(
        self,
        image_bgr: np.ndarray,
        height: int,
        width: int,
        flipped: bool,
    ) -> dict[str, Optional[str]]:
        raise NotImplementedError("use detect()")

    def _identify(self, cell: np.ndarray) -> Optional[str]:
        raise NotImplementedError("use detect()")

    @staticmethod
    def _ncc(cell: np.ndarray, tmpl: np.ndarray) -> float:
        raise NotImplementedError("use fast_matcher")


def match_template_cli() -> None:
    """CLI entry: capture screen region, run template matching, print FEN."""
    import argparse

    parser = argparse.ArgumentParser(description="Chess board template matching")
    parser.add_argument("--image", "-i", help="Path to board image (omit for screen select)")
    parser.add_argument("--flipped", "-f", action="store_true", help="Board is flipped (black at bottom)")
    parser.add_argument("--show", "-s", action="store_true", help="Show preview window")
    parser.add_argument("--json", "-j", action="store_true", help="Output JSON")
    args = parser.parse_args()

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"Error: could not read {args.image}")
            return
    else:
        import mss
        with mss.MSS() if hasattr(mss, "MSS") else mss.mss() as sct:
            monitor = sct.monitors[1]
            shot = sct.grab(monitor)
            img = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)
        print("Captured full screen — for best results crop to just the board.")

    result = match_template(img, flipped=args.flipped)
    if args.json:
        import json
        print(json.dumps({
            "fen": result.fen,
            "board_fen": result.board_fen,
            "confidence": round(result.confidence, 3),
            "pieces": sum(1 for v in result.board_map.values() if v is not None),
        }))
    else:
        print(f"FEN:         {result.fen}")
        print(f"Board FEN:   {result.board_fen}")
        print(f"Confidence:  {result.confidence:.0%}")
        print(f"Pieces:      {sum(1 for v in result.board_map.values() if v is not None)}/64")
        if result.warnings:
            print(f"Warnings:    {'; '.join(result.warnings)}")

    if args.show:
        preview_matches(img, result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def preview_matches(
    image: np.ndarray,
    result: MatchResult,
    *,
    cell_size: int = CELL_PX,
    detect_board_px: int = DETECT_BOARD_PX,
    block: bool = True,
) -> None:
    """Show the detection result in an OpenCV window for visual debugging."""
    preview = cv2.resize(image, (detect_board_px, detect_board_px), interpolation=cv2.INTER_AREA)

    for index, sym in enumerate(result.symbols):
        row, col = divmod(index, 8)
        y1, x1 = row * cell_size, col * cell_size
        label = sym if sym else "?"
        color = (0, 200, 0) if sym else (0, 0, 200)
        cv2.putText(preview, label, (x1 + 4, y1 + cell_size - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.rectangle(preview, (x1, y1), (x1 + cell_size, y1 + cell_size), color, 1)

    cv2.imshow("Template Match Result", preview)
    cv2.setWindowProperty("Template Match Result", cv2.WND_PROP_TOPMOST, 1)
    if block:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


@dataclass
class MatchResult:
    board_map: dict[str, Optional[str]]
    fen: str
    board_fen: str
    symbols: list[Optional[str]]
    confidence: float
    warnings: list[str] = field(default_factory=list)


def match_template(
    image: Optional[np.ndarray] = None,
    *,
    flipped: bool = False,
    cell_size: int = CELL_PX,
    detect_board_px: int = DETECT_BOARD_PX,
) -> MatchResult:
    """Match chess piece templates on a board image (or capture screen if no image given).

    Args:
        image: BGR image of the chess board (just the 8x8 grid of squares).
               If None, captures the whole screen (you'll need to crop manually later).
        flipped: True if black pieces are at the bottom (board flipped view).
        cell_size: Size to resize each cell/template to for matching.
        detect_board_px: Size to resize the input board image before processing.

    Returns:
        MatchResult with board_map (sq→piece), FENs, and per-cell symbol list.

    Example:
        >>> result = match_template()
        >>> print(result.fen)
        rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
    """
    if image is None:
        import mss
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            shot = sct.grab(monitor)
            image = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)

    board = cv2.resize(image, (detect_board_px, detect_board_px), interpolation=cv2.INTER_AREA)

    templates = load_templates(cell_size)
    prepared = prepare_templates(templates, cell_size)

    gray_board = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    cells_gray = (
        gray_board.reshape(8, cell_size, 8, cell_size)
        .transpose(0, 2, 1, 3)
        .reshape(64, cell_size, cell_size)
        .astype(np.float32)
    )

    symbols = batch_match_from_gray(cells_gray, prepared)

    raw: dict[str, Optional[str]] = {}
    for index, sym in enumerate(symbols):
        row, col = divmod(index, 8)
        if flipped:
            rank = row + 1
            file_char = FILES_FLIPPED[col]
        else:
            rank = 8 - row
            file_char = FILES_NORMAL[col]
        raw[f"{file_char}{rank}"] = sym

    clean_map, warnings = sanitize_board_map(raw)
    fen = board_map_to_fen(clean_map)
    board_fen = fen.split()[0]

    confidence = sum(1 for v in clean_map.values() if v is not None) / max(len(clean_map), 1)

    return MatchResult(
        board_map=clean_map,
        fen=fen,
        board_fen=board_fen,
        symbols=symbols,
        confidence=confidence,
        warnings=warnings,
    )


if __name__ == "__main__":
    match_template_cli()
