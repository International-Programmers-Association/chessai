from __future__ import annotations

import math
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import chess

try:
    from PIL import Image, ImageTk
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

ASSETS_DIR = Path(__file__).resolve().parent.parent / "board" if not getattr(sys, 'frozen', False) else Path(sys._MEIPASS) / "board"
PIECE_DIR = Path(__file__).resolve().parent.parent / "pieces" if not getattr(sys, 'frozen', False) else Path(sys._MEIPASS) / "pieces"

# Runtime overrides (set by app config)
CURRENT_PIECE_DIR: Optional[str] = None
BOARD_LIGHT: Optional[str] = None
BOARD_DARK: Optional[str] = None
ARROW_MY_COLOR: Optional[str] = None
ARROW_OPP_COLOR: Optional[str] = None

PIECE_UNICODE = {
    "P": "\u2659", "N": "\u2658", "B": "\u2657", "R": "\u2656", "Q": "\u2655", "K": "\u2654",
    "p": "\u265F", "n": "\u265E", "b": "\u265D", "r": "\u265C", "q": "\u265B", "k": "\u265A",
}

PIECE_PNG = {
    "P": "wp.png", "N": "wn.png", "B": "wb.png", "R": "wr.png", "Q": "wq.png", "K": "wk.png",
    "p": "bp.png", "n": "bn.png", "b": "bb.png", "r": "br.png", "q": "bq.png", "k": "bk.png",
}

LIGHT = "#E9E5D4"
DARK = "#6A994F"
SELECTED = "#f6f669"
LAST_MOVE = "#CDD26A"
HIGHLIGHT = "#7FC97F"
MY_ARROW = "#58D90F"
OPP_ARROW = "#E53935"

def _color(base: str, override: Optional[str]) -> str:
    return override if override is not None else base


@dataclass(frozen=True)
class MoveArrow:
    move: chess.Move
    color: str
    label: str = ""


class BoardWidget(tk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        *,
        square_size: int = 64,
        on_move: Optional[Callable[[chess.Move], None]] = None,
        flip: bool = False,
    ) -> None:
        super().__init__(master, bg="#312e2b")
        self.square_size = square_size
        self.on_move = on_move
        self.flip = flip
        self.board = chess.Board()
        self.interactive = True
        self.selected: Optional[chess.Square] = None
        self.legal_targets: set[chess.Square] = set()
        self.last_move: Optional[chess.Move] = None
        self.suggestion_arrows: list[MoveArrow] = []
        self.show_arrows = True
        self._piece_images: dict[str, tk.PhotoImage] = {}

        size = square_size * 8
        self.canvas = tk.Canvas(
            self,
            width=size,
            height=size,
            highlightthickness=0,
            bg="#312e2b",
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_click)
        self._load_piece_images(square_size)
        self._draw()

    def _load_piece_images(self, size: int) -> None:
        if not _HAS_PIL:
            return
        piece_dir = Path(CURRENT_PIECE_DIR) if CURRENT_PIECE_DIR else PIECE_DIR
        for sym, filename in PIECE_PNG.items():
            path = piece_dir / filename
            if not path.exists():
                continue
            try:
                img = Image.open(str(path)).convert("RGBA")
                img = img.resize((size, size), Image.LANCZOS)
                self._piece_images[sym] = ImageTk.PhotoImage(img)
            except Exception:
                pass

    def set_board(self, board: chess.Board, *, keep_arrows: bool = False) -> None:
        self.board = board.copy()
        self.selected = None
        self.legal_targets = set()
        if not keep_arrows:
            self.suggestion_arrows = []
        self._draw()

    def set_flip(self, flip: bool) -> None:
        if self.flip == flip:
            return
        self.flip = flip
        self._draw()

    def set_last_move(self, move: Optional[chess.Move]) -> None:
        self.last_move = move
        self._draw()

    def set_suggestion_arrows(self, arrows: list[MoveArrow]) -> None:
        self.suggestion_arrows = arrows
        self._draw()

    def set_show_arrows(self, show: bool) -> None:
        self.show_arrows = show
        self._draw()

    def set_interactive(self, interactive: bool) -> None:
        self.interactive = interactive
        if not interactive:
            self.selected = None
            self.legal_targets = set()
        self.canvas.config(cursor="arrow" if not interactive else "")
        self._draw()

    def _center(self, square: chess.Square) -> tuple[float, float]:
        x0, y0 = self._coords(square)
        s = self.square_size
        return x0 + s / 2, y0 + s / 2

    def _draw_arrow(
        self,
        move: chess.Move,
        color: str,
        *,
        offset_index: int = 0,
    ) -> None:
        x0, y0 = self._center(move.from_square)
        x1, y1 = self._center(move.to_square)
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1:
            return

        inset = self.square_size * 0.2
        x0 += dx / length * inset
        y0 += dy / length * inset
        x1 -= dx / length * inset
        y1 -= dy / length * inset

        if offset_index:
            nx, ny = -dy / length, dx / length
            shift = offset_index * 5
            x0 += nx * shift
            y0 += ny * shift
            x1 += nx * shift
            y1 += ny * shift

        self.canvas.create_line(
            x0,
            y0,
            x1,
            y1,
            fill=color,
            width=10,
            arrow=tk.LAST,
            arrowshape=(14, 16, 6),
            capstyle=tk.ROUND,
            joinstyle=tk.ROUND,
            smooth=True,
        )

    def _square_at(self, x: int, y: int) -> Optional[chess.Square]:
        col = x // self.square_size
        row = y // self.square_size
        if not (0 <= col < 8 and 0 <= row < 8):
            return None
        if self.flip:
            file_i = col
            rank_i = row
        else:
            file_i = col
            rank_i = 7 - row
        return chess.square(file_i, rank_i)

    def _coords(self, square: chess.Square) -> tuple[int, int]:
        file_i = chess.square_file(square)
        rank_i = chess.square_rank(square)
        if self.flip:
            # Vertical flip: a1 at top-left, rank 1 at top, a-h left-to-right
            col = file_i
            row = rank_i
        else:
            # Standard: a8 at top-left, rank 8 at top, a-h left-to-right
            col = file_i
            row = 7 - rank_i
        x0 = col * self.square_size
        y0 = row * self.square_size
        return x0, y0

    def _on_click(self, event: tk.Event) -> None:
        if not self.interactive:
            return
        square = self._square_at(event.x, event.y)
        if square is None:
            return

        piece = self.board.piece_at(square)
        if self.selected is None:
            if piece and piece.color == self.board.turn:
                self.selected = square
                self.legal_targets = {
                    m.to_square
                    for m in self.board.legal_moves
                    if m.from_square == square
                }
                self._draw()
            return

        if square == self.selected:
            self.selected = None
            self.legal_targets = set()
            self._draw()
            return

        move = self._find_move(self.selected, square)
        if move is None:
            if piece and piece.color == self.board.turn:
                self.selected = square
                self.legal_targets = {
                    m.to_square
                    for m in self.board.legal_moves
                    if m.from_square == square
                }
            else:
                self.selected = None
                self.legal_targets = set()
            self._draw()
            return

        self.board.push(move)
        self.last_move = move
        self.selected = None
        self.legal_targets = set()
        self._draw()
        if self.on_move:
            self.on_move(move)

    def _find_move(self, from_sq: chess.Square, to_sq: chess.Square) -> Optional[chess.Move]:
        for move in self.board.legal_moves:
            if move.from_square == from_sq and move.to_square == to_sq:
                return move
        return None

    def _draw(self) -> None:
        self.canvas.delete("all")
        s = self.square_size
        for square in chess.SQUARES:
            x0, y0 = self._coords(square)
            x1, y1 = x0 + s, y0 + s
            display_col = x0 // s
            display_row = y0 // s
            is_light = (display_col + display_row) % 2 == 0
            color = _color(LIGHT, BOARD_LIGHT) if is_light else _color(DARK, BOARD_DARK)

            if self.last_move and square in (
                self.last_move.from_square,
                self.last_move.to_square,
            ):
                color = LAST_MOVE
            if square == self.selected:
                color = SELECTED
            elif square in self.legal_targets:
                color = HIGHLIGHT

            self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")

            if square in self.legal_targets:
                cx, cy = x0 + s // 2, y0 + s // 2
                r = s // 8
                self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill="#333333")

            piece = self.board.piece_at(square)
            if piece:
                sym = piece.symbol()
                if sym in self._piece_images:
                    img = self._piece_images[sym]
                    self.canvas.create_image(x0, y0, anchor=tk.NW, image=img)
                else:
                    glyph = PIECE_UNICODE[sym]
                    fg = "#ffffff" if piece.color else "#000000"
                    self.canvas.create_text(
                        x0 + s // 2,
                        y0 + s // 2,
                        text=glyph,
                        font=("Segoe UI Symbol", int(s * 0.72)),
                        fill=fg,
                    )

        if self.show_arrows:
            for i, arrow in enumerate(self.suggestion_arrows):
                col = _color(arrow.color, ARROW_MY_COLOR if arrow.color == MY_ARROW else ARROW_OPP_COLOR if arrow.color == OPP_ARROW else None)
                self._draw_arrow(arrow.move, col, offset_index=i)

        labels = "hgfedcba" if self.flip else "abcdefgh"
        rank_nums = range(1, 9) if self.flip else range(8, 0, -1)
        for i, label in enumerate(labels):
            is_light = (i + 7) % 2 == 0
            self.canvas.create_text(
                i * s + s // 2,
                8 * s - 8,
                text=label,
                fill="#000000" if is_light else "#ffffff",
                font=("Segoe UI", 9),
            )
        for i, num in enumerate(rank_nums):
            is_light = (0 + i) % 2 == 0
            self.canvas.create_text(
                8,
                i * s + s // 2,
                text=str(num),
                fill="#000000" if is_light else "#ffffff",
                font=("Segoe UI", 9),
            )
