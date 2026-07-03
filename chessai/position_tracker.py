from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import chess


def infer_turn_from_position(board_fen: str) -> chess.Color:
    """Guess side to move by comparing piece placement to the starting position."""
    if board_fen == chess.STARTING_BOARD_FEN:
        return chess.WHITE

    start = chess.Board()
    board = chess.Board(f"{board_fen} w - - 0 1")

    white_moves = _estimate_side_moves(start, board, chess.WHITE)
    black_moves = _estimate_side_moves(start, board, chess.BLACK)

    if white_moves > black_moves:
        return chess.BLACK
    if black_moves > white_moves:
        return chess.WHITE

    return _infer_turn_by_check_legality(board_fen)


def _estimate_side_moves(start: chess.Board, board: chess.Board, color: chess.Color) -> int:
    if color == chess.WHITE:
        home_ranks = (0, 1)
        castling_king_squares = (chess.G1, chess.C1)
        pawn_advanced = lambda rank: rank > 1
    else:
        home_ranks = (6, 7)
        castling_king_squares = (chess.G8, chess.C8)
        pawn_advanced = lambda rank: rank < 6

    departures = 0
    for sq in chess.SQUARES:
        rank = chess.square_rank(sq)
        if rank not in home_ranks:
            continue
        start_piece = start.piece_at(sq)
        if not start_piece or start_piece.color != color:
            continue
        if board.piece_at(sq) != start_piece:
            departures += 1

    king_sq = board.king(color)
    if king_sq in castling_king_squares and departures >= 2:
        departures -= 1

    advanced_pawns = sum(
        1
        for sq in chess.SQUARES
        if (piece := board.piece_at(sq))
        and piece.color == color
        and piece.piece_type == chess.PAWN
        and pawn_advanced(chess.square_rank(sq))
    )

    return max(departures, advanced_pawns)


def _infer_turn_by_check_legality(board_fen: str) -> chess.Color:
    white_board = chess.Board(f"{board_fen} w - - 0 1")
    black_board = chess.Board(f"{board_fen} b - - 0 1")
    white_ok = white_board.is_valid()
    black_ok = black_board.is_valid()
    if white_ok and not black_ok:
        return chess.WHITE
    if black_ok and not white_ok:
        return chess.BLACK
    return chess.WHITE


def resolve_turn(
    board_fen: str,
    *,
    turn_hint: Optional[chess.Color] = None,
    manual_turn: Optional[chess.Color] = None,
    auto_turn: bool = True,
) -> chess.Color:
    if not auto_turn and manual_turn is not None:
        return manual_turn
    if turn_hint is not None:
        return turn_hint
    return infer_turn_from_position(board_fen)


def is_plausible_board(board_fen: str) -> bool:
    kings = sum(1 for c in board_fen if c in "Kk")
    if kings != 2:
        return False
    pieces = sum(1 for c in board_fen if c.isalpha())
    if pieces < 4:
        return False
    try:
        board = chess.Board(f"{board_fen} w - - 0 1")
    except ValueError:
        return False
    return board.is_valid()


@dataclass
class TrackedPosition:
    board_fen: str
    turn: chess.Color
    last_move: Optional[chess.Move] = None
    last_move_san: str = ""
    ply: int = 0


@dataclass
class MoveDetection:
    move: chess.Move
    mover: chess.Color
    board_fen: str
    turn: chess.Color
    san: str = ""


class PositionTracker:
    """Track position, moves and turn order from visual board changes."""

    def __init__(self) -> None:
        self._last: Optional[TrackedPosition] = None
        self._candidate_fen: Optional[str] = None
        self._candidate_hits = 0

    def reset(self) -> None:
        self._last = None
        self._candidate_fen = None
        self._candidate_hits = 0

    @property
    def last(self) -> Optional[TrackedPosition]:
        return self._last

    def ingest(
        self,
        raw_board_fen: str,
        *,
        changed_squares: Optional[set[chess.Square]] = None,
    ) -> Optional[TrackedPosition]:
        if not is_plausible_board(raw_board_fen):
            return None

        if self._last is None:
            self._last = TrackedPosition(raw_board_fen, chess.WHITE, None, 0)
            self._candidate_fen = None
            self._candidate_hits = 0
            return self._last

        if raw_board_fen == self._last.board_fen:
            return None

        required_hits = 1 if changed_squares and len(changed_squares) <= 3 else 2
        if raw_board_fen == self._candidate_fen:
            self._candidate_hits += 1
        else:
            self._candidate_fen = raw_board_fen
            self._candidate_hits = 1

        if self._candidate_hits < required_hits:
            return None

        move_info = detect_move(self._last.board_fen, self._last.turn, raw_board_fen)
        if move_info is None:
            if _looks_like_new_game(raw_board_fen):
                move_info = MoveDetection(
                    move=chess.Move.null(),
                    mover=chess.BLACK,
                    board_fen=raw_board_fen,
                    turn=chess.WHITE,
                )
            else:
                return None

        ply = self._last.ply + (0 if move_info.move == chess.Move.null() else 1)
        self._last = TrackedPosition(
            board_fen=raw_board_fen,
            turn=move_info.turn,
            last_move=None if move_info.move == chess.Move.null() else move_info.move,
            last_move_san=move_info.san,
            ply=ply,
        )
        self._candidate_fen = None
        self._candidate_hits = 0
        return self._last

    def force(self, board_fen: str, turn: chess.Color) -> TrackedPosition:
        self._last = TrackedPosition(board_fen, turn, None, "", 0)
        self._candidate_fen = None
        self._candidate_hits = 0
        return self._last

    def ingest_scan(
        self,
        board_fen: str,
        turn_hint: Optional[chess.Color] = None,
        *,
        manual_turn: Optional[chess.Color] = None,
        auto_turn: bool = True,
    ) -> Optional[TrackedPosition]:
        """Accept a scan result directly (no multi-frame stability needed)."""
        if not is_plausible_board(board_fen):
            return None
        if self._last and self._last.board_fen == board_fen:
            if not auto_turn and manual_turn is not None and self._last.turn != manual_turn:
                self._last = TrackedPosition(
                    board_fen=board_fen,
                    turn=manual_turn,
                    last_move=self._last.last_move,
                    last_move_san=self._last.last_move_san,
                    ply=self._last.ply,
                )
                return self._last
            return None

        if not auto_turn and manual_turn is not None:
            if self._last is None:
                self._last = TrackedPosition(board_fen, manual_turn)
            else:
                move_info = detect_move(self._last.board_fen, self._last.turn, board_fen)
                if move_info:
                    turn = move_info.turn
                    self._last = TrackedPosition(
                        board_fen=board_fen,
                        turn=turn,
                        last_move=move_info.move,
                        last_move_san=move_info.san,
                        ply=self._last.ply + 1,
                    )
                else:
                    self._last = TrackedPosition(
                        board_fen=board_fen,
                        turn=manual_turn,
                        ply=self._last.ply,
                    )
            self._candidate_fen = None
            self._candidate_hits = 0
            return self._last

        resolved_hint = turn_hint
        if resolved_hint is None:
            resolved_hint = infer_turn_from_position(board_fen)

        if self._last is None:
            self._last = TrackedPosition(board_fen, resolved_hint)
            return self._last

        move_info = detect_move(self._last.board_fen, self._last.turn, board_fen)
        if move_info:
            self._last = TrackedPosition(
                board_fen=board_fen,
                turn=move_info.turn,
                last_move=move_info.move,
                last_move_san=move_info.san,
                ply=self._last.ply + 1,
            )
        elif _looks_like_new_game(board_fen):
            self._last = TrackedPosition(board_fen, chess.WHITE)
        elif turn_hint is not None:
            self._last = TrackedPosition(
                board_fen=board_fen,
                turn=turn_hint,
                ply=self._last.ply,
            )
        else:
            inferred = infer_turn_from_position(board_fen)
            self._last = TrackedPosition(
                board_fen=board_fen,
                turn=inferred,
                ply=self._last.ply,
            )
        self._candidate_fen = None
        self._candidate_hits = 0
        return self._last


def detect_move(
    prev_board_fen: str,
    prev_turn: chess.Color,
    new_board_fen: str,
) -> Optional[MoveDetection]:
    if prev_board_fen == new_board_fen:
        return None

    for turn in (prev_turn, not prev_turn):
        turn_char = "w" if turn == chess.WHITE else "b"
        board = chess.Board(f"{prev_board_fen} {turn_char} - - 0 1")
        for move in board.legal_moves:
            copy = board.copy()
            copy.push(move)
            if copy.board_fen() == new_board_fen:
                return MoveDetection(
                    move=move,
                    mover=turn,
                    board_fen=new_board_fen,
                    turn=not turn,
                    san=board.san(move),
                )

    return None


def infer_turn_from_diff(
    prev_board_fen: str,
    prev_turn: chess.Color,
    new_board_fen: str,
) -> Optional[chess.Color]:
    info = detect_move(prev_board_fen, prev_turn, new_board_fen)
    return info.turn if info else None


def _looks_like_new_game(board_fen: str) -> bool:
    return board_fen == chess.STARTING_BOARD_FEN or sum(
        1 for c in board_fen if c.isalpha()
    ) >= 28
