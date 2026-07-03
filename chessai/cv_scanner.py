from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Union

import chess
import cv2
import numpy as np

from chessai.local_detector import board_map_to_fen, detect_board_flip
from chessai.position_tracker import infer_turn_from_position
from chessai.classifier import UNetBoardClassifier

CHESSVISION_PREDICT_URL = "http://app.chessvision.ai/predict"
MAX_IMAGE_SIZE = 1800


@dataclass
class ScanResult:
    success: bool
    fen: Optional[str] = None
    board_fen: Optional[str] = None
    turn: Optional[chess.Color] = None
    board_orientation: str = "predict"
    error: Optional[str] = None
    source: str = "api"
    warnings: list[str] = field(default_factory=list)


def scan_board(
    image_bgr: np.ndarray,
    *,
    current_player: Literal["white", "black"] = "white",
    flipped: Union[bool, Literal["auto"]] = False,
    prefer_local: bool = True,
    allow_api: bool = True,
    timeout: float = 8.0,
) -> ScanResult:
    """Scan board: fast local matching first, optional Chessvision API fallback."""
    local = ScanResult(success=False, error="local scan skipped", source="local")
    if prefer_local:
        local = scan_board_local(image_bgr, current_player=current_player, flipped=flipped)
        if local.success:
            return local
    if not allow_api:
        return local
    api = scan_board_image(
        image_bgr,
        current_player=current_player,
        cropped=True,
        timeout=timeout,
    )
    if api.success:
        return api
    if prefer_local:
        return local
    return api


def _remap_flipped(board_map: dict[str, Optional[str]]) -> dict[str, Optional[str]]:
    """Remap ranks for a flipped board (black at bottom)."""
    result = {}
    for sq, sym in board_map.items():
        file_char = sq[0]
        rank = int(sq[1])
        new_rank = 9 - rank  # flip rank: 1↔8, 2↔7, etc.
        result[f"{file_char}{new_rank}"] = sym
    return result


def scan_board_local(
    image_bgr: np.ndarray,
    *,
    current_player: Literal["white", "black"] = "white",
    flipped: Union[bool, Literal["auto"]] = False,
) -> ScanResult:
    """Recognize position using U-Net board classifier."""
    warnings: list[str] = []
    try:
        if flipped == "auto":
            gray_preview = cv2.cvtColor(
                cv2.resize(image_bgr, (480, 480)), cv2.COLOR_BGR2GRAY
            )
            flipped = detect_board_flip(gray_preview)
        flipped = bool(flipped)

        board_map = UNetBoardClassifier().classify_board(image_bgr)
        board_masked: dict[str, Optional[str]] = {}
        for sq, p in board_map.items():
            board_masked[sq] = p if p else None

        if flipped:
            board_masked = _remap_flipped(board_masked)

        board_map, board, warnings = _build_valid_position(board_masked, warnings)
        if board is None:
            return ScanResult(success=False, error="illegal position", source="local", warnings=warnings)
        board_fen = board.board_fen()
        fen = board.fen()
    except ValueError as exc:
        return ScanResult(success=False, error=str(exc), source="local", warnings=warnings)
    except Exception as exc:
        return ScanResult(success=False, error=str(exc), source="local")

    piece_count = sum(1 for c in board.board_fen() if c.isalpha())
    if piece_count < 4:
        return ScanResult(success=False, error="too few pieces detected", source="local", warnings=warnings)

    orientation = "black" if flipped else "white"
    return ScanResult(
        success=True,
        fen=fen,
        board_fen=board.board_fen(),
        turn=board.turn,
        board_orientation=orientation,
        source="local",
        warnings=warnings,
    )


def _build_valid_position(
    board_map: dict[str, Optional[str]],
    warnings: list[str],
) -> tuple[dict[str, Optional[str]], Optional[chess.Board], list[str]]:
    """Repeatedly remove least reliable pieces until position is legal."""
    squares = [sq for sq, v in board_map.items() if v is not None]

    # If UNet found many pieces, try just adding missing kings first
    if len(squares) >= 20:
        for turn_side in (chess.WHITE, chess.BLACK):
            try:
                fen = board_map_to_fen(board_map, active=turn_side)
                board = chess.Board(fen)
                if board.is_valid():
                    return board_map, board, warnings
            except ValueError:
                continue

    for _ in range(len(squares) + 1):
        for turn_side in (chess.WHITE, chess.BLACK):
            try:
                fen = board_map_to_fen(board_map, active=turn_side)
                board = chess.Board(fen)
                if board.is_valid():
                    return board_map, board, warnings
            except ValueError:
                continue
        # Remove the least certain piece — start with pawns on extreme ranks
        for sq in sorted(board_map.keys(), reverse=True):
            p = board_map.get(sq)
            if p in ("P", "p") and (sq[1] in ("1", "8")):
                warnings.append(f"Removed {p} at {sq} to fix legality")
                board_map[sq] = None
                break
        else:
            removals = [sq for sq in sorted(board_map.keys(), reverse=True)
                        if board_map.get(sq) is not None]
            if not removals:
                break
            sq = removals[0]
            warnings.append(f"Removed {board_map[sq]} at {sq} to fix legality")
            board_map[sq] = None
    return board_map, None, warnings


def scan_board_image(
    image_bgr: np.ndarray,
    *,
    current_player: Literal["white", "black"] = "white",
    cropped: bool = True,
    timeout: float = 20.0,
) -> ScanResult:
    """Scan a chess board image using Chessvision.ai-style API (image -> FEN)."""
    data_url = _encode_image(image_bgr)
    payload = {
        "image": data_url,
        "cropped": cropped,
        "current_player": current_player,
        "board_orientation": "predict",
        "predict_turn": True,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        CHESSVISION_PREDICT_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return ScanResult(success=False, error=str(exc))
    except json.JSONDecodeError:
        return ScanResult(success=False, error="invalid API response")

    if not raw.get("success") or not raw.get("result"):
        return ScanResult(success=False, error="position not recognized")

    fen = _normalize_fen(raw["result"], raw.get("turn"), current_player)
    if fen is None:
        return ScanResult(success=False, error="invalid FEN from scanner")

    try:
        board = chess.Board(fen)
    except ValueError as exc:
        return ScanResult(success=False, error=str(exc))

    if not board.is_valid():
        return ScanResult(success=False, error="illegal position")

    board_fen = board.board_fen()
    inferred_turn = infer_turn_from_position(board_fen)
    if board.turn != inferred_turn:
        parts = fen.split()
        parts[1] = "w" if inferred_turn == chess.WHITE else "b"
        fen = " ".join(parts[:6]) if len(parts) >= 6 else f"{board_fen} {'w' if inferred_turn == chess.WHITE else 'b'} - - 0 1"
        board = chess.Board(fen)

    return ScanResult(
        success=True,
        fen=fen,
        board_fen=board.board_fen(),
        turn=board.turn,
        board_orientation=str(raw.get("board_orientation", "predict")),
    )


def _encode_image(image_bgr: np.ndarray) -> str:
    h, w = image_bgr.shape[:2]
    scale = min(1.0, MAX_IMAGE_SIZE / max(h, w))
    if scale < 1.0:
        image_bgr = cv2.resize(
            image_bgr,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise ValueError("failed to encode image")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _normalize_fen(
    result: str,
    turn_raw: Optional[str],
    current_player: str,
) -> Optional[str]:
    result = result.strip()
    # Chessvision API returns underscores instead of spaces: board_b_KQkq_-_0_1
    if " " not in result and "_" in result:
        result = result.replace("_", " ")

    parts = result.split()
    if len(parts) == 1:
        turn = turn_raw or current_player
        turn_char = "w" if str(turn).lower() in ("white", "w") else "b"
        return f"{parts[0]} {turn_char} - - 0 1"
    if len(parts) >= 6:
        return " ".join(parts[:6])
    if len(parts) >= 2:
        board, turn = parts[0], parts[1]
        turn_char = "w" if turn.lower() in ("white", "w") else "b"
        castling = parts[2] if len(parts) > 2 else "-"
        ep = parts[3] if len(parts) > 3 else "-"
        half = parts[4] if len(parts) > 4 else "0"
        full = parts[5] if len(parts) > 5 else "1"
        return f"{board} {turn_char} {castling} {ep} {half} {full}"
    return None


def frame_changed(prev: Optional[np.ndarray], current: np.ndarray, threshold: float = 1.0) -> bool:
    if prev is None:
        return True
    if prev.shape != current.shape:
        return True
    step_y = max(1, prev.shape[0] // 32)
    step_x = max(1, prev.shape[1] // 32)
    sample_prev = prev[::step_y, ::step_x]
    sample_curr = current[::step_y, ::step_x]
    diff = float(np.mean(cv2.absdiff(sample_prev, sample_curr)))
    return diff >= threshold
