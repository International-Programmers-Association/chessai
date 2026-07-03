from __future__ import annotations

import sys
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np

from chessai.local_detector import (
    CELL_PX,
    DETECT_BOARD_PX,
    FILES_NORMAL,
    sanitize_board_map,
    load_templates,
)
from chessai.fast_matcher import batch_match_from_gray, prepare_templates

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

MODELS_DIR = (Path(sys._MEIPASS) / "chessai" / "models" if getattr(sys, 'frozen', False)
              else Path(__file__).resolve().parent / "models")

# Global overrides
CURRENT_UNET_MODEL: Optional[str] = None
FORCE_DEVICE: Optional[str] = None  # "cpu" or "cuda" — None = auto
CLASSES = ["K", "Q", "R", "B", "N", "P", "k", "q", "r", "b", "n", "p"]


class PieceClassifier(ABC):
    @abstractmethod
    def classify_board(self, board_bgr: np.ndarray) -> dict[str, Optional[str]]:
        ...


class TemplateMatcher(PieceClassifier):
    """Template matching — exact pixel comparison with piece PNGs. Fast and reliable."""

    def __init__(self, cell_size: int = CELL_PX, detect_board_px: int = DETECT_BOARD_PX):
        self.cell_size = cell_size
        self.detect_board_px = detect_board_px
        self._prepared = None
        self._cached_size = 0

    def classify_board(self, board_bgr: np.ndarray) -> dict[str, Optional[str]]:
        board = cv2.resize(board_bgr, (self.detect_board_px, self.detect_board_px), interpolation=cv2.INTER_AREA)

        if self._prepared is None or self._cached_size != self.cell_size:
            templates = load_templates(self.cell_size)
            self._prepared = prepare_templates(templates, self.cell_size)
            self._cached_size = self.cell_size

        gray_board = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        cells = (
            gray_board.reshape(8, self.cell_size, 8, self.cell_size)
            .transpose(0, 2, 1, 3)
            .reshape(64, self.cell_size, self.cell_size)
            .astype(np.float32)
        )

        symbols = batch_match_from_gray(cells, self._prepared)
        raw: dict[str, Optional[str]] = {}
        for idx, sym in enumerate(symbols):
            row, col = divmod(idx, 8)
            rank = 8 - row
            raw[f"{FILES_NORMAL[col]}{rank}"] = sym

        clean, _ = sanitize_board_map(raw)
        return clean


class CNNClassifier(PieceClassifier):
    """Two-pass CNN with batched GPU inference.

    Binary empty/piece model filters cells, then 12-class MobileNetV3-Small
    classifies the rest.  Both models run on GPU if available.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or str(MODELS_DIR / "chess_mobilenet.pt")
        self.binary_path = str(MODELS_DIR / "chess_binary.pt")
        self.classes_path = str(MODELS_DIR / "classes.json")
        self._model = None
        self._binary_model = None
        self._classes: list[str] = []
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available — install torch or use TemplateMatcher")
        if not Path(self.model_path).exists() or not Path(self.binary_path).exists():
            raise FileNotFoundError(f"Models not found: {self.model_path}, {self.binary_path}")
        import json
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = torch.jit.load(self.model_path, map_location=self._device)
        self._model.eval()
        self._binary_model = torch.jit.load(self.binary_path, map_location=self._device)
        self._binary_model.eval()
        with open(self.classes_path) as f:
            self._classes = json.load(f)

    def classify_board(self, board_bgr: np.ndarray) -> dict[str, Optional[str]]:
        self._load()

        board = cv2.resize(board_bgr, (480, 480))

        # Extract all 64 cells into a batch — preprocessing MUST match training:
        #   60×60 BGR crop → resize to 64×64 BGR → grayscale
        cells = np.zeros((64, 3, 64, 64), dtype=np.float32)
        for i in range(64):
            row, col = divmod(i, 8)
            cell = board[row * 60:(row + 1) * 60, col * 60:(col + 1) * 60]
            cell_resized = cv2.resize(cell, (64, 64))
            gray = cv2.cvtColor(cell_resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            cells[i] = np.stack([gray, gray, gray], axis=0)

        batch = torch.from_numpy(cells).to(self._device)

        with torch.no_grad():
            # Pass 1: binary model on all 64 cells
            bin_logits = self._binary_model(batch)
            bin_probs = torch.softmax(bin_logits, dim=1)
            piece_mask = bin_probs[:, 1] >= 0.5

            # Pass 2: piece model on non-empty cells only
            piece_indices = piece_mask.nonzero(as_tuple=False).squeeze(1)
            raw = {f"{FILES_NORMAL[col]}{8 - row}": None for row in range(8) for col in range(8)}

            if len(piece_indices) > 0:
                piece_batch = batch[piece_indices]
                logits = self._model(piece_batch)
                probs = torch.softmax(logits, dim=1)
                best_idx = logits.argmax(dim=1)
                best_prob = probs.max(dim=1).values

                for j, idx in enumerate(piece_indices.tolist()):
                    if best_prob[j].item() >= 0.4 and best_idx[j].item() < len(self._classes):
                        sym = self._classes[best_idx[j].item()]
                        row, col = divmod(int(idx), 8)
                        raw[f"{FILES_NORMAL[col]}{8 - row}"] = sym

        clean, _ = sanitize_board_map(raw)
        return clean


class UNetBoardClassifier(PieceClassifier):
    """U-Net: full 480x480 board → 8x8 piece types. Trained with random scales."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or CURRENT_UNET_MODEL or str(MODELS_DIR / "chess_unet.pt")
        self._model = None
        self._device = None
        self._classes = CLASSES

    def _load(self) -> None:
        if self._model is not None:
            return
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if FORCE_DEVICE == "cuda" and not torch.cuda.is_available():
            self._device = torch.device("cpu")
        elif FORCE_DEVICE:
            self._device = torch.device(FORCE_DEVICE)
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = torch.jit.load(self.model_path, map_location=self._device)
        self._model.eval()

    def classify_board(self, board_bgr: np.ndarray, conf_thresh: float = 0.5) -> dict[str, Optional[str]]:
        self._load()
        h, w = board_bgr.shape[:2]
        size = min(h, w)
        x, y = (w - size) // 2, (h - size) // 2
        cropped = board_bgr[y:y+size, x:x+size]
        board = cv2.resize(cropped, (480, 480))
        gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(self._device)

        with torch.no_grad():
            output = self._model(tensor).squeeze(0)  # (13, 8, 8)
            probs = torch.softmax(output, dim=0)

        raw: dict[str, Optional[str]] = {}
        for row in range(8):
            for col in range(8):
                sq = f"{FILES_NORMAL[col]}{8 - row}"
                cls = probs[:, row, col].argmax().item()
                raw[sq] = None if cls == 12 else self._classes[cls]

        return raw
