from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

_MATCH_BACKEND: Optional[str] = None
_USE_OPENCL = False


@dataclass
class PreparedTemplates:
    symbols: list[str]
    size: int
    gray: np.ndarray
    masks: np.ndarray
    norms: np.ndarray


def match_backend_name() -> str:
    global _MATCH_BACKEND
    if _MATCH_BACKEND is None:
        _MATCH_BACKEND = _detect_backend()
    return _MATCH_BACKEND


def _detect_backend() -> str:
    try:
        import cupy as cp  # noqa: F401

        if cp.cuda.runtime.getDeviceCount() > 0:
            return "cuda"
    except Exception:
        pass

    try:
        if cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL():
            return "opencl"
    except Exception:
        pass

    if cv2.ocl.haveOpenCL():
        try:
            cv2.ocl.setUseOpenCL(True)
            if cv2.ocl.useOpenCL():
                return "opencl"
        except Exception:
            pass

    return "cpu"


def prepare_templates(templates: dict[str, np.ndarray], cell_size: int) -> PreparedTemplates:
    symbols = list(templates.keys())
    count = len(symbols)
    gray = np.zeros((count, cell_size, cell_size), dtype=np.float32)
    masks = np.zeros((count, cell_size, cell_size), dtype=np.float32)

    for index, sym in enumerate(symbols):
        tmpl = templates[sym]
        if tmpl.shape[0] != cell_size or tmpl.shape[1] != cell_size:
            tmpl = cv2.resize(tmpl, (cell_size, cell_size), interpolation=cv2.INTER_AREA)
        if tmpl.shape[2] == 4:
            rgb = tmpl[:, :, :3]
            mask = (tmpl[:, :, 3] > 128).astype(np.float32)
        else:
            rgb = tmpl[:, :, :3]
            # Uniform mask — compare whole cell (background removal was unreliable)
            mask = np.ones((cell_size, cell_size), dtype=np.float32)
        gray[index] = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
        masks[index] = mask

    norms = np.zeros(count, dtype=np.float32)
    for index in range(count):
        mask = masks[index]
        weight = mask.sum()
        if weight <= 0:
            continue
        mean = (gray[index] * mask).sum() / weight
        diff = (gray[index] - mean) * mask
        norms[index] = float(np.sqrt((diff**2).sum())) + 1e-9

    return PreparedTemplates(symbols=symbols, size=cell_size, gray=gray, masks=masks, norms=norms)


def _equalize_cells(cells: np.ndarray) -> np.ndarray:
    """Apply CLAHE to each cell individually for better contrast."""
    if cells.size == 0:
        return cells
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    out = cells.copy()
    for i in range(out.shape[0]):
        cell_uint8 = out[i].astype(np.uint8)
        out[i] = clahe.apply(cell_uint8).astype(np.float32)
    return out


def batch_match_from_gray(
    cells_gray: np.ndarray,
    prepared: PreparedTemplates,
    *,
    empty_std_threshold: float = 25.0,
    score_threshold: float = 0.30,
    equalize: bool = True,
) -> list[Optional[str]]:
    if cells_gray.size == 0:
        return []

    cell_count = cells_gray.shape[0]
    size = prepared.size
    cells = cells_gray.astype(np.float32, copy=False)
    if equalize:
        cells = _equalize_cells(cells)
    cell_std = cells.reshape(cell_count, -1).std(axis=1)

    backend = match_backend_name()
    if backend == "cuda":
        scores = _scores_cuda(cells, prepared)
    else:
        scores = _scores_cpu(cells, prepared)

    results: list[Optional[str]] = []
    for index in range(cell_count):
        if cell_std[index] < empty_std_threshold:
            results.append(None)
            continue
        best_idx = int(np.argmax(scores[index]))
        best_score = float(scores[index, best_idx])
        if best_score >= score_threshold:
            results.append(prepared.symbols[best_idx])
            continue
        center = cells[index, size // 5 : size * 4 // 5, size // 5 : size * 4 // 5]
        if center.size == 0 or float(center.std()) < 15:
            results.append(None)
        elif float(center.mean()) > 160:
            results.append("P")
        elif float(center.mean()) < 90:
            results.append("p")
        else:
            results.append(None)
    return results


def batch_match_cells(
    cells_bgr: np.ndarray,
    prepared: PreparedTemplates,
    *,
    empty_std_threshold: float = 25.0,
    score_threshold: float = 0.30,
) -> list[Optional[str]]:
    """Match N cells against prepared templates. Returns list of piece symbols."""
    if cells_bgr.size == 0:
        return []

    size = prepared.size
    cells_gray = _cells_to_gray(cells_bgr, size)
    return batch_match_from_gray(
        cells_gray,
        prepared,
        empty_std_threshold=empty_std_threshold,
        score_threshold=score_threshold,
    )


def _cells_to_gray(cells_bgr: np.ndarray, size: int) -> np.ndarray:
    cell_count = cells_bgr.shape[0]
    if cells_bgr.shape[1] == size and cells_bgr.shape[2] == size:
        side = int(np.sqrt(cell_count))
        if side * side == cell_count:
            board_gray = cv2.cvtColor(
                cells_bgr.reshape(side, size, side, size, 3).reshape(side * size, size, 3),
                cv2.COLOR_BGR2GRAY,
            )
            return board_gray.reshape(cell_count, size, size).astype(np.float32)

    cells_gray = np.zeros((cell_count, size, size), dtype=np.float32)
    for index in range(cell_count):
        cell = cells_bgr[index]
        if cell.shape[0] != size or cell.shape[1] != size:
            cell = cv2.resize(cell, (size, size), interpolation=cv2.INTER_AREA)
        cells_gray[index] = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cells_gray


def _scores_cpu(cells_gray: np.ndarray, prepared: PreparedTemplates) -> np.ndarray:
    cell_count = cells_gray.shape[0]
    template_count = len(prepared.symbols)
    scores = np.zeros((cell_count, template_count), dtype=np.float32)

    for t_idx in range(template_count):
        mask = prepared.masks[t_idx]
        tmpl = prepared.gray[t_idx]
        weight = mask.sum()
        if weight <= 0:
            continue
        tmpl_mean = (tmpl * mask).sum() / weight
        tmpl_diff = (tmpl - tmpl_mean) * mask
        tmpl_norm = prepared.norms[t_idx]

        masked = cells_gray * mask
        cell_means = masked.sum(axis=(1, 2)) / weight
        cell_diff = cells_gray - cell_means[:, None, None]
        cell_diff *= mask
        cell_norms = np.sqrt((cell_diff**2).sum(axis=(1, 2))) + 1e-9
        numer = (cell_diff * tmpl_diff).sum(axis=(1, 2))
        scores[:, t_idx] = numer / (cell_norms * tmpl_norm)

    return scores


def _scores_cuda(cells_gray: np.ndarray, prepared: PreparedTemplates) -> np.ndarray:
    import cupy as cp

    cells = cp.asarray(cells_gray)
    masks = cp.asarray(prepared.masks)
    tmpls = cp.asarray(prepared.gray)
    tmpl_norms = cp.asarray(prepared.norms)

    cell_count = cells.shape[0]
    template_count = masks.shape[0]
    scores = cp.zeros((cell_count, template_count), dtype=cp.float32)

    for t_idx in range(template_count):
        mask = masks[t_idx]
        tmpl = tmpls[t_idx]
        weight = float(mask.sum())
        if weight <= 0:
            continue
        tmpl_mean = (tmpl * mask).sum() / weight
        tmpl_diff = (tmpl - tmpl_mean) * mask
        tmpl_norm = tmpl_norms[t_idx]

        masked = cells * mask
        cell_means = masked.sum(axis=(1, 2)) / weight
        cell_diff = cells - cell_means[:, None, None]
        cell_diff *= mask
        cell_norms = cp.sqrt((cell_diff**2).sum(axis=(1, 2))) + 1e-9
        numer = (cell_diff * tmpl_diff).sum(axis=(1, 2))
        scores[:, t_idx] = numer / (cell_norms * tmpl_norm)

    return cp.asnumpy(scores)
