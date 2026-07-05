from __future__ import annotations
from pathlib import Path
from typing import Optional
import json
import cv2
import numpy as np
import torch
import torch.nn as nn

MODELS_DIR = Path(__file__).resolve().parent / "models"
CLASSES = ["K", "Q", "R", "B", "N", "P", "k", "q", "r", "b", "n", "p"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)
HOG_FEATURES = 1764  # computed from HOG params below


def _make_hog() -> cv2.HOGDescriptor:
    win_size = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    return cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)


def extract_hog(cell_64x64: np.ndarray) -> np.ndarray:
    hog = _make_hog()
    gray = cv2.cvtColor(cell_64x64, cv2.COLOR_BGR2GRAY) if cell_64x64.ndim == 3 else cell_64x64
    feats = hog.compute(gray)
    return feats.flatten().astype(np.float32)


def extract_hog_batch(boards_bgr: np.ndarray) -> np.ndarray:
    n = boards_bgr.shape[0]
    feats = np.zeros((n, HOG_FEATURES), dtype=np.float32)
    hog = _make_hog()
    for i in range(n):
        gray = cv2.cvtColor(boards_bgr[i], cv2.COLOR_BGR2GRAY) if boards_bgr[i].ndim == 3 else boards_bgr[i]
        feats[i] = hog.compute(gray).flatten()
    return feats


class HOGMLP(nn.Module):
    def __init__(self, input_dim: int = HOG_FEATURES, hidden: int = 512, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class BinaryHOGMLP(nn.Module):
    def __init__(self, input_dim: int = HOG_FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


class HOGClassifier:
    """HOG + small neural network for piece classification."""

    def __init__(self, model_path: Optional[str] = None, binary_path: Optional[str] = None):
        self.model_path = model_path or str(MODELS_DIR / "hog_piece.pt")
        self.binary_path = binary_path or str(MODELS_DIR / "hog_binary.pt")
        self._piece_model: Optional[HOGMLP] = None
        self._binary_model: Optional[BinaryHOGMLP] = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load(self) -> None:
        if self._piece_model is not None:
            return
        if not Path(self.model_path).exists() or not Path(self.binary_path).exists():
            raise FileNotFoundError(f"HOG models not found: {self.model_path}, {self.binary_path}")
        self._piece_model = HOGMLP()
        self._piece_model.load_state_dict(torch.load(self.model_path, map_location=self._device, weights_only=True))
        self._piece_model.to(self._device).eval()
        self._binary_model = BinaryHOGMLP()
        self._binary_model.load_state_dict(torch.load(self.binary_path, map_location=self._device, weights_only=True))
        self._binary_model.to(self._device).eval()

    def _extract_hog_cells(self, board_bgr: np.ndarray) -> np.ndarray:
        """Resize to 480, apply CLAHE, extract HOG from all 64 cells."""
        board = cv2.resize(board_bgr, (480, 480))
        feats = np.zeros((64, HOG_FEATURES), dtype=np.float32)
        hog = _make_hog()

        for i in range(64):
            row, col = divmod(i, 8)
            cell = board[row * 60:(row + 1) * 60, col * 60:(col + 1) * 60]
            cell = cv2.resize(cell, (64, 64))

            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if cell.ndim == 3 else cell
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            gray = clahe.apply(gray)

            feats[i] = hog.compute(gray).flatten()
        return feats

    @staticmethod
    def _find_board_quad(image_bgr: np.ndarray) -> np.ndarray:
        """Find largest chessboard-like quadrilateral and warp to 480×480."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        # Edge detection
        edges = cv2.Canny(gray, 30, 150)
        # Dilate to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours[:20]:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
                # Order: top-left, top-right, bottom-right, bottom-left
                rect = np.zeros((4, 2), dtype=np.float32)
                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]
                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]
                dst = np.array([[0, 0], [480, 0], [480, 480], [0, 480]], dtype=np.float32)
                try:
                    M = cv2.getPerspectiveTransform(rect, dst)
                    warped = cv2.warpPerspective(image_bgr, M, (480, 480))
                    # Verify it looks like a board: 8x8 alternating pattern
                    gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                    cell_h, cell_w = 60, 60
                    colors = []
                    for r in range(8):
                        for c in range(8):
                            cell = gray_w[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
                            colors.append(float(cell.mean()))
                    colors = np.array(colors)
                    # Board should have 2 distinct color clusters (light/dark squares)
                    mean1, mean2 = sorted(colors)[::32]  # sample extremes
                    if abs(mean1 - mean2) > 30:
                        return warped
                except cv2.error:
                    continue

        # Fallback: center-crop to square and resize
        h, w = image_bgr.shape[:2]
        size = min(h, w)
        x = (w - size) // 2
        y = (h - size) // 2
        return cv2.resize(image_bgr[y:y+size, x:x+size], (480, 480))

    def classify_board(self, image_bgr: np.ndarray) -> dict[str, Optional[str]]:
        self._load()

        board_480 = self._find_board_quad(image_bgr)
        feats = self._extract_hog_cells(board_480)
        batch = torch.from_numpy(feats).to(self._device)

        from chessai.local_detector import FILES_NORMAL
        with torch.no_grad():
            bin_logits = self._binary_model(batch)
            bin_probs = torch.softmax(bin_logits, dim=1)
            logits = self._piece_model(batch)
            probs = torch.softmax(logits, dim=1)
            best_idx = logits.argmax(dim=1)
            best_prob = probs.max(dim=1).values

        raw: dict[str, Optional[str]] = {}
        for row in range(8):
            for col in range(8):
                idx = row * 8 + col
                sq = f"{FILES_NORMAL[col]}{8 - row}"
                raw[sq] = None
                bin_conf = bin_probs[idx, 1].item()
                if bin_conf >= 0.5 and best_prob[idx].item() >= 0.4 and best_idx[idx].item() < NUM_CLASSES:
                    raw[sq] = IDX_TO_CLASS[best_idx[idx].item()]

        from chessai.local_detector import sanitize_board_map
        clean, _ = sanitize_board_map(raw)
        return clean


def train_hog_models(synthetic_dir: str, real_meta_path: str, synthetic_meta_path: str):
    """Train HOG piece + binary models on synthetic + real data."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Group samples by source directory
    dirs: dict[str, list[dict]] = {}
    synth_base = Path(synthetic_dir)
    if Path(synthetic_meta_path).exists():
        with open(synthetic_meta_path) as f:
            dirs[str(synth_base)] = json.load(f)
    real_base = Path(real_meta_path).parent
    if Path(real_meta_path).exists():
        with open(real_meta_path) as f:
            dirs[str(real_base)] = json.load(f)
    total_boards = sum(len(v) for v in dirs.values())
    print(f"Total boards: {total_boards}")

    from torch.utils.data import Dataset, DataLoader

    class HOGDataset(Dataset):
        def __init__(self, dirs_dict: dict[str, list[dict]]):
            self.data: list[tuple[np.ndarray, int]] = []
            files = "abcdefgh"
            hog = _make_hog()
            for base_dir, samples_list in dirs_dict.items():
                base = Path(base_dir)
                for s in samples_list:
                    img_path = base / s["image"]
                    if not img_path.exists():
                        continue
                    board = cv2.imread(str(img_path))
                    if board is None:
                        continue
                    board = cv2.resize(board, (480, 480))
                    for row in range(8):
                        for col in range(8):
                            sq = f"{files[col]}{8 - row}"
                            piece = s["cells"].get(sq, ".")
                            if piece == ".":
                                continue
                            if piece not in CLASS_TO_IDX:
                                continue
                            y1, y2 = row * 60, (row + 1) * 60
                            x1, x2 = col * 60, (col + 1) * 60
                            cell = board[y1:y2, x1:x2]
                            cell = cv2.resize(cell, (64, 64))
                            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
                            feat = hog.compute(gray).flatten().astype(np.float32)
                            self.data.append((feat, CLASS_TO_IDX[piece]))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            feat, label = self.data[idx]
            return torch.from_numpy(feat), torch.tensor(label, dtype=torch.long)

    class BinaryHOGDataset(Dataset):
        def __init__(self, dirs_dict: dict[str, list[dict]]):
            self.data: list[tuple[np.ndarray, int]] = []
            files = "abcdefgh"
            hog = _make_hog()
            for base_dir, samples_list in dirs_dict.items():
                base = Path(base_dir)
                for s in samples_list:
                    img_path = base / s["image"]
                    if not img_path.exists():
                        continue
                    board = cv2.imread(str(img_path))
                    if board is None:
                        continue
                    board = cv2.resize(board, (480, 480))
                    for row in range(8):
                        for col in range(8):
                            sq = f"{files[col]}{8 - row}"
                            piece = s["cells"].get(sq, ".")
                            y1, y2 = row * 60, (row + 1) * 60
                            x1, x2 = col * 60, (col + 1) * 60
                            cell = board[y1:y2, x1:x2]
                            cell = cv2.resize(cell, (64, 64))
                            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
                            feat = hog.compute(gray).flatten().astype(np.float32)
                            label = 1 if piece != "." else 0
                            self.data.append((feat, label))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            feat, label = self.data[idx]
            return torch.from_numpy(feat), torch.tensor(label, dtype=torch.long)

    BATCH_SIZE = 256

    print("Loading piece dataset...")
    piece_ds = HOGDataset(dirs)
    piece_loader = DataLoader(piece_ds, batch_size=BATCH_SIZE, shuffle=True)
    print(f"  {len(piece_ds)} piece samples")

    piece_model = HOGMLP().to(device)
    optimizer = torch.optim.Adam(piece_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("Training piece model...")
    piece_model.train()
    for epoch in range(20):
        total = correct = 0
        for feats, labels in piece_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            out = piece_model(feats)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            _, pred = out.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
        acc = correct / total * 100
        print(f"  Epoch {epoch+1}: {acc:.1f}%")
        if acc >= 99.5:
            break

    torch.save(piece_model.state_dict(), MODELS_DIR / "hog_piece.pt")
    print(f"Saved: {MODELS_DIR / 'hog_piece.pt'}")

    print("\nLoading binary dataset...")
    bin_ds = BinaryHOGDataset(dirs)
    bin_loader = DataLoader(bin_ds, batch_size=BATCH_SIZE, shuffle=True)
    print(f"  {len(bin_ds)} binary samples")

    bin_model = BinaryHOGMLP().to(device)
    bin_optimizer = torch.optim.Adam(bin_model.parameters(), lr=1e-3)
    bin_criterion = nn.CrossEntropyLoss()

    print("Training binary model...")
    bin_model.train()
    for epoch in range(15):
        total = correct = 0
        for feats, labels in bin_loader:
            feats, labels = feats.to(device), labels.to(device)
            bin_optimizer.zero_grad()
            out = bin_model(feats)
            loss = bin_criterion(out, labels)
            loss.backward()
            bin_optimizer.step()
            _, pred = out.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
        acc = correct / total * 100
        print(f"  Epoch {epoch+1}: {acc:.1f}%")
        if acc >= 99.5:
            break

    torch.save(bin_model.state_dict(), MODELS_DIR / "hog_binary.pt")
    print(f"Saved: {MODELS_DIR / 'hog_binary.pt'}")
    print("\nDone!")


def train_hog_models_real(real_meta_path: str):
    """Train HOG models on real boards only (from datasetgw)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    real_base = Path(real_meta_path).parent
    with open(real_meta_path) as f:
        samples = json.load(f)
    print(f"Real boards: {len(samples)}")

    from torch.utils.data import Dataset, DataLoader

    class RealHOGDataset(Dataset):
        def __init__(self, samples_list, base_dir: Path):
            self.data: list[tuple[np.ndarray, int]] = []
            files = "abcdefgh"
            hog = _make_hog()
            for s in samples_list:
                img_path = base_dir / s["image"]
                board = cv2.imread(str(img_path))
                if board is None:
                    continue
                board = cv2.resize(board, (480, 480))
                for row in range(8):
                    for col in range(8):
                        sq = f"{files[col]}{8 - row}"
                        piece = s["cells"].get(sq, ".")
                        if piece == ".":
                            continue
                        if piece not in CLASS_TO_IDX:
                            continue
                        y1, y2 = row * 60, (row + 1) * 60
                        x1, x2 = col * 60, (col + 1) * 60
                        cell = board[y1:y2, x1:x2]
                        cell = cv2.resize(cell, (64, 64))
                        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
                        feat = hog.compute(gray).flatten().astype(np.float32)
                        self.data.append((feat, CLASS_TO_IDX[piece]))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            feat, label = self.data[idx]
            return torch.from_numpy(feat), torch.tensor(label, dtype=torch.long)

    class RealBinaryHOGDataset(Dataset):
        def __init__(self, samples_list, base_dir: Path):
            self.data: list[tuple[np.ndarray, int]] = []
            files = "abcdefgh"
            hog = _make_hog()
            for s in samples_list:
                img_path = base_dir / s["image"]
                board = cv2.imread(str(img_path))
                if board is None:
                    continue
                board = cv2.resize(board, (480, 480))
                for row in range(8):
                    for col in range(8):
                        sq = f"{files[col]}{8 - row}"
                        piece = s["cells"].get(sq, ".")
                        y1, y2 = row * 60, (row + 1) * 60
                        x1, x2 = col * 60, (col + 1) * 60
                        cell = board[y1:y2, x1:x2]
                        cell = cv2.resize(cell, (64, 64))
                        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
                        feat = hog.compute(gray).flatten().astype(np.float32)
                        label = 1 if piece != "." else 0
                        self.data.append((feat, label))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            feat, label = self.data[idx]
            return torch.from_numpy(feat), torch.tensor(label, dtype=torch.long)

    BATCH_SIZE = 256

    print("\nLoading real piece dataset...")
    piece_ds = RealHOGDataset(samples, real_base)
    piece_loader = DataLoader(piece_ds, batch_size=BATCH_SIZE, shuffle=True)
    print(f"  {len(piece_ds)} piece samples")

    piece_model = HOGMLP().to(device)
    optimizer = torch.optim.Adam(piece_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("Training piece model on real data...")
    piece_model.train()
    for epoch in range(50):
        total = correct = 0
        for feats, labels in piece_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            out = piece_model(feats)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            _, pred = out.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
        acc = correct / total * 100
        print(f"  Epoch {epoch+1}: {acc:.1f}%")
        if epoch >= 10 and acc >= 99.0:
            break

    torch.save(piece_model.state_dict(), MODELS_DIR / "hog_piece.pt")
    print(f"Saved: {MODELS_DIR / 'hog_piece.pt'}")

    print("\nLoading real binary dataset...")
    bin_ds = RealBinaryHOGDataset(samples, real_base)
    bin_loader = DataLoader(bin_ds, batch_size=BATCH_SIZE, shuffle=True)
    print(f"  {len(bin_ds)} binary samples")

    bin_model = BinaryHOGMLP().to(device)
    bin_optimizer = torch.optim.Adam(bin_model.parameters(), lr=1e-3)
    bin_criterion = nn.CrossEntropyLoss()

    print("Training binary model on real data...")
    bin_model.train()
    for epoch in range(50):
        total = correct = 0
        for feats, labels in bin_loader:
            feats, labels = feats.to(device), labels.to(device)
            bin_optimizer.zero_grad()
            out = bin_model(feats)
            loss = bin_criterion(out, labels)
            loss.backward()
            bin_optimizer.step()
            _, pred = out.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
        acc = correct / total * 100
        print(f"  Epoch {epoch+1}: {acc:.1f}%")
        if epoch >= 10 and acc >= 99.0:
            break

    torch.save(bin_model.state_dict(), MODELS_DIR / "hog_binary.pt")
    print(f"Saved: {MODELS_DIR / 'hog_binary.pt'}")
    print("\nDone!")
