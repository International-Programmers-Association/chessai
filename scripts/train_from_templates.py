"""Train CNN on actual piece PNGs composited onto board square backgrounds."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "piece_templates"
MODELS_DIR = Path(__file__).resolve().parent.parent / "chessai" / "models"

CLASSES = ["K", "Q", "R", "B", "N", "P", "k", "q", "r", "b", "n", "p"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Board square colors (BGR)
LIGHT_BGR = np.array([212, 229, 233], dtype=np.uint8)    # #E9E5D4
DARK_BGR = np.array([79, 153, 106], dtype=np.uint8)      # #6A994F
CELL_SIZE = 64


def hex_to_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


# Extra background colors for augmentation (slightly shifted)
BG_VARIANTS = [
    0.0,       # exact
    0.04,      # slight darker
    -0.04,     # slight lighter
    0.08,      # a bit darker
]

LIGHT_BGR_F = LIGHT_BGR.astype(np.float32)
DARK_BGR_F = DARK_BGR.astype(np.float32)


def composite_piece(piece_img: np.ndarray, bg_color: np.ndarray,
                    offset_x: int = 0, offset_y: int = 0,
                    scale: float = 1.0) -> np.ndarray:
    """Composite piece PNG onto 64x64 background with optional shift/scale."""
    canvas = np.full((CELL_SIZE, CELL_SIZE, 3), bg_color, dtype=np.uint8)

    h, w = piece_img.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    if new_w % 2: new_w += 1
    if new_h % 2: new_h += 1
    resized = cv2.resize(piece_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center with offset
    cx = (CELL_SIZE - new_w) // 2 + offset_x
    cy = (CELL_SIZE - new_h) // 2 + offset_y

    # Blend with alpha
    if resized.shape[2] == 4:
        alpha = resized[:, :, 3].astype(np.float32) / 255.0
        bgr = resized[:, :, :3].astype(np.float32)
        y1, y2 = max(0, cy), min(CELL_SIZE, cy + new_h)
        x1, x2 = max(0, cx), min(CELL_SIZE, cx + new_w)
        ry1, ry2 = max(0, -cy), min(new_h, CELL_SIZE - cy)
        rx1, rx2 = max(0, -cx), min(new_w, CELL_SIZE - cx)

        region = canvas[y1:y2, x1:x2].astype(np.float32)
        blended = bgr[ry1:ry2, rx1:rx2] * alpha[ry1:ry2, rx1:rx2, None] + \
                  region * (1.0 - alpha[ry1:ry2, rx1:rx2, None])
        canvas[y1:y2, x1:x2] = blended.clip(0, 255).astype(np.uint8)
    else:
        cy = max(0, cy)
        cx = max(0, cx)
        canvas[cy:cy+new_h, cx:cx+new_w] = resized[:, :, :3]

    return canvas


def make_empty_cell(bg_bgr: np.ndarray, factor: float = 0.0) -> np.ndarray:
    """Create empty square with optional brightness shift."""
    shifted = np.clip(bg_bgr.astype(np.float32) * (1 + factor), 0, 255).astype(np.uint8)
    return np.full((CELL_SIZE, CELL_SIZE, 3), shifted, dtype=np.uint8)


class TemplatePGNDataset(Dataset):
    def __init__(self, num_per_class: int = 3000, empty_ratio: float = 0.5):
        self.data: list[tuple[np.ndarray, int]] = []

        pieces: dict[str, np.ndarray] = {}
        sym_to_file = {
            "K": "wk.png", "Q": "wq.png", "R": "wr.png",
            "B": "wb.png", "N": "wn.png", "P": "wp.png",
            "k": "bk.png", "q": "bq.png", "r": "br.png",
            "b": "bb.png", "n": "bn.png", "p": "bp.png",
        }
        for sym, fn in sym_to_file.items():
            img = cv2.imread(str(TEMPLATE_DIR / fn), cv2.IMREAD_UNCHANGED)
            if img is not None:
                pieces[sym] = img

        rng = np.random.default_rng(42)

        # Generate piece cells
        for sym, img in pieces.items():
            bg_colors = [LIGHT_BGR, DARK_BGR]
            for _ in range(num_per_class):
                bg = bg_colors[rng.integers(0, 2)]
                offset_x = int(rng.integers(-3, 4))
                offset_y = int(rng.integers(-3, 4))
                scale = rng.uniform(0.55, 1.0)
                cell = composite_piece(img, bg, offset_x, offset_y, scale)
                self.data.append((cell, CLASS_TO_IDX[sym]))

        # Generate empty cells
        num_empty = int(len(self.data) * empty_ratio / (1 - empty_ratio))
        for _ in range(num_empty):
            bg = [LIGHT_BGR, DARK_BGR][rng.integers(0, 2)]
            factor = rng.uniform(-0.08, 0.08)
            cell = make_empty_cell(bg, factor)
            self.data.append((cell, -1))  # -1 = empty

        rng.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_bgr, label = self.data[idx]

        # Random brightness shift
        if np.random.rand() < 0.5:
            delta = np.random.uniform(-0.15, 0.15) * 255
            img_bgr = np.clip(img_bgr.astype(np.float32) + delta, 0, 255).astype(np.uint8)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.stack([gray, gray, gray], axis=0))
        return tensor, torch.tensor(max(0, label), dtype=torch.long)


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    NUM_PER_CLASS = 4000
    print(f"Generating {NUM_PER_CLASS} cells per class...")
    full_ds = TemplatePGNDataset(num_per_class=NUM_PER_CLASS, empty_ratio=0.5)
    print(f"Total cells: {len(full_ds)}")

    # Split into piece-only and binary
    piece_data = [(img, lbl) for img, lbl in full_ds.data if lbl >= 0]
    bin_data = [(img, 0 if lbl == -1 else 1) for img, lbl in full_ds.data]

    rng = np.random.default_rng(42)
    rng.shuffle(piece_data)
    rng.shuffle(bin_data)

    # 90/10 split
    piece_split = int(len(piece_data) * 0.9)
    bin_split = int(len(bin_data) * 0.9)

    class SimpleDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            img, lbl = self.data[idx]
            if np.random.rand() < 0.5:
                delta = np.random.uniform(-0.15, 0.15) * 255
                img = np.clip(img.astype(np.float32) + delta, 0, 255).astype(np.uint8)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            return torch.from_numpy(np.stack([gray, gray, gray], axis=0)), torch.tensor(lbl, dtype=torch.long)

    train_piece = SimpleDataset(piece_data[:piece_split])
    val_piece = SimpleDataset(piece_data[piece_split:])
    train_bin = SimpleDataset(bin_data[:bin_split])
    val_bin = SimpleDataset(bin_data[bin_split:])

    train_piece_loader = DataLoader(train_piece, batch_size=128, shuffle=True)
    val_piece_loader = DataLoader(val_piece, batch_size=128)
    train_bin_loader = DataLoader(train_bin, batch_size=256, shuffle=True)
    val_bin_loader = DataLoader(val_bin, batch_size=256)

    print(f"Piece: train={len(train_piece)}, val={len(val_piece)}")
    print(f"Binary: train={len(train_bin)}, val={len(val_bin)}")

    # --- Piece model ---
    print("\n--- Training piece model ---")
    piece_model = models.mobilenet_v3_small(weights=None).to(DEVICE)
    in_features = piece_model.classifier[0].in_features
    piece_model.classifier = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 12),
    ).to(DEVICE)

    piece_criterion = nn.CrossEntropyLoss()
    piece_optimizer = optim.Adam(piece_model.parameters(), lr=5e-4)
    piece_scheduler = optim.lr_scheduler.CosineAnnealingLR(piece_optimizer, T_max=20)

    best_acc = 0.0
    for epoch in range(1, 21):
        piece_model.train()
        correct, total = 0, 0
        for images, labels in train_piece_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            piece_optimizer.zero_grad()
            outputs = piece_model(images)
            loss = piece_criterion(outputs, labels)
            loss.backward()
            piece_optimizer.step()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total * 100

        piece_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_piece_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = piece_model(images)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total * 100
        piece_scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
        if epoch % 3 == 0 or val_acc == best_acc:
            print(f"  Epoch {epoch:2d}  train={train_acc:.1f}%  val={val_acc:.1f}%  best={best_acc:.1f}%")

    piece_model.eval()
    example = torch.randn(1, 3, 64, 64)
    scripted = torch.jit.trace(piece_model.cpu(), example)
    scripted.save(str(MODELS_DIR / "chess_mobilenet.pt"))
    with open(MODELS_DIR / "classes.json", "w") as f:
        json.dump(CLASSES, f)
    print(f"Saved: {MODELS_DIR / 'chess_mobilenet.pt'} (best: {best_acc:.1f}%)")

    # --- Binary model ---
    print("\n--- Training binary model ---")
    bin_model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(128, 256, 3, padding=1),
        nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(256, 256, 3, padding=1),
        nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(256, 2),
    ).to(DEVICE)

    bin_criterion = nn.CrossEntropyLoss()
    bin_optimizer = optim.Adam(bin_model.parameters(), lr=5e-4)
    bin_scheduler = optim.lr_scheduler.CosineAnnealingLR(bin_optimizer, T_max=20)

    best_bin = 0.0
    for epoch in range(1, 21):
        bin_model.train()
        correct, total = 0, 0
        for images, labels in train_bin_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            bin_optimizer.zero_grad()
            outputs = bin_model(images)
            loss = bin_criterion(outputs, labels)
            loss.backward()
            bin_optimizer.step()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total * 100

        bin_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_bin_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = bin_model(images)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total * 100
        bin_scheduler.step()

        if val_acc > best_bin:
            best_bin = val_acc
        if epoch % 3 == 0 or val_acc == best_bin:
            print(f"  Binary epoch {epoch:2d}  train={train_acc:.1f}%  val={val_acc:.1f}%  best={best_bin:.1f}%")

    bin_model.eval()
    bin_scripted = torch.jit.trace(bin_model.cpu(), example)
    bin_scripted.save(str(MODELS_DIR / "chess_binary.pt"))
    print(f"Saved: {MODELS_DIR / 'chess_binary.pt'} (best: {best_bin:.1f}%)")
    print("\nDone!")


if __name__ == "__main__":
    main()
