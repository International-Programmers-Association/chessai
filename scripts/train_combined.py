"""Train CNN on synthetic + real boards with brightness augmentation."""
import json
import sys
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
REAL_DIR = Path(__file__).resolve().parent.parent / "datasetgw"
MODELS_DIR = Path(__file__).resolve().parent.parent / "chessai" / "models"

CLASSES = ["K", "Q", "R", "B", "N", "P", "k", "q", "r", "b", "n", "p"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


def augment_brightness(cell_bgr: np.ndarray) -> np.ndarray:
    """Apply ±20% brightness shift."""
    delta = np.random.uniform(-0.2, 0.2) * 255
    aug = cell_bgr.astype(np.float32) + delta
    return np.clip(aug, 0, 255).astype(np.uint8)


def augment_blur(cell_bgr: np.ndarray) -> np.ndarray:
    """Apply light Gaussian blur (σ up to 0.5)."""
    sigma = np.random.uniform(0, 0.5)
    if sigma < 0.1:
        return cell_bgr
    ksize = int(2 * round(sigma * 3) + 1)  # odd kernel
    ksize = max(3, ksize if ksize % 2 == 1 else ksize + 1)
    return cv2.GaussianBlur(cell_bgr, (ksize, ksize), sigma)


def maybe_augment(cell_bgr: np.ndarray) -> np.ndarray:
    if np.random.rand() < 0.5:
        cell_bgr = augment_brightness(cell_bgr)
    if np.random.rand() < 0.3:
        cell_bgr = augment_blur(cell_bgr)
    return cell_bgr


def load_cells_from(samples: list[dict], base_dir: Path, scale_aug: bool = True) -> list[tuple[np.ndarray, str]]:
    cells: list[tuple[np.ndarray, str]] = []
    files = "abcdefgh"
    for s in samples:
        img_path = base_dir / s["image"]
        board = cv2.imread(str(img_path))
        if board is None:
            continue
        # Random intermediate resize for scale invariance
        if scale_aug:
            h, w = board.shape[:2]
            if h == w:
                inter_size = np.random.randint(400, 800)
                board = cv2.resize(board, (inter_size, inter_size))
        board = cv2.resize(board, (480, 480))
        for row in range(8):
            for col in range(8):
                sq = f"{files[col]}{8 - row}"
                piece = s["cells"].get(sq, ".")
                y1, y2 = row * 60, (row + 1) * 60
                x1, x2 = col * 60, (col + 1) * 60
                cell = board[y1:y2, x1:x2]
                if cell.size == 0:
                    continue
                cell = cv2.resize(cell, (64, 64))
                cells.append((cell, piece))
    return cells


class CombinedDataset(Dataset):
    def __init__(self, cells: list[tuple[np.ndarray, str]], is_piece: bool):
        self.data: list[tuple[np.ndarray, int]] = []
        for cell_bgr, piece in cells:
            if is_piece:
                if piece == "." or piece not in CLASS_TO_IDX:
                    continue
                self.data.append((cell_bgr, CLASS_TO_IDX[piece]))
            else:
                label = 0 if piece == "." else 1
                self.data.append((cell_bgr, label))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_bgr, label = self.data[idx]
        img_bgr = maybe_augment(img_bgr)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.stack([gray, gray, gray], axis=0))
        return tensor, torch.tensor(label, dtype=torch.long)


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load all data
    synthetic = json.load(open(DATASET_DIR / "metadata.json"))
    real = json.load(open(REAL_DIR / "metadata.json"))
    print(f"Synthetic: {len(synthetic)} boards, Real: {len(real)} boards")

    synth_cells = load_cells_from(synthetic, DATASET_DIR)
    real_cells = load_cells_from(real, REAL_DIR)
    all_cells = synth_cells + real_cells
    print(f"Total cells: {len(all_cells)}")

    # Shuffle
    rng = np.random.default_rng(42)
    rng.shuffle(all_cells)

    # Split 90/10
    split = int(len(all_cells) * 0.9)
    train_raw, val_raw = all_cells[:split], all_cells[split:]

    train_piece_ds = CombinedDataset(train_raw, is_piece=True)
    val_piece_ds = CombinedDataset(val_raw, is_piece=True)
    train_bin_ds = CombinedDataset(train_raw, is_piece=False)
    val_bin_ds = CombinedDataset(val_raw, is_piece=False)

    print(f"Train piece: {len(train_piece_ds)}, Val piece: {len(val_piece_ds)}")
    print(f"Train binary: {len(train_bin_ds)}, Val binary: {len(val_bin_ds)}")

    train_piece_loader = DataLoader(train_piece_ds, batch_size=128, shuffle=True)
    val_piece_loader = DataLoader(val_piece_ds, batch_size=128)
    train_bin_loader = DataLoader(train_bin_ds, batch_size=256, shuffle=True)
    val_bin_loader = DataLoader(val_bin_ds, batch_size=256)

    # --- Piece model ---
    print("\n--- Training piece model ---")
    piece_model = models.mobilenet_v3_small(weights='IMAGENET1K_V1').to(DEVICE)
    in_features = piece_model.classifier[0].in_features
    piece_model.classifier = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 12),
    ).to(DEVICE)

    labels = [lbl for _, lbl in train_piece_ds.data]
    counts = Counter(labels)
    class_weight = torch.FloatTensor([
        1.0 / np.sqrt(max(counts.get(CLASS_TO_IDX[c], 1), 1))
        for c in CLASSES
    ]).to(DEVICE)
    class_weight = class_weight / class_weight.sum() * len(CLASSES)
    piece_criterion = nn.CrossEntropyLoss(weight=class_weight)
    piece_optimizer = optim.Adam(piece_model.parameters(), lr=1e-3)
    piece_scheduler = optim.lr_scheduler.CosineAnnealingLR(piece_optimizer, T_max=30)

    best_acc = 0.0
    for epoch in range(1, 31):
        piece_model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_piece_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            piece_optimizer.zero_grad()
            outputs = piece_model(images)
            loss = piece_criterion(outputs, labels)
            loss.backward()
            piece_optimizer.step()
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
        train_acc = train_correct / train_total * 100

        piece_model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_piece_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = piece_model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total * 100
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
    print(f"Saved: {MODELS_DIR / 'chess_mobilenet.pt'} (best val: {best_acc:.1f}%)")

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

    empty_cnt = sum(1 for _, l in train_bin_ds.data if l == 0)
    piece_cnt = sum(1 for _, l in train_bin_ds.data if l == 1)
    print(f"Binary: empty={empty_cnt}, piece={piece_cnt}")

    empty_w = 1.0 / np.sqrt(max(empty_cnt, 1))
    piece_w = 1.0 / np.sqrt(max(piece_cnt, 1))
    bin_weight = torch.FloatTensor([empty_w, piece_w]).to(DEVICE)
    bin_weight = bin_weight / bin_weight.sum() * 2
    bin_criterion = nn.CrossEntropyLoss(weight=bin_weight)
    bin_optimizer = optim.Adam(bin_model.parameters(), lr=5e-4)
    bin_scheduler = optim.lr_scheduler.CosineAnnealingLR(bin_optimizer, T_max=40)

    best_bin = 0.0
    for epoch in range(1, 41):
        bin_model.train()
        train_correct, train_total = 0, 0
        for images, labels in train_bin_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            bin_optimizer.zero_grad()
            outputs = bin_model(images)
            loss = bin_criterion(outputs, labels)
            loss.backward()
            bin_optimizer.step()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
        train_acc = train_correct / train_total * 100

        bin_model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_bin_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = bin_model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total * 100
        bin_scheduler.step()

        if val_acc > best_bin:
            best_bin = val_acc
        if epoch % 3 == 0 or val_acc == best_bin:
            print(f"  Binary epoch {epoch:2d}  train={train_acc:.1f}%  val={val_acc:.1f}%  best={best_bin:.1f}%")

    bin_model.eval()
    bin_scripted = torch.jit.trace(bin_model.cpu(), example)
    bin_scripted.save(str(MODELS_DIR / "chess_binary.pt"))
    print(f"Saved: {MODELS_DIR / 'chess_binary.pt'} (best val: {best_bin:.1f}%)")


if __name__ == "__main__":
    main()
