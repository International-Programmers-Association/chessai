"""Fine-tune CNN on real boards from datasetgw."""
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

REAL_DIR = Path(__file__).resolve().parent.parent / "datasetgw"
MODELS_DIR = Path(__file__).resolve().parent.parent / "chessai" / "models"

CLASSES = ["K", "Q", "R", "B", "N", "P", "k", "q", "r", "b", "n", "p"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


class RealBoardDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.data: list[tuple[np.ndarray, int]] = []
        files = "abcdefgh"
        for sample in samples:
            img_path = REAL_DIR / sample["image"]
            board = cv2.imread(str(img_path))
            if board is None:
                continue
            board = cv2.resize(board, (480, 480))
            for row in range(8):
                for col in range(8):
                    sq = f"{files[col]}{8 - row}"
                    piece = sample["cells"].get(sq, ".")
                    if piece not in CLASS_TO_IDX:
                        continue
                    y1, y2 = row * 60, (row + 1) * 60
                    x1, x2 = col * 60, (col + 1) * 60
                    cell = board[y1:y2, x1:x2]
                    if cell.size == 0:
                        continue
                    cell = cv2.resize(cell, (64, 64))
                    self.data.append((cell, CLASS_TO_IDX[piece]))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, label = self.data[idx]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.stack([gray, gray, gray], axis=0))
        return tensor, torch.tensor(label, dtype=torch.long)


class RealBinaryDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.data: list[tuple[np.ndarray, int]] = []
        files = "abcdefgh"
        for sample in samples:
            img_path = REAL_DIR / sample["image"]
            board = cv2.imread(str(img_path))
            if board is None:
                continue
            board = cv2.resize(board, (480, 480))
            for row in range(8):
                for col in range(8):
                    sq = f"{files[col]}{8 - row}"
                    piece = sample["cells"].get(sq, ".")
                    y1, y2 = row * 60, (row + 1) * 60
                    x1, x2 = col * 60, (col + 1) * 60
                    cell = board[y1:y2, x1:x2]
                    if cell.size == 0:
                        continue
                    cell = cv2.resize(cell, (64, 64))
                    label = 0 if piece == "." else 1
                    self.data.append((cell, label))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, label = self.data[idx]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.stack([gray, gray, gray], axis=0))
        return tensor, torch.tensor(label, dtype=torch.long)


def load_piece_model():
    """Create MobileNetV3-Small with same architecture as trained model."""
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 12),
    )
    return model


def load_binary_model():
    return nn.Sequential(
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
    )


def copy_weights(script_model, new_model):
    """Copy weights from TorchScript model to new nn.Module."""
    new_state = {}
    for name, param in script_model.named_parameters():
        if name in dict(new_model.named_parameters()):
            new_state[name] = param.data.clone()
    new_model.load_state_dict(new_state, strict=False)


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    meta_path = REAL_DIR / "metadata.json"
    if not meta_path.exists():
        print(f"Not found: {meta_path}")
        return

    with open(meta_path) as f:
        samples = json.load(f)
    print(f"Real boards: {len(samples)}")

    # Build datasets (all real, no synthetic)
    piece_ds = RealBoardDataset(samples)
    bin_ds = RealBinaryDataset(samples)
    print(f"Piece cells: {len(piece_ds)}, Binary cells: {len(bin_ds)}")

    if len(piece_ds) < 10:
        print("Too few piece samples")
        return

    # Use all data for training (small dataset)
    piece_loader = DataLoader(piece_ds, batch_size=64, shuffle=True)
    bin_loader = DataLoader(bin_ds, batch_size=64, shuffle=True)

    # --- Fine-tune piece model ---
    print("\n--- Fine-tuning piece model ---")
    piece_model = load_piece_model().to(DEVICE)

    # Try loading pretrained weights from disk
    pretrained_path = MODELS_DIR / "chess_mobilenet.pt"
    if pretrained_path.exists():
        try:
            scripted = torch.jit.load(str(pretrained_path), map_location=DEVICE)
            copy_weights(scripted, piece_model)
            print(f"Loaded pretrained: {pretrained_path}")
        except Exception as e:
            print(f"Could not load pretrained: {e}")

    # Class weights for balance
    labels = [lbl for _, lbl in piece_ds.data]
    counts = Counter(labels)
    class_weight = torch.FloatTensor([
        1.0 / np.sqrt(max(counts.get(CLASS_TO_IDX[c], 1), 1))
        for c in CLASSES
    ])
    class_weight = class_weight / class_weight.sum() * len(CLASSES)
    criterion = nn.CrossEntropyLoss(weight=class_weight.to(DEVICE))
    optimizer = optim.Adam(piece_model.parameters(), lr=5e-5)

    best_acc = 0.0
    for epoch in range(1, 101):
        piece_model.train()
        total, correct = 0, 0
        for images, labels in piece_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = piece_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total += labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
        acc = correct / total * 100

        if acc > best_acc:
            best_acc = acc
        if epoch % 5 == 0 or acc == best_acc:
            print(f"  Epoch {epoch:2d}: {acc:.1f}%  best={best_acc:.1f}%")
        if best_acc >= 99.0 and acc >= 98.0:
            break

    # Save fine-tuned piece model
    piece_model.eval()
    example = torch.randn(1, 3, 64, 64)
    scripted = torch.jit.trace(piece_model.cpu(), example)
    scripted.save(str(MODELS_DIR / "chess_mobilenet.pt"))
    with open(MODELS_DIR / "classes.json", "w") as f:
        json.dump(CLASSES, f)
    print(f"\nPiece model saved: {MODELS_DIR / 'chess_mobilenet.pt'}")

    # --- Fine-tune binary model ---
    print("\n--- Fine-tuning binary model ---")
    bin_model = load_binary_model().to(DEVICE)

    bin_pretrained = MODELS_DIR / "chess_binary.pt"
    if bin_pretrained.exists():
        try:
            bin_scripted = torch.jit.load(str(bin_pretrained), map_location=DEVICE)
            copy_weights(bin_scripted, bin_model)
            print(f"Loaded pretrained: {bin_pretrained}")
        except Exception as e:
            print(f"Could not load: {e}")

    empty_cnt = sum(1 for _, l in bin_ds.data if l == 0)
    piece_cnt = sum(1 for _, l in bin_ds.data if l == 1)
    print(f"Binary: empty={empty_cnt}, piece={piece_cnt}")

    empty_w = 1.0 / np.sqrt(max(empty_cnt, 1))
    piece_w = 1.0 / np.sqrt(max(piece_cnt, 1))
    bin_weight = torch.FloatTensor([empty_w, piece_w])
    bin_weight = bin_weight / bin_weight.sum() * 2
    bin_criterion = nn.CrossEntropyLoss(weight=bin_weight.to(DEVICE))
    bin_optimizer = optim.Adam(bin_model.parameters(), lr=5e-5)

    best_bin_acc = 0.0
    for epoch in range(1, 101):
        bin_model.train()
        total, correct = 0, 0
        for images, labels in bin_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            bin_optimizer.zero_grad()
            outputs = bin_model(images)
            loss = bin_criterion(outputs, labels)
            loss.backward()
            bin_optimizer.step()
            total += labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
        acc = correct / total * 100

        if acc > best_bin_acc:
            best_bin_acc = acc
        if epoch % 5 == 0 or acc == best_bin_acc:
            print(f"  Binary epoch {epoch:2d}: {acc:.1f}%  best={best_bin_acc:.1f}%")
        if best_bin_acc >= 99.0 and acc >= 98.0:
            break

    bin_model.eval()
    bin_scripted = torch.jit.trace(bin_model.cpu(), example)
    bin_scripted.save(str(MODELS_DIR / "chess_binary.pt"))
    print(f"Binary model saved: {MODELS_DIR / 'chess_binary.pt'}")
    print(f"\nDone! Best piece: {best_acc:.1f}%, Best binary: {best_bin_acc:.1f}%")


if __name__ == "__main__":
    main()
