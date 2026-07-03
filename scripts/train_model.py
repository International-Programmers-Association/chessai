"""Train CNN on synthetic chess board data.

Uses generated boards from scripts/generate_dataset.py (or the real
dataset).  The dataset is split 80/20 by board.  Two models are trained:
  - Piece model (12-class, MobileNetV3-Small) — classifies piece type
  - Binary model (empty vs piece) — filters empty cells

Run:  python scripts/generate_dataset.py   # first time
      python scripts/train_model.py
"""

import json
import sys
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
MODELS_DIR = Path(__file__).resolve().parent.parent / "chessai" / "models"

# 12 piece types (binary model handles empty cells separately)
CLASSES = ["K", "Q", "R", "B", "N", "P", "k", "q", "r", "b", "n", "p"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


class ChessBoardDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.data: list[tuple[str, int]] = []
        files = "abcdefgh"

        for sample in samples:
            img_path = DATASET_DIR / sample["image"]
            if not img_path.exists():
                continue
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


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = DATASET_DIR / "metadata.json"

    if not meta_path.exists():
        print(f"Dataset not found: {meta_path}")
        print("Run 'python scripts/collect_dataset.py' first!")
        return

    with open(meta_path) as f:
        samples = json.load(f)

    if len(samples) < 2:
        print(f"Only {len(samples)} board(s). Need at least 2.")
        return

    # Stratified split by board: ensure validation covers diverse boards
    rng = np.random.default_rng(42)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    # Sort by piece count so rich boards go to train, but we still keep some variety
    split = max(1, int(len(indices) * 0.8))
    # Stratified split by board: 80/20
    rng = np.random.default_rng(42)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    split = max(1, int(len(indices) * 0.8))
    train_ds = ChessBoardDataset([samples[i] for i in indices[:split]])
    val_ds = ChessBoardDataset([samples[i] for i in indices[split:]])

    # Soft class weights: sqrt of inverse frequency
    labels = [lbl for _, lbl in train_ds.data]
    class_counts = Counter(labels)
    weight_per_class = {c: 1.0 / np.sqrt(max(cnt, 1)) for c, cnt in class_counts.items()}
    sample_weights = [weight_per_class[lbl] for lbl in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=128, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=128)

    print(f"Train: {len(train_ds)} cells, Val: {len(val_ds)} cells")
    counts = {}
    for _, lbl in train_ds.data:
        c = IDX_TO_CLASS[lbl]
        counts[c] = counts.get(c, 0) + 1
    for c in CLASSES:
        print(f"  {c}: {counts.get(c, 0)} train")

    # MobileNetV3-Small pretrained on ImageNet (12 classes, no empty)
    print("\nLoading MobileNetV3-Small (pretrained)...")
    model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
    # Replace classifier head: 576 → 128 → 12
    in_features = model.classifier[0].in_features  # 576
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 12),
    )
    model = model.to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters())
    print(f"Total params: {trainable:,}")

    # Balanced weights: sqrt inverse
    class_weight = torch.FloatTensor([
        1.0 / np.sqrt(max(class_counts.get(CLASS_TO_IDX[c], 1), 1))
        for c in CLASSES
    ])
    class_weight = class_weight / class_weight.sum() * len(CLASSES)
    criterion = nn.CrossEntropyLoss(weight=class_weight.to(DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    best_acc = 0.0
    best_epoch = 0
    for epoch in range(1, 31):
        model.train()
        train_loss, train_acc = 0.0, 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += (outputs.argmax(1) == labels).float().mean().item()
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                val_acc += (outputs.argmax(1) == labels).float().mean().item()
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
        if epoch % 3 == 0 or val_acc == best_acc:
            print(f"Epoch {epoch:2d}  train={train_loss:.4f}/{train_acc:.3f}  val={val_loss:.4f}/{val_acc:.3f}  best={best_acc:.3f}")
        if epoch - best_epoch > 5:
            print(f"Early stopping at epoch {epoch}")
            break

    # Save as TorchScript for fast inference
    model.eval()
    example = torch.randn(1, 3, 64, 64)
    scripted = torch.jit.trace(model.cpu(), example)
    model_path = MODELS_DIR / "chess_mobilenet.pt"
    scripted.save(str(model_path))
    with open(MODELS_DIR / "classes.json", "w") as f:
        json.dump(CLASSES, f)

    print(f"\nBest val accuracy: {best_acc:.1%}")
    print(f"Model: {model_path}")

    # Per-class accuracy on val
    if len(val_ds) > 0:
        model = model.to(DEVICE)
        model.eval()
        correct, total = {}, {}
        for c in CLASSES:
            correct[c] = total[c] = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                preds = model(images).argmax(1)
                for i in range(len(labels)):
                    true = IDX_TO_CLASS[int(labels[i])]
                    pred = IDX_TO_CLASS[int(preds[i])]
                    total[true] += 1
                    if true == pred:
                        correct[true] += 1
        print("\nPer-class validation:")
        for c in CLASSES:
            t = total.get(c, 0)
            if t > 0:
                print(f"  {c}: {correct[c]}/{t} = {correct[c]/t*100:.1f}%")
            else:
                print(f"  {c}: no data")

    # --- Binary empty/piece classifier ---
    print("\n--- Training binary empty/piece classifier ---")
    BINARY_CLASSES = [".", "piece"]  # 0=empty, 1=piece
    BINARY_CLS_TO_IDX = {".": 0, "piece": 1}

    def binary_target(piece: str) -> int:
        return 0 if piece == "." else 1

    class BinaryDataset(Dataset):
        def __init__(self, samples: list[dict]):
            self.data: list[tuple[np.ndarray, int]] = []
            files = "abcdefgh"
            for sample in samples:
                img_path = DATASET_DIR / sample["image"]
                if not img_path.exists():
                    continue
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
                        self.data.append((cell, binary_target(piece)))

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            img, label = self.data[idx]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            tensor = torch.from_numpy(np.stack([gray, gray, gray], axis=0))
            return tensor, torch.tensor(label, dtype=torch.long)

    train_bin = BinaryDataset([samples[i] for i in indices[:split]])
    val_bin = BinaryDataset([samples[i] for i in indices[split:]])

    train_bin_loader = DataLoader(train_bin, batch_size=256, shuffle=True)
    val_bin_loader = DataLoader(val_bin, batch_size=256)

    empty_cnt = sum(1 for _, l in train_bin.data if l == 0)
    piece_cnt = sum(1 for _, l in train_bin.data if l == 1)
    print(f"Train binary: empty={empty_cnt}, piece={piece_cnt}")

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

    bin_params = sum(p.numel() for p in bin_model.parameters())
    print(f"Binary model params: {bin_params:,}")

    # Balanced weights: empty ~5x more common, give it less weight
    empty_w = 1.0 / np.sqrt(empty_cnt)
    piece_w = 1.0 / np.sqrt(piece_cnt)
    bin_weight = torch.FloatTensor([empty_w, piece_w])
    bin_weight = bin_weight / bin_weight.sum() * 2
    bin_criterion = nn.CrossEntropyLoss(weight=bin_weight.to(DEVICE))

    bin_optimizer = optim.Adam(bin_model.parameters(), lr=0.0005)
    bin_scheduler = optim.lr_scheduler.CosineAnnealingLR(bin_optimizer, T_max=40)

    best_bin_acc = 0.0
    best_bin_epoch = 0
    for epoch in range(1, 41):
        bin_model.train()
        loss_sum, acc_sum = 0.0, 0.0
        for images, labels in train_bin_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            bin_optimizer.zero_grad()
            outputs = bin_model(images)
            loss = bin_criterion(outputs, labels)
            loss.backward()
            bin_optimizer.step()
            loss_sum += loss.item()
            acc_sum += (outputs.argmax(1) == labels).float().mean().item()
        loss_sum /= len(train_bin_loader)
        acc_sum /= len(train_bin_loader)

        bin_model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for images, labels in val_bin_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = bin_model(images)
                val_loss += bin_criterion(outputs, labels).item()
                val_acc += (outputs.argmax(1) == labels).float().mean().item()
        val_loss /= len(val_bin_loader)
        val_acc /= len(val_bin_loader)
        bin_scheduler.step()

        if val_acc > best_bin_acc:
            best_bin_acc = val_acc
            best_bin_epoch = epoch
        if epoch % 3 == 0 or val_acc == best_bin_acc:
            print(f"  Binary epoch {epoch:2d}  train={loss_sum:.4f}/{acc_sum:.3f}  val={val_loss:.4f}/{val_acc:.3f}  best={best_bin_acc:.3f}")
        if epoch - best_bin_epoch > 5:
            break

    # Save binary model
    bin_model.eval()
    bin_scripted = torch.jit.trace(bin_model.cpu(), example)
    bin_path = MODELS_DIR / "chess_binary.pt"
    bin_scripted.save(str(bin_path))
    print(f"Binary model: {bin_path}")
    print(f"Best binary val accuracy: {best_bin_acc:.1%}")


if __name__ == "__main__":
    main()
