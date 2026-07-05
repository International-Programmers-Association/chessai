"""Train U-Net on one site dataset from D:\chessbot\dataset\<site>."""
import json
import sys
from pathlib import Path

# CUDA torch path — must be before any torch/cv2/numpy import
_torch_dir = Path(__file__).resolve().parent / "pkg_torch_cuda"
if str(_torch_dir) not in sys.path:
    sys.path.insert(0, str(_torch_dir))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

from train_unet import UNetBoard, CLASS_TO_IDX

DATASET_DIR = Path("D:/chessbot/dataset")
MODELS_DIR = Path("D:/chessbot/models")

SITES = ["chesscom", "lichess"]


class DiskBoardDataset(Dataset):
    """Load pre-rendered board images from disk."""

    def __init__(self, site: str, split: str = "train"):
        self.site = site
        site_dir = DATASET_DIR / site
        meta_path = site_dir / "metadata.json"

        with open(meta_path) as f:
            all_samples = json.load(f)

        split_at = int(len(all_samples) * 0.9)
        samples = all_samples[:split_at] if split == "train" else all_samples[split_at:]
        self.samples = samples
        self.site_dir = site_dir

        self.files = "abcdefgh"
        print(f"  {site}/{split}: {len(samples)} boards")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.imread(str(self.site_dir / s["image"]))
        if img is None:
            img = np.zeros((480, 480, 3), dtype=np.uint8)

        if np.random.rand() < 0.5:
            delta = np.random.uniform(-0.12, 0.12) * 255
            img = np.clip(img.astype(np.float32) + delta, 0, 255).astype(np.uint8)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        labels = np.zeros((8, 8), dtype=np.int64)
        for row in range(8):
            for col in range(8):
                sq = f"{self.files[col]}{8 - row}"
                cell = s["cells"].get(sq, ".")
                labels[row, col] = CLASS_TO_IDX.get(cell, 12)

        return torch.from_numpy(gray).unsqueeze(0), torch.from_numpy(labels)


def train_site(site: str):
    print(f"\n{'='*50}")
    print(f"Training U-Net for {site}...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = DiskBoardDataset(site, "train")
    val_ds = DiskBoardDataset(site, "val")
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)

    model = UNetBoard(num_classes=13).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_acc = 0.0
    for epoch in range(1, 21):
        model.train()
        train_correct, train_total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = outputs.max(1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.numel()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = outputs.max(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.numel()

        train_acc = train_correct / train_total * 100
        val_acc = val_correct / val_total * 100
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
        if epoch % 2 == 0 or is_best:
            print(f"    Epoch {epoch:2d}  train={train_acc:.1f}%  val={val_acc:.1f}%  best={best_acc:.1f}%")
            sys.stdout.flush()
        if best_acc >= 99.9 and epoch >= 3:
            print("    Early stop: 100% accuracy")
            break

    model_path = MODELS_DIR / f"chess_unet_{site}.pt"
    model.eval()
    example = torch.randn(1, 1, 480, 480)
    scripted = torch.jit.trace(model.cpu(), example)
    scripted.save(str(model_path))
    print(f"  Saved: {model_path} (best: {best_acc:.1f}%)")
    sys.stdout.flush()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("site", nargs="?", default=None, help="Site to train: chesscom, lichess, or all")
    args = parser.parse_args()

    if args.site and args.site != "all":
        train_site(args.site)
    else:
        for site in SITES:
            train_site(site)
    print(f"\nDone! Models in {MODELS_DIR}/")
