"""Train U-Net on 3 site themes: chess.com, lichess, worldchess (2000 boards each)."""
import json
import sys
import random
from pathlib import Path

# CUDA torch path
_torch_dir = Path(__file__).resolve().parent / "pkg_torch_cuda"
if str(_torch_dir) not in sys.path:
    sys.path.insert(0, str(_torch_dir))

import chess
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODELS_DIR = Path(__file__).resolve().parent / "models"
DATA_DIR = Path(__file__).resolve().parent / "data"

CLASSES = ["K", "Q", "R", "B", "N", "P", "k", "q", "r", "b", "n", "p"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)
NUM_OUT = 13

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

SITES = [
    {
        "name": "chesscom",
        "light": (208, 236, 235),
        "dark": (82, 149, 115),
        "pieces_dir": DATA_DIR / "chesscom",
    },
    {
        "name": "lichess",
        "light": (181, 217, 240),
        "dark": (99, 136, 181),
        "pieces_dir": DATA_DIR / "lichess",
    },
    {
        "name": "worldchess",
        "light": (212, 229, 233),
        "dark": (79, 153, 106),
        "pieces_dir": DATA_DIR / "worldchess",
    },
]

PIECE_FILES = {
    "K": "wk.png", "Q": "wq.png", "R": "wr.png",
    "B": "wb.png", "N": "wn.png", "P": "wp.png",
    "k": "bk.png", "q": "bq.png", "r": "br.png",
    "b": "bb.png", "n": "bn.png", "p": "bp.png",
}


def load_site_pieces(pieces_dir: Path) -> dict[str, np.ndarray]:
    pieces: dict[str, np.ndarray] = {}
    for sym, fn in PIECE_FILES.items():
        path = pieces_dir / fn
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is not None:
            pieces[sym] = img
    return pieces


def composite_piece(piece_img: np.ndarray, bg: np.ndarray,
                    x: int, y: int, size: int) -> np.ndarray:
    h, w = piece_img.shape[:2]
    if size % 2:
        size += 1
    resized = cv2.resize(piece_img, (size, size), interpolation=cv2.INTER_AREA)
    ox = x + (60 - size) // 2
    oy = y + (60 - size) // 2
    out = bg.copy()
    if resized.shape[2] == 4:
        alpha = resized[:, :, 3].astype(np.float32) / 255.0
        bgr = resized[:, :, :3].astype(np.float32)
        y1, y2 = max(0, oy), min(60, oy + size)
        x1, x2 = max(0, ox), min(60, ox + size)
        ry1, ry2 = max(0, -oy), min(size, 60 - oy)
        rx1, rx2 = max(0, -ox), min(size, 60 - ox)
        region = out[y1:y2, x1:x2].astype(np.float32)
        blended = bgr[ry1:ry2, rx1:rx2] * alpha[ry1:ry2, rx1:rx2, None] + \
                  region * (1.0 - alpha[ry1:ry2, rx1:rx2, None])
        out[y1:y2, x1:x2] = blended.clip(0, 255).astype(np.uint8)
    return out


def _random_board_fen(rng: random.Random) -> str:
    board = chess.Board()
    for _ in range(rng.randint(10, 60)):
        legal = list(board.legal_moves)
        if not legal:
            break
        board.push(rng.choice(legal))
    return board.fen()


def _render_board(board_fen: str, light: np.ndarray, dark: np.ndarray,
                  pieces: dict[str, np.ndarray],
                  rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    board = chess.Board(board_fen)
    img = np.zeros((480, 480, 3), dtype=np.uint8)
    labels = np.zeros((8, 8), dtype=np.int64)

    for row in range(8):
        for col in range(8):
            y, x = row * 60, col * 60
            is_dark = (row + col) % 2 == 0
            img[y:y+60, x:x+60] = dark if is_dark else light

            sq = chess.square(col, 7 - row)
            piece = board.piece_at(sq)
            if piece is None:
                labels[row, col] = 12
                continue

            sym = piece.symbol()
            tmpl = pieces.get(sym)
            if tmpl is None:
                labels[row, col] = 12
                continue

            psize = rng.randint(30, 56)
            ox = rng.randint(-3, 3)
            oy = rng.randint(-3, 3)
            cell = composite_piece(tmpl, img[y:y+60, x:x+60].copy(), ox, oy, psize)
            img[y:y+60, x:x+60] = cell
            labels[row, col] = CLASS_TO_IDX.get(sym, 12)

    return img, labels


class _SiteConfig:
    __slots__ = ("name", "light", "dark", "pieces", "fens")

    def __init__(self, name: str, light, dark, pieces: dict, fens: list[str]):
        self.name = name
        self.light = light
        self.dark = dark
        self.pieces = pieces
        self.fens = fens


class MultiSiteBoardDataset(Dataset):
    """Generate boards for all 3 sites on the fly (low memory)."""

    def __init__(self, boards_per_site: int = 2000, seed: int = 42):
        rng = random.Random(seed)
        self.sites: list[_SiteConfig] = []
        self._indices: list[tuple[int, int]] = []  # (site_idx, fen_idx)

        for si, site in enumerate(SITES):
            pieces = load_site_pieces(site["pieces_dir"])
            if len(pieces) < 12:
                print(f"  WARNING: {site['name']} has only {len(pieces)} pieces, skipping")
                continue

            print(f"  Pre-generating FENs for {site['name']}...")
            fens = [_random_board_fen(rng) for _ in range(boards_per_site)]

            self.sites.append(_SiteConfig(
                name=site["name"],
                light=np.array(site["light"], dtype=np.uint8),
                dark=np.array(site["dark"], dtype=np.uint8),
                pieces=pieces,
                fens=fens,
            ))
            for fi in range(len(fens)):
                self._indices.append((si, fi))

        rng.shuffle(self._indices)
        total = len(self._indices)
        print(f"Total boards: {total} ({total // boards_per_site} sites) "
              f"— rendered on demand, ~{total // boards_per_site * boards_per_site // 3} MB RAM")

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        si, fi = self._indices[idx]
        site = self.sites[si]
        fen = site.fens[fi]

        rng = random.Random(hash((si, fi)) & 0xFFFFFFFF)
        img, labels = _render_board(fen, site.light, site.dark, site.pieces, rng)

        if np.random.rand() < 0.5:
            delta = np.random.uniform(-0.12, 0.12) * 255
            img = np.clip(img.astype(np.float32) + delta, 0, 255).astype(np.uint8)

        if np.random.rand() < 0.2:
            noise = np.random.normal(0, 5, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        return torch.from_numpy(gray).unsqueeze(0), torch.from_numpy(labels)


class UNetBoard(nn.Module):
    """U-Net: 480x480 input -> 8x8x13 output."""

    def __init__(self, num_classes: int = 13):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c), nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c), nn.ReLU(),
            )

        self.enc1 = conv_block(1, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(128, 256)
        self.pool4 = nn.MaxPool2d(2)

        self.bridge = conv_block(256, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = conv_block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = conv_block(64, 32)

        self.out = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bridge(self.pool4(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    boards_per_site = 2000
    print(f"Generating {boards_per_site} boards per site ({len(SITES)} sites)...")
    ds = MultiSiteBoardDataset(boards_per_site=boards_per_site, seed=42)

    total = len(ds)
    split = int(total * 0.9)
    train_ds, val_ds = torch.utils.data.Subset(ds, range(split)), torch.utils.data.Subset(ds, range(split, total))
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)

    print(f"Train: {len(train_ds)} boards, Val: {len(val_ds)} boards")

    model = UNetBoard(num_classes=13).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_acc = 0.0
    for epoch in range(1, 21):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
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
        if epoch % 3 == 0 or is_best:
            print(f"  Epoch {epoch:2d}  train={train_acc:.1f}%  val={val_acc:.1f}%  best={best_acc:.1f}%")
            sys.stdout.flush()

        if best_acc >= 99.9 and epoch >= 3:
            print("  Early stop: 100% accuracy reached")
            break

    model.eval()
    example = torch.randn(1, 1, 480, 480)
    scripted = torch.jit.trace(model.cpu(), example)
    scripted.save(str(MODELS_DIR / "chess_unet.pt"))
    print(f"Saved: {MODELS_DIR / 'chess_unet.pt'} (best: {best_acc:.1f}%)")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
