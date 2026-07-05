"""Train end-to-end board CNN with variable piece scale + U-Net-like architecture."""
import json
import sys
import random
from pathlib import Path

import chess
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "piece_templates"
MODELS_DIR = Path(__file__).resolve().parent.parent / "chessai" / "models"

CLASSES = ["K", "Q", "R", "B", "N", "P", "k", "q", "r", "b", "n", "p"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)  # 12 pieces + empty = 13
NUM_OUT = 13  # 12 pieces + 1 empty (index 12)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

LIGHT_BGR = np.array([212, 229, 233], dtype=np.uint8)
DARK_BGR = np.array([79, 153, 106], dtype=np.uint8)


def composite_piece(piece_img: np.ndarray, bg: np.ndarray,
                    x: int, y: int, size: int) -> np.ndarray:
    """Place piece at (x,y) with given size onto 60x60 canvas."""
    h, w = piece_img.shape[:2]
    if size % 2: size += 1
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


class FullBoardDataset(Dataset):
    """Generate full 480x480 boards with random piece sizes."""

    def __init__(self, num_boards: int = 5000):
        sym_to_file = {
            "K": "wk.png", "Q": "wq.png", "R": "wr.png",
            "B": "wb.png", "N": "wn.png", "P": "wp.png",
            "k": "bk.png", "q": "bq.png", "r": "br.png",
            "b": "bb.png", "n": "bn.png", "p": "bp.png",
        }
        self.pieces: dict[str, np.ndarray] = {}
        for sym, fn in sym_to_file.items():
            img = cv2.imread(str(TEMPLATE_DIR / fn), cv2.IMREAD_UNCHANGED)
            if img is not None:
                self.pieces[sym] = img

        self.num_boards = num_boards
        self.rng = random.Random(42)
        # Pre-generate random boards
        self.boards: list[tuple[np.ndarray, np.ndarray]] = []
        files = "abcdefgh"
        for _ in range(num_boards):
            board = chess.Board()
            num_moves = self.rng.randint(10, 60)
            for _ in range(num_moves):
                legal = list(board.legal_moves)
                if not legal:
                    break
                board.push(self.rng.choice(legal))

            img = np.zeros((480, 480, 3), dtype=np.uint8)
            labels = np.zeros((8, 8), dtype=np.int64)  # 0-11 = piece, 12 = empty

            for row in range(8):
                for col in range(8):
                    y, x = row * 60, col * 60
                    is_dark = (row + col) % 2 == 0
                    bg_bgr = DARK_BGR if is_dark else LIGHT_BGR
                    img[y:y+60, x:x+60] = bg_bgr

                    sq = chess.square(col, 7 - row)
                    piece = board.piece_at(sq)
                    if piece is None:
                        labels[row, col] = 12  # empty
                        continue

                    sym = piece.symbol()
                    tmpl = self.pieces.get(sym)
                    if tmpl is None:
                        labels[row, col] = 12
                        continue

                    # Random piece size: 30-56px (60px cell)
                    psize = self.rng.randint(30, 56)
                    ox = self.rng.randint(-3, 3)
                    oy = self.rng.randint(-3, 3)
                    cell = composite_piece(tmpl, img[y:y+60, x:x+60].copy(), ox, oy, psize)
                    img[y:y+60, x:x+60] = cell
                    labels[row, col] = CLASS_TO_IDX.get(sym, 12)

            self.boards.append((img, labels))

    def __len__(self):
        return self.num_boards

    def __getitem__(self, idx):
        img, labels = self.boards[idx]

        # Brightness augmentation
        if np.random.rand() < 0.5:
            delta = np.random.uniform(-0.2, 0.2) * 255
            img = np.clip(img.astype(np.float32) + delta, 0, 255).astype(np.uint8)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        tensor = torch.from_numpy(gray).unsqueeze(0)  # (1, 480, 480)
        return tensor, torch.from_numpy(labels)


class UNetBoard(nn.Module):
    """U-Net style: 480x480 input → 8x8x13 output (board cell classification)."""

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
        self.pool1 = nn.MaxPool2d(2)  # 240
        self.enc2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)  # 120
        self.enc3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)  # 60
        self.enc4 = conv_block(128, 256)
        self.pool4 = nn.MaxPool2d(2)  # 30

        self.bridge = conv_block(256, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = conv_block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = conv_block(64, 32)

        # Output: 480 → avg pool to 8×8
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

        return self.out(d1)  # (B, 13, 8, 8)


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating board dataset with random piece scales...")
    NUM = 3000
    ds = FullBoardDataset(num_boards=NUM)
    split = int(NUM * 0.9)
    train_ds, val_ds = torch.utils.data.Subset(ds, range(split)), torch.utils.data.Subset(ds, range(split, NUM))
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
            outputs = model(images)  # (B, 13, 8, 8)
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

        if val_acc > best_acc:
            best_acc = val_acc
        if epoch % 3 == 0 or val_acc == best_acc:
            print(f"  Epoch {epoch:2d}  train={train_acc:.1f}%  val={val_acc:.1f}%  best={best_acc:.1f}%")

    # Save as TorchScript
    model.eval()
    example = torch.randn(1, 1, 480, 480)
    scripted = torch.jit.trace(model.cpu(), example)
    scripted.save(str(MODELS_DIR / "chess_unet.pt"))
    print(f"Saved: {MODELS_DIR / 'chess_unet.pt'} (best: {best_acc:.1f}%)")


class UNetClassifier:
    """Wrapper for U-Net inference on board images."""

    def __init__(self, model_path=None):
        self.model_path = model_path or str(MODELS_DIR / "chess_unet.pt")
        self._model = None
        self._device = None

    def _load(self):
        if self._model is not None:
            return
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = torch.jit.load(self.model_path, map_location=self._device)
        self._model.eval()

    def classify_board(self, board_bgr: np.ndarray) -> dict:
        self._load()
        h, w = board_bgr.shape[:2]
        size = min(h, w)
        x, y = (w - size) // 2, (h - size) // 2
        cropped = board_bgr[y:y+size, x:x+size]
        board = cv2.resize(cropped, (480, 480))
        gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(self._device)

        from chessai.local_detector import FILES_NORMAL
        with torch.no_grad():
            output = self._model(tensor)  # (1, 13, 8, 8)
            probs = torch.softmax(output, dim=1).squeeze(0)  # (13, 8, 8)

        raw = {}
        for row in range(8):
            for col in range(8):
                sq = f"{FILES_NORMAL[col]}{8 - row}"
                cls = probs[:, row, col].argmax().item()
                if cls == 12:  # empty
                    raw[sq] = None
                else:
                    raw[sq] = IDX_TO_CLASS[cls]
        from chessai.local_detector import sanitize_board_map
        clean, _ = sanitize_board_map(raw)
        return clean


def finetune_real():
    """Fine-tune pre-trained UNet on 48 real boards from datasetgw."""
    REAL_DIR = Path(__file__).resolve().parent.parent / "datasetgw"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(REAL_DIR / "metadata.json") as f:
        samples = json.load(f)

    real_data: list[tuple[np.ndarray, np.ndarray]] = []
    files = "abcdefgh"
    for s in samples:
        img_path = REAL_DIR / s["image"]
        board_bgr = cv2.imread(str(img_path))
        if board_bgr is None:
            continue
        board = cv2.resize(board_bgr, (480, 480))
        gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        labels = np.zeros((8, 8), dtype=np.int64)
        for row in range(8):
            for col in range(8):
                sq = f"{files[col]}{8 - row}"
                piece = s["cells"].get(sq, ".")
                labels[row, col] = CLASS_TO_IDX.get(piece, 12) if piece != "." else 12
        real_data.append((gray, labels))

    print(f"Real boards loaded: {len(real_data)}")

    if len(real_data) < 10:
        print("Too few real boards")
        return

    model_path = MODELS_DIR / "chess_unet.pt"
    if not model_path.exists():
        print(f"Pre-trained model not found: {model_path}")
        return

    model = torch.jit.load(str(model_path), map_location=DEVICE)
    # Convert ScriptModule back to nn.Module for training
    # We need the original architecture
    model = UNetBoard(num_classes=13).to(DEVICE)
    # Try loading saved state dict
    try:
        saved = torch.jit.load(str(model_path), map_location=DEVICE)
        model.load_state_dict(saved.state_dict(), strict=False)
    except Exception as e:
        print(f"Could not load weights: {e}")

    rng = np.random.default_rng(42)
    indices = list(range(len(real_data)))
    rng.shuffle(indices)
    split = max(1, int(len(indices) * 0.8))
    train_idx, val_idx = indices[:split], indices[split:]

    train_imgs = torch.stack([torch.from_numpy(real_data[i][0]).unsqueeze(0) for i in train_idx])
    train_labels = torch.stack([torch.from_numpy(real_data[i][1]) for i in train_idx])
    val_imgs = torch.stack([torch.from_numpy(real_data[i][0]).unsqueeze(0) for i in val_idx])
    val_labels = torch.stack([torch.from_numpy(real_data[i][1]) for i in val_idx])

    train_ds = torch.utils.data.TensorDataset(train_imgs, train_labels)
    val_ds = torch.utils.data.TensorDataset(val_imgs, val_labels)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)

    print(f"Train: {len(train_ds)} boards, Val: {len(val_ds)} boards")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_acc = 0.0
    for epoch in range(1, 101):
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

        if val_acc > best_acc:
            best_acc = val_acc
            # Save best
            model.eval()
            example = torch.randn(1, 1, 480, 480)
            scripted = torch.jit.trace(model.cpu(), example)
            scripted.save(str(MODELS_DIR / "chess_unet.pt"))
            model.to(DEVICE)

        if epoch % 5 == 0 or val_acc == best_acc:
            print(f"  Epoch {epoch:2d}  train={train_acc:.1f}%  val={val_acc:.1f}%  best={best_acc:.1f}%")
        if best_acc >= 95.0:
            break

    print(f"Fine-tuned saved: {MODELS_DIR / 'chess_unet.pt'} (best: {best_acc:.1f}%)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "finetune":
        finetune_real()
    else:
        main()
