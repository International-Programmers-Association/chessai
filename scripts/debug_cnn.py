"""Debug CNN predictions on a board image."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
from chessai.classifier import CNNClassifier

c = CNNClassifier()
c._load()

img = cv2.imread(sys.argv[1])
board = cv2.resize(img, (480, 480))

cells = np.zeros((64, 3, 64, 64), dtype=np.float32)
for i in range(64):
    row, col = divmod(i, 8)
    cell = board[row*60:(row+1)*60, col*60:(col+1)*60]
    cell = cv2.resize(cell, (64, 64))
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    cells[i] = np.stack([gray, gray, gray], axis=0)

batch = torch.from_numpy(cells).to(c._device)
with torch.no_grad():
    bin_logits = c._binary_model(batch)
    bin_probs = torch.softmax(bin_logits, dim=1)
    logits = c._model(batch)
    probs = torch.softmax(logits, dim=1)
    best_idx = logits.argmax(dim=1)
    best_prob = probs.max(dim=1).values

for row in range(8):
    line = ""
    for col in range(8):
        i = row * 8 + col
        sq = f"{chr(97+col)}{8-row}"
        binp = bin_probs[i, 1].item()
        if binp >= 0.5:
            p = best_idx[i].item()
            bp = best_prob[i].item()
            sym = c._classes[p]
            line += f"{sq}:{sym}({bp:.2f}) "
        else:
            line += f"{sq}:empty "
    print(line)
