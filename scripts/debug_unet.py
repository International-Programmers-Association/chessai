"""Debug UNet confidence on a board image."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
from chessai.classifier import UNetBoardClassifier

c = UNetBoardClassifier()
c._load()

board_bgr = cv2.imread(sys.argv[1])
h, w = board_bgr.shape[:2]
size = min(h, w)
x, y = (w - size) // 2, (h - size) // 2
cropped = board_bgr[y:y+size, x:x+size]
board = cv2.resize(cropped, (480, 480))
gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(c._device)

with torch.no_grad():
    output = c._model(tensor).squeeze(0)
    probs = torch.softmax(output, dim=0)

for row in range(8):
    line = ""
    for col in range(8):
        sq = f"{chr(97+col)}{8-row}"
        cls = probs[:, row, col].argmax().item()
        prob = probs[:, row, col].max().item()
        sym = "empty" if cls == 12 else c._classes[cls]
        line += f"{sq}:{sym}({prob:.2f}) "
    print(line)
