"""Evaluate piece classification on test_boards."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import chess
import cv2
import numpy as np

test_dir = Path("test_boards")

from chessai.classifier import CNNClassifier, TemplateMatcher

try:
    classifier = CNNClassifier()
    classifier._load()  # verify torch works
except Exception:
    classifier = TemplateMatcher()

print(f"Using: {classifier.__class__.__name__}")

METADATA = test_dir / "metadata.json"
if METADATA.exists():
    with open(METADATA) as f:
        samples = json.load(f)
    total = errors = 0
    for s in samples:
        img_path = test_dir / "synthetic" / s["image"]
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        board_map = classifier.classify_board(img)
        for sq, expected in s["cells"].items():
            predicted = board_map.get(sq, ".")
            if predicted is None:
                predicted = "."
            if predicted != expected:
                errors += 1
                if errors <= 10:
                    print(f"  {s['image']} {sq}: expected {expected}, got {predicted}")
            total += 1
    if total > 0:
        acc = (total - errors) / total * 100
        print(f"Synthetic boards: {acc:.1f}% ({total-errors}/{total})")
    else:
        print("Synthetic boards: no images found")

print("\nReal boards:")
for p in sorted((test_dir / "real").glob("*.png")):
    img = cv2.imread(str(p))
    if img is None:
        continue
    board_map = classifier.classify_board(img)
    pieces = sum(1 for v in board_map.values() if v is not None)
    print(f"  {p.name}: {pieces} pieces")
    for sq in sorted(board_map.keys()):
        v = board_map.get(sq)
        if v:
            print(f"    {sq}: {v}")
    if pieces > 0:
        from chessai.local_detector import board_map_to_fen
        fen = board_map_to_fen(board_map)
        print(f"    FEN: {fen}")
