"""Launch piece template trainer (from chess-vision-ai). Run once per board theme."""

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
TRAINER = ROOT / "chess-vision-ai" / "Identifying_chess_pieces.py"

if not TRAINER.exists():
    print(f"Trainer not found: {TRAINER}")
    sys.exit(1)

sys.path.insert(0, str(ROOT / "chess-vision-ai"))
runpy.run_path(str(TRAINER), run_name="__main__")
