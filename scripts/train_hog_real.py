"""Train HOG + MLP classifier on real boards only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chessai.hog_classifier import train_hog_models_real

REAL_DIR = Path(__file__).resolve().parent.parent / "datasetgw"

train_hog_models_real(
    real_meta_path=str(REAL_DIR / "metadata.json"),
)
