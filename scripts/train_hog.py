"""Train HOG + MLP classifier on synthetic + real boards."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chessai.hog_classifier import train_hog_models

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
REAL_DIR = Path(__file__).resolve().parent.parent / "datasetgw"

train_hog_models(
    synthetic_dir=str(DATASET_DIR),
    real_meta_path=str(REAL_DIR / "metadata.json"),
    synthetic_meta_path=str(DATASET_DIR / "metadata.json"),
)
