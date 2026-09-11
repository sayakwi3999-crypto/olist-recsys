# -*- coding: utf-8 -*-
"""Global configuration: database connection, paths, experiment parameters.

The database password is never hard-coded: it is read from environment variables
first, then from a .env file next to this file (.env is git-ignored; see .env.example).
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "output"


def _load_dotenv(path: Path) -> None:
    """Minimal .env parser: read KEY=VALUE line by line; existing env vars win."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(BASE_DIR / ".env")

# Local MySQL (olist_clean holds the cleaned tables)
DB_CONFIG = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "3306")),
    user=os.getenv("DB_USER", "root"),
    password=os.getenv("DB_PASSWORD", ""),
    database=os.getenv("DB_NAME", "olist_clean"),
    charset="utf8mb4",
)

# Recommendation experiment settings
TOP_ITEMS_POOL = 5000      # candidate pool: the N most popular items in the training set
TEST_RATIO = 0.20          # time-based split: the last 20% of interactions form the test set
K_VALUES = [5, 10, 20]     # Top-K values used for evaluation
RANDOM_SEED = 42

# Hybrid weights (user-normalised scores are fused by weighted sum)
HYBRID_WEIGHTS = {
    "itemcf": 0.40,
    "content": 0.25,
    "als": 0.15,
    "svd": 0.10,
    "popular": 0.10,
}

# Matrix factorisation hyperparameters
SVD_K = 20
SVD_EPOCHS = 12
SVD_LR = 0.01
SVD_REG = 0.05

ALS_K = 20
ALS_ITERS = 10
ALS_REG = 10.0

ITEMCF_TOP_K = 50          # number of nearest neighbours kept per item in ItemCF
