"""Configuration loader and environment settings for Customer Sentiment Analysis."""

import os
from pathlib import Path
from typing import Any

import yaml

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"


def load_raw_config() -> dict[str, Any]:
    """Load configuration from config.yaml if present, or return defaults."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}
    return cfg


_cfg = load_raw_config()

# Global settings with environment variable fallbacks
SEED = int(os.getenv("SEED", _cfg.get("seed", 42)))

# Data paths
DATA_CFG = _cfg.get("data", {})
RAW_DATA_PATH = Path(
    os.getenv("RAW_DATA_PATH", BASE_DIR / DATA_CFG.get("raw_path", "data/raw/reviews.csv"))
)
PROCESSED_DIR = Path(
    os.getenv("PROCESSED_DIR", BASE_DIR / DATA_CFG.get("processed_dir", "data/processed"))
)
SPLIT_IDS_PATH = Path(
    os.getenv(
        "SPLIT_IDS_PATH",
        BASE_DIR / DATA_CFG.get("split_ids_path", "data/processed/split_ids.json"),
    )
)
TRAIN_RATIO = float(DATA_CFG.get("train_ratio", 0.70))
VAL_RATIO = float(DATA_CFG.get("val_ratio", 0.15))
TEST_RATIO = float(DATA_CFG.get("test_ratio", 0.15))

# Database settings
DB_CFG = _cfg.get("database", {})
DB_PATH = Path(os.getenv("DB_PATH", BASE_DIR / DB_CFG.get("db_path", "data/sentiment.db")))
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH.as_posix()}")

# Model settings
MODEL_CFG = _cfg.get("model", {})
DEFAULT_MODEL_VERSION = os.getenv("MODEL_VERSION", MODEL_CFG.get("version", "v1.0.0"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / MODEL_CFG.get("models_dir", "models")))

# Feature extraction settings
FEATURES_CFG = _cfg.get("features", {})
NGRAM_RANGE = tuple(FEATURES_CFG.get("ngram_range", [1, 2]))
MIN_DF = int(FEATURES_CFG.get("min_df", 3))
MAX_DF = float(FEATURES_CFG.get("max_df", 0.9))
SUBLINEAR_TF = bool(FEATURES_CFG.get("sublinear_tf", True))
MAX_FEATURES = int(FEATURES_CFG.get("max_features", 50000))

# API settings
API_CFG = _cfg.get("api", {})
API_HOST = os.getenv("API_HOST", API_CFG.get("host", "127.0.0.1"))
API_PORT = int(os.getenv("API_PORT", API_CFG.get("port", 5000)))
MAX_BATCH_SIZE = int(API_CFG.get("max_batch_size", 100))
MAX_CHAR_LENGTH = int(API_CFG.get("max_char_length", 5000))
REQUEST_SIZE_LIMIT = int(API_CFG.get("request_size_limit_bytes", 1048576))

# Reports settings
REPORTS_CFG = _cfg.get("reports", {})
FIGURES_DIR = Path(BASE_DIR / REPORTS_CFG.get("figures_dir", "reports/figures"))
EXPORTS_DIR = Path(BASE_DIR / REPORTS_CFG.get("exports_dir", "reports/exports"))
EVALUATION_PATH = Path(BASE_DIR / REPORTS_CFG.get("evaluation_path", "reports/evaluation.md"))
DATA_QUALITY_PATH = Path(BASE_DIR / REPORTS_CFG.get("data_quality_path", "reports/data_quality.md"))
ERRORS_PATH = Path(BASE_DIR / REPORTS_CFG.get("errors_path", "reports/errors.csv"))
THEMES_PATH = Path(BASE_DIR / REPORTS_CFG.get("themes_path", "reports/themes.md"))

# Labels mapping
LABEL_MAP = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}

LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}

ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
CLASSES = ["negative", "neutral", "positive"]
