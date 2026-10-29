"""Stratified dataset splitting and persistence."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import SEED, SPLIT_IDS_PATH, TEST_RATIO, TRAIN_RATIO, VAL_RATIO


def make_stratified_split(
    df: pd.DataFrame,
    label_col: str = "true_label",
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_state: int = SEED,
    save_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    """Perform stratified 70/15/15 split on the dataset and save split indices.

    Returns:
        (df_with_split_column, split_indices_dict)
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    assert np.isclose(total_ratio, 1.0), f"Ratios must sum to 1.0, got {total_ratio}"

    df = df.copy()
    indices = df.index.to_numpy()
    labels = df[label_col].to_numpy()

    # First split off the test set (15%)
    train_val_idx, test_idx, train_val_labels, _ = train_test_split(
        indices,
        labels,
        test_size=test_ratio,
        stratify=labels,
        random_state=random_state,
    )

    # Next split train and val from train_val
    # relative val ratio = val_ratio / (train_ratio + val_ratio) = 0.15 / 0.85
    rel_val_ratio = val_ratio / (train_ratio + val_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=rel_val_ratio,
        stratify=train_val_labels,
        random_state=random_state,
    )

    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"

    split_ids: dict[str, list[int]] = {
        "train": [int(i) for i in train_idx],
        "val": [int(i) for i in val_idx],
        "test": [int(i) for i in test_idx],
    }

    out_path = Path(save_path) if save_path else SPLIT_IDS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(split_ids, f, indent=2)

    return df, split_ids


def load_split_ids(filepath: str | Path | None = None) -> dict[str, list[int]]:
    """Load precomputed split indices from JSON."""
    path = Path(filepath) if filepath else SPLIT_IDS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Split IDs file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
