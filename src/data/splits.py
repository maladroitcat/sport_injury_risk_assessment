from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data.metadata import load_metadata


def create_stratified_folds(
    metadata_path: str | Path = "data/metadata.csv",
    video_dir: str | Path = "data/cv_module_videos",
    output_path: str | Path = "artifacts/splits/folds.csv",
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create reusable stratified folds at video level."""
    df = load_metadata(metadata_path, video_dir)
    folds = df[["video_id", "risk_level", "label_id"]].copy()
    folds["fold"] = -1

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold_idx, (_, val_idx) in enumerate(splitter.split(folds["video_id"], folds["label_id"])):
        folds.loc[val_idx, "fold"] = fold_idx

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output_path, index=False)
    return folds


def load_or_create_folds(
    output_path: str | Path = "artifacts/splits/folds.csv",
    **kwargs,
) -> pd.DataFrame:
    output_path = Path(output_path)
    if output_path.exists():
        return pd.read_csv(output_path)
    return create_stratified_folds(output_path=output_path, **kwargs)
