from pathlib import Path

import pandas as pd


LABEL_ORDER = ["low", "moderate", "high"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABEL_ORDER)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

METADATA_COLUMNS = [
    "video_id",
    "sport",
    "impact_type",
    "body_region",
    "player_down",
    "risk_level",
]

VIDEO_EXTENSIONS = [".mov", ".mp4", ".avi", ".mkv", ".m4v"]


def resolve_video_path(video_dir: Path, video_id: str) -> Path:
    """Resolve a video file by ID across supported video extensions."""
    for suffix in VIDEO_EXTENSIONS:
        candidate = video_dir / f"{video_id}{suffix}"
        if candidate.exists():
            return candidate

    lower_video_id = video_id.lower()
    for suffix in VIDEO_EXTENSIONS:
        candidate = video_dir / f"{lower_video_id}{suffix}"
        if candidate.exists():
            return candidate

    return video_dir / f"{video_id}{VIDEO_EXTENSIONS[0]}"


def load_metadata(
    metadata_path: str | Path = "data/metadata.csv",
    video_dir: str | Path = "data/cv_module_videos",
) -> pd.DataFrame:
    """Load metadata, drop empty CSV columns, validate labels/files, and add paths."""
    metadata_path = Path(metadata_path)
    video_dir = Path(video_dir)
    df = pd.read_csv(metadata_path)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    df = df.dropna(axis=1, how="all")
    df = df[[c for c in METADATA_COLUMNS if c in df.columns]].copy()

    missing_cols = sorted(set(METADATA_COLUMNS) - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing metadata columns: {missing_cols}")

    for col in METADATA_COLUMNS:
        df[col] = df[col].astype(str).str.strip()

    invalid_labels = sorted(set(df["risk_level"]) - set(LABEL_ORDER))
    if invalid_labels:
        raise ValueError(f"Unexpected risk labels: {invalid_labels}")

    df["video_path"] = df["video_id"].map(lambda vid: str(resolve_video_path(video_dir, vid)))
    missing_files = df.loc[~df["video_path"].map(lambda p: Path(p).exists()), "video_path"]
    if not missing_files.empty:
        sample = missing_files.head(10).tolist()
        raise FileNotFoundError(f"Missing {len(missing_files)} video files. Sample: {sample}")

    df["label_id"] = df["risk_level"].map(LABEL_TO_ID)
    return df


def save_label_mapping(path: str | Path) -> None:
    """Persist the shared label mapping as JSON."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"label_order": LABEL_ORDER, "label_to_id": LABEL_TO_ID}, f, indent=2)
