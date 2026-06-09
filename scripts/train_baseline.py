import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.metadata import LABEL_TO_ID, load_metadata
from src.data.splits import load_or_create_folds
from src.evaluation.metrics import compute_metrics, save_confusion_matrix, save_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate naive majority-class baseline.")
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument("--video-dir", default="data/cv_module_videos")
    parser.add_argument("--folds", default="artifacts/splits/folds.csv")
    parser.add_argument("--output-dir", default="artifacts/models/baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_metadata(args.metadata, args.video_dir)
    folds = load_or_create_folds(output_path=args.folds, metadata_path=args.metadata, video_dir=args.video_dir)
    df = df.merge(folds[["video_id", "fold"]], on="video_id", how="left")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_predictions = []
    for fold in sorted(df["fold"].unique()):
        train_df = df[df["fold"] != fold]
        val_df = df[df["fold"] == fold]
        majority_label = train_df["risk_level"].value_counts().idxmax()
        y_pred = np.full(len(val_df), LABEL_TO_ID[majority_label], dtype=int)
        fold_predictions = val_df[["video_id", "risk_level", "label_id", "fold"]].copy()
        fold_predictions["pred_label_id"] = y_pred
        fold_predictions["pred_risk_level"] = majority_label
        all_predictions.append(fold_predictions)

    predictions = pd.concat(all_predictions, ignore_index=True)
    y_true = predictions["label_id"].to_numpy(dtype=int)
    y_pred = predictions["pred_label_id"].to_numpy(dtype=int)
    metrics = compute_metrics(y_true, y_pred)

    predictions.to_csv(output_dir / "predictions.csv", index=False)
    save_metrics(metrics, output_dir / "metrics.json")
    save_confusion_matrix(y_true, y_pred, output_dir / "confusion_matrix.png")
    print(metrics)


if __name__ == "__main__":
    main()
