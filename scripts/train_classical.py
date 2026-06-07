import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.metadata import ID_TO_LABEL, load_metadata
from src.data.splits import load_or_create_folds
from src.evaluation.metrics import compute_metrics, save_confusion_matrix, save_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train strict classical CV classifiers.")
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument("--video-dir", default="data/cv_module_videos")
    parser.add_argument("--folds", default="artifacts/splits/folds.csv")
    parser.add_argument("--features", default="artifacts/features/classical_hog_motion.joblib")
    parser.add_argument("--output-dir", default="artifacts/models/classical")
    return parser.parse_args()


def build_models(random_state: int = 42) -> dict[str, object]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "linear_svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearSVC(class_weight="balanced", random_state=random_state, max_iter=5000)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def main() -> None:
    args = parse_args()
    df = load_metadata(args.metadata, args.video_dir)
    folds = load_or_create_folds(output_path=args.folds, metadata_path=args.metadata, video_dir=args.video_dir)
    df = df.merge(folds[["video_id", "fold"]], on="video_id", how="left")
    feature_bundle = joblib.load(args.features)
    feature_df = pd.DataFrame({"video_id": feature_bundle["video_ids"]})
    feature_df["feature_idx"] = np.arange(len(feature_df))
    df = df.merge(feature_df, on="video_id", how="inner").reset_index(drop=True)

    X = feature_bundle["features"]
    y = df["label_id"].to_numpy(dtype=int)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    model_predictions = {}
    for model_name, model in build_models().items():
        fold_predictions = []
        for fold in sorted(df["fold"].unique()):
            train_rows = df.index[df["fold"] != fold].to_numpy()
            val_rows = df.index[df["fold"] == fold].to_numpy()
            train_idx = df.loc[train_rows, "feature_idx"].to_numpy(dtype=int)
            val_idx = df.loc[val_rows, "feature_idx"].to_numpy(dtype=int)
            model.fit(X[train_idx], y[train_rows])
            pred = model.predict(X[val_idx])
            pred_df = df.loc[val_rows, ["video_id", "risk_level", "label_id", "fold"]].copy()
            pred_df["pred_label_id"] = pred
            pred_df["pred_risk_level"] = [ID_TO_LABEL[int(p)] for p in pred]
            pred_df["model"] = model_name
            fold_predictions.append(pred_df)

        predictions = pd.concat(fold_predictions, ignore_index=True)
        metrics = compute_metrics(
            predictions["label_id"].to_numpy(dtype=int),
            predictions["pred_label_id"].to_numpy(dtype=int),
        )
        summary[model_name] = metrics
        model_predictions[model_name] = predictions
        predictions.to_csv(output_dir / f"{model_name}_predictions.csv", index=False)
        save_metrics(metrics, output_dir / f"{model_name}_metrics.json")
        save_confusion_matrix(
            predictions["label_id"].to_numpy(dtype=int),
            predictions["pred_label_id"].to_numpy(dtype=int),
            output_dir / f"{model_name}_confusion_matrix.png",
        )

    best_name = max(summary, key=lambda name: (summary[name]["macro_f1"], summary[name]["high_recall"]))
    best_model = build_models()[best_name]
    all_idx = df["feature_idx"].to_numpy(dtype=int)
    best_model.fit(X[all_idx], y)
    joblib.dump(
        {"model": best_model, "feature_config": feature_bundle.get("config", {}), "label_order": list(ID_TO_LABEL.values())},
        output_dir / "best_model.joblib",
    )
    model_predictions[best_name].to_csv(output_dir / "best_predictions.csv", index=False)
    save_metrics({"best_model": best_name, "models": summary}, output_dir / "summary_metrics.json")
    print(f"Best classical model: {best_name}")


if __name__ == "__main__":
    main()
