import argparse
import copy
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.metadata import ID_TO_LABEL, load_metadata
from src.data.splits import load_or_create_folds
from src.evaluation.metrics import compute_metrics, save_confusion_matrix, save_metrics
from src.models.deep_video import MeanPooledResNetHead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train neural temporal head on frozen ResNet50 frame embeddings.")
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument("--video-dir", default="data/cv_module_videos")
    parser.add_argument("--folds", default="artifacts/splits/folds.csv")
    parser.add_argument("--features", default="artifacts/features/resnet50_frame_embeddings.joblib")
    parser.add_argument("--output-dir", default="artifacts/models/deep_resnet50")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
) -> tuple[MeanPooledResNetHead, np.ndarray, dict]:
    torch.manual_seed(args.seed)
    model = MeanPooledResNetHead(embedding_dim=X_train.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    best_state = copy.deepcopy(model.state_dict())
    best_macro_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(X_val_t).argmax(dim=1).cpu().numpy()
        metrics = compute_metrics(y_val, pred)
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(X_val_t).argmax(dim=1).cpu().numpy()
    return model, pred, {"best_epoch": best_epoch, "best_macro_f1": best_macro_f1}


def main() -> None:
    args = parse_args()
    df = load_metadata(args.metadata, args.video_dir)
    folds = load_or_create_folds(output_path=args.folds, metadata_path=args.metadata, video_dir=args.video_dir)
    df = df.merge(folds[["video_id", "fold"]], on="video_id", how="left")
    bundle = joblib.load(args.features)
    feature_df = pd.DataFrame({"video_id": bundle["video_ids"], "feature_idx": np.arange(len(bundle["video_ids"]))})
    df = df.merge(feature_df, on="video_id", how="inner").reset_index(drop=True)

    X_all = bundle["features"]
    y_all = df["label_id"].to_numpy(dtype=int)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_predictions = []
    fold_details = {}
    for fold in sorted(df["fold"].unique()):
        train_rows = df.index[df["fold"] != fold].to_numpy()
        val_rows = df.index[df["fold"] == fold].to_numpy()
        train_idx = df.loc[train_rows, "feature_idx"].to_numpy(dtype=int)
        val_idx = df.loc[val_rows, "feature_idx"].to_numpy(dtype=int)

        _, pred, details = train_one_fold(X_all[train_idx], y_all[train_rows], X_all[val_idx], y_all[val_rows], args)
        fold_details[str(fold)] = details
        pred_df = df.loc[val_rows, ["video_id", "risk_level", "label_id", "fold"]].copy()
        pred_df["pred_label_id"] = pred
        pred_df["pred_risk_level"] = [ID_TO_LABEL[int(p)] for p in pred]
        all_predictions.append(pred_df)

    predictions = pd.concat(all_predictions, ignore_index=True)
    y_true = predictions["label_id"].to_numpy(dtype=int)
    y_pred = predictions["pred_label_id"].to_numpy(dtype=int)
    metrics = compute_metrics(y_true, y_pred)
    metrics["fold_details"] = fold_details
    metrics["feature_config"] = bundle.get("config", {})

    predictions.to_csv(output_dir / "predictions.csv", index=False)
    save_metrics(metrics, output_dir / "metrics.json")
    save_confusion_matrix(y_true, y_pred, output_dir / "confusion_matrix.png")

    final_args = copy.copy(args)
    final_args.epochs = max(10, min(args.epochs, int(np.mean([d["best_epoch"] for d in fold_details.values()])) + 1))
    all_idx = df["feature_idx"].to_numpy(dtype=int)
    final_model, _, _ = train_one_fold(X_all[all_idx], y_all, X_all[all_idx], y_all, final_args)
    torch.save(
        {"model_state": final_model.state_dict(), "label_order": list(ID_TO_LABEL.values()), "feature_config": bundle.get("config", {})},
        output_dir / "final_head.pt",
    )
    print(metrics)


if __name__ == "__main__":
    main()
