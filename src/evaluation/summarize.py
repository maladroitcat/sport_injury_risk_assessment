import json
from pathlib import Path

import pandas as pd


def flatten_metrics(name: str, metrics_path: str | Path) -> dict:
    with Path(metrics_path).open(encoding="utf-8") as f:
        metrics = json.load(f)
    return {
        "model": name,
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
        "high_recall": metrics.get("high_recall"),
    }


def summarize_metric_files(metric_files: dict[str, str | Path], output_path: str | Path) -> pd.DataFrame:
    rows = [flatten_metrics(name, path) for name, path in metric_files.items()]
    df = pd.DataFrame(rows).sort_values(["macro_f1", "high_recall"], ascending=False)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
