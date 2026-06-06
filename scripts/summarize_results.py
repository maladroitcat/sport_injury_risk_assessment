import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.summarize import summarize_metric_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize available model metrics into one CSV.")
    parser.add_argument("--output", default="artifacts/reports/model_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = {
        "baseline": "artifacts/models/baseline/metrics.json",
        "classical_logistic_regression": "artifacts/models/classical/logistic_regression_metrics.json",
        "classical_linear_svm": "artifacts/models/classical/linear_svm_metrics.json",
        "classical_random_forest": "artifacts/models/classical/random_forest_metrics.json",
        "deep_resnet50": "artifacts/models/deep_resnet50/metrics.json",
    }
    existing = {name: path for name, path in candidates.items() if Path(path).exists()}
    if not existing:
        raise FileNotFoundError("No metrics files found. Train at least one model first.")
    df = summarize_metric_files(existing, args.output)
    print(df)


if __name__ == "__main__":
    main()
