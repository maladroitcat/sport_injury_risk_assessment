import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frame-count experiments for classical and deep models.")
    parser.add_argument("--frame-counts", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def maybe_run(command: list[str], output_path: Path, skip_existing: bool) -> None:
    if skip_existing and output_path.exists():
        print(f"Skipping existing {output_path}", flush=True)
        return
    run_command(command)


def collect_metrics(base_dir: Path) -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(base_dir.glob("**/*_metrics.json")):
        if metrics_path.name == "summary_metrics.json":
            continue
        with metrics_path.open() as f:
            metrics = json.load(f)

        parent = metrics_path.parent.name
        model = metrics_path.stem.replace("_metrics", "")
        parts = parent.split("_")
        frames = int(parts[-1])
        approach = "_".join(parts[:-1])
        rows.append(
            {
                "frames": frames,
                "approach": approach,
                "model": model,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "high_recall": metrics["high_recall"],
            }
        )

    for metrics_path in sorted(base_dir.glob("deep_resnet50_*/metrics.json")):
        with metrics_path.open() as f:
            metrics = json.load(f)
        frames = int(metrics_path.parent.name.split("_")[-1])
        rows.append(
            {
                "frames": frames,
                "approach": "deep_resnet50",
                "model": "temporal_head",
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "high_recall": metrics["high_recall"],
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["macro_f1", "high_recall"], ascending=False)
    return df


def best_by_frame_and_approach(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    ranked = summary.sort_values(
        ["frames", "approach", "macro_f1", "high_recall"],
        ascending=[True, True, False, False],
    )
    return ranked.drop_duplicates(["frames", "approach"], keep="first").sort_values(
        ["macro_f1", "high_recall"],
        ascending=False,
    )


def main() -> None:
    args = parse_args()
    base_dir = Path("artifacts/experiments/frame_count")
    base_dir.mkdir(parents=True, exist_ok=True)

    for frames in args.frame_counts:
        plain_feature_path = base_dir / f"plain_hog_motion_{frames}.joblib"
        plain_model_dir = base_dir / f"plain_classical_{frames}"
        contrast_feature_path = base_dir / f"contrast_hog_motion_{frames}.joblib"
        contrast_model_dir = base_dir / f"contrast_classical_{frames}"
        resnet_feature_path = base_dir / f"resnet50_embeddings_{frames}.joblib"
        resnet_model_dir = base_dir / f"deep_resnet50_{frames}"

        maybe_run(
            [
                sys.executable,
                "scripts/extract_classical_features.py",
                "--frames",
                str(frames),
                "--size",
                str(args.size),
                "--output",
                str(plain_feature_path),
            ],
            plain_feature_path,
            args.skip_existing,
        )
        maybe_run(
            [
                sys.executable,
                "scripts/train_classical.py",
                "--features",
                str(plain_feature_path),
                "--output-dir",
                str(plain_model_dir),
            ],
            plain_model_dir / "summary_metrics.json",
            args.skip_existing,
        )

        maybe_run(
            [
                sys.executable,
                "scripts/extract_classical_features.py",
                "--frames",
                str(frames),
                "--size",
                str(args.size),
                "--contrast-normalize",
                "--output",
                str(contrast_feature_path),
            ],
            contrast_feature_path,
            args.skip_existing,
        )
        maybe_run(
            [
                sys.executable,
                "scripts/train_classical.py",
                "--features",
                str(contrast_feature_path),
                "--output-dir",
                str(contrast_model_dir),
            ],
            contrast_model_dir / "summary_metrics.json",
            args.skip_existing,
        )

        maybe_run(
            [
                sys.executable,
                "scripts/extract_resnet_features.py",
                "--frames",
                str(frames),
                "--output",
                str(resnet_feature_path),
            ],
            resnet_feature_path,
            args.skip_existing,
        )
        maybe_run(
            [
                sys.executable,
                "scripts/train_deep.py",
                "--features",
                str(resnet_feature_path),
                "--output-dir",
                str(resnet_model_dir),
            ],
            resnet_model_dir / "metrics.json",
            args.skip_existing,
        )

    summary = collect_metrics(base_dir)
    summary.to_csv(base_dir / "summary.csv", index=False)
    best_summary = best_by_frame_and_approach(summary)
    best_summary.to_csv(base_dir / "summary_best.csv", index=False)
    print(best_summary)


if __name__ == "__main__":
    main()
