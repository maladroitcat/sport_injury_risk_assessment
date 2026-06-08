import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare light preprocessing transformations.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--deep-frames", type=int, default=4)
    return parser.parse_args()


def maybe_run(command: list[str], output_path: str | Path, skip_existing: bool) -> None:
    output_path = Path(output_path)
    if skip_existing and output_path.exists():
        print(f"Skipping existing {output_path}", flush=True)
        return
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def load_metrics(name: str, path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        metrics = json.load(f)
    return {
        "model": name,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "high_recall": metrics["high_recall"],
    }


def main() -> None:
    """Compare deterministic classical and ResNet50 preprocessing transformations."""
    args = parse_args()
    base_dir = Path("artifacts/experiments/transformations")

    configs = [
        ("plain", []),
        ("contrast_normalized", ["--contrast-normalize"]),
    ]
    for name, extra_args in configs:
        feature_path = base_dir / f"{name}_hog_motion.joblib"
        model_dir = base_dir / f"{name}_classical"
        maybe_run(
            [
                sys.executable,
                "scripts/extract_classical_features.py",
                "--output",
                str(feature_path),
                *extra_args,
            ],
            feature_path,
            args.skip_existing,
        )
        maybe_run(
            [
                sys.executable,
                "scripts/train_classical.py",
                "--features",
                str(feature_path),
                "--output-dir",
                str(model_dir),
            ],
            model_dir / "summary_metrics.json",
            args.skip_existing,
        )

    deep_configs = ["none", "brightness_boost", "contrast_boost", "mild_blur"]
    for transform_name in deep_configs:
        feature_name = "plain" if transform_name == "none" else transform_name
        feature_path = base_dir / f"resnet50_{feature_name}_{args.deep_frames}f.joblib"
        model_dir = base_dir / f"deep_resnet50_{feature_name}_{args.deep_frames}f"
        command = [
            sys.executable,
            "scripts/extract_resnet_features.py",
            "--frames",
            str(args.deep_frames),
            "--output",
            str(feature_path),
        ]
        if transform_name != "none":
            command.extend(["--light-transform", transform_name])
        maybe_run(command, feature_path, args.skip_existing)
        maybe_run(
            [
                sys.executable,
                "scripts/train_deep.py",
                "--features",
                str(feature_path),
                "--output-dir",
                str(model_dir),
            ],
            model_dir / "metrics.json",
            args.skip_existing,
        )

    rows = []
    for metrics_path in sorted(base_dir.glob("*_classical/*_metrics.json")):
        if metrics_path.name == "summary_metrics.json":
            continue
        rows.append(load_metrics(f"{metrics_path.parent.name}_{metrics_path.stem.replace('_metrics', '')}", metrics_path))
    for metrics_path in sorted(base_dir.glob("deep_resnet50_*_metrics.json")):
        rows.append(load_metrics(metrics_path.stem.replace("_metrics", ""), metrics_path))
    for metrics_path in sorted(base_dir.glob("deep_resnet50_*/*metrics.json")):
        rows.append(load_metrics(metrics_path.parent.name, metrics_path))

    summary = pd.DataFrame(rows).drop_duplicates("model").sort_values(["macro_f1", "high_recall"], ascending=False)
    summary.to_csv(base_dir / "summary.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()
