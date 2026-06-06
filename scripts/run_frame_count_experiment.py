import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classical frame-count experiment.")
    parser.add_argument("--frame-counts", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for frames in args.frame_counts:
        feature_path = f"artifacts/experiments/frame_count/features_hog_motion_{frames}.joblib"
        model_dir = f"artifacts/experiments/frame_count/classical_{frames}"
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                "scripts/extract_classical_features.py",
                "--frames",
                str(frames),
                "--size",
                str(args.size),
                "--output",
                feature_path,
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "scripts/train_classical.py",
                "--features",
                feature_path,
                "--output-dir",
                model_dir,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
