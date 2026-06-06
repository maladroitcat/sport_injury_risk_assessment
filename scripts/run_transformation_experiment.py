import subprocess
import sys


def main() -> None:
    """Compare deterministic classical preprocessing with and without contrast normalization."""
    configs = [
        ("plain", []),
        ("contrast_normalized", ["--contrast-normalize"]),
    ]
    for name, extra_args in configs:
        feature_path = f"artifacts/experiments/transformations/{name}_hog_motion.joblib"
        model_dir = f"artifacts/experiments/transformations/{name}_classical"
        subprocess.run(
            [
                sys.executable,
                "scripts/extract_classical_features.py",
                "--output",
                feature_path,
                *extra_args,
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
