import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.metadata import save_label_mapping
from src.data.splits import create_stratified_folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create reusable stratified CV folds.")
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument("--video-dir", default="data/cv_module_videos")
    parser.add_argument("--output", default="artifacts/splits/folds.csv")
    parser.add_argument("--labels", default="artifacts/splits/labels.json")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folds = create_stratified_folds(
        metadata_path=args.metadata,
        video_dir=args.video_dir,
        output_path=args.output,
        n_splits=args.n_splits,
        random_state=args.seed,
    )
    save_label_mapping(args.labels)
    print(f"Saved {len(folds)} fold allocations to {args.output}")


if __name__ == "__main__":
    main()
