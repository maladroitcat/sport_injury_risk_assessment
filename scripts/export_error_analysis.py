import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.metadata import load_metadata
from src.data.video import sample_video_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export five mispredictions and representative frames.")
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument("--video-dir", default="data/cv_module_videos")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", default="artifacts/error_analysis")
    parser.add_argument("--frames", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", "artifacts/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "artifacts/cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = parse_args()
    metadata = load_metadata(args.metadata, args.video_dir)
    preds = pd.read_csv(args.predictions)
    merged = preds.merge(metadata.drop(columns=["risk_level", "label_id"]), on="video_id", how="left")
    mistakes = merged[merged["label_id"] != merged["pred_label_id"]].head(5).copy()

    output_dir = Path(args.output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    mistakes.to_csv(output_dir / "mispredictions.csv", index=False)

    for row in mistakes.itertuples(index=False):
        frames = sample_video_frames(row.video_path, num_frames=args.frames, resize=(224, 224), rgb=True)
        fig, axes = plt.subplots(1, len(frames), figsize=(3 * len(frames), 3))
        if len(frames) == 1:
            axes = [axes]
        for ax, frame in zip(axes, frames):
            ax.imshow(frame)
            ax.axis("off")
        fig.suptitle(f"{row.video_id}: true={row.risk_level}, pred={row.pred_risk_level}")
        fig.tight_layout()
        fig.savefig(frames_dir / f"{row.video_id}.png", dpi=160)
        plt.close(fig)

    print(f"Exported {len(mistakes)} mistakes to {output_dir}")


if __name__ == "__main__":
    main()
