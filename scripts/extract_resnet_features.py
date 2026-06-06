import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.metadata import load_metadata
from src.data.video import sample_video_frames
from src.features.resnet import build_resnet50_backbone, extract_resnet50_frame_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frame-level ResNet50 embeddings for deep model.")
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument("--video-dir", default="data/cv_module_videos")
    parser.add_argument("--output", default="artifacts/features/resnet50_frame_embeddings.joblib")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_metadata(args.metadata, args.video_dir)
    model, transform = build_resnet50_backbone(args.device)
    features = []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="resnet50 embeddings"):
        frames = sample_video_frames(row.video_path, num_frames=args.frames, resize=None, rgb=True)
        embeddings = extract_resnet50_frame_embeddings(
            frames,
            model=model,
            transform=transform,
            device=args.device,
            batch_size=args.batch_size,
        )
        features.append(embeddings)

    tensor = np.stack(features).astype(np.float32)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "video_ids": df["video_id"].tolist(),
            "features": tensor,
            "labels": df["label_id"].to_numpy(dtype=int),
            "risk_level": df["risk_level"].tolist(),
            "config": {"frames": args.frames, "feature_type": "resnet50_frame_embeddings"},
        },
        output,
    )
    print(f"Saved ResNet50 embeddings {tensor.shape} to {output}")


if __name__ == "__main__":
    main()
