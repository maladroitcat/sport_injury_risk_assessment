import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.metadata import load_metadata
from src.data.video import sample_video_frames, to_grayscale
from src.features.classical import extract_classical_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract strict classical CV features.")
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument("--video-dir", default="data/cv_module_videos")
    parser.add_argument("--output", default="artifacts/features/classical_hog_motion.joblib")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument(
        "--contrast-normalize",
        action="store_true",
        help="Apply CLAHE contrast normalization before HOG/motion extraction.",
    )
    return parser.parse_args()


def maybe_contrast_normalize(gray_frames: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return gray_frames
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = []
    for frame in gray_frames:
        frame_u8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        normalized.append(clahe.apply(frame_u8).astype(np.float32) / 255.0)
    return np.stack(normalized)


def main() -> None:
    args = parse_args()
    df = load_metadata(args.metadata, args.video_dir)
    features = []
    failed = []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="classical features"):
        try:
            frames = sample_video_frames(
                row.video_path,
                num_frames=args.frames,
                resize=(args.size, args.size),
                rgb=True,
            )
            gray = to_grayscale(frames)
            gray = maybe_contrast_normalize(gray, args.contrast_normalize)
            features.append(extract_classical_features(gray))
        except Exception as exc:
            failed.append({"video_id": row.video_id, "error": str(exc)})
            features.append(None)

    valid_mask = np.array([feature is not None for feature in features])
    if not valid_mask.all():
        failed_df = pd.DataFrame(failed)
        failed_path = Path(args.output).with_suffix(".failures.csv")
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        failed_df.to_csv(failed_path, index=False)
        raise RuntimeError(f"Feature extraction failed for {len(failed)} videos. See {failed_path}")

    matrix = np.stack(features).astype(np.float32)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "video_ids": df["video_id"].tolist(),
            "features": matrix,
            "labels": df["label_id"].to_numpy(dtype=int),
            "risk_level": df["risk_level"].tolist(),
            "config": {
                "frames": args.frames,
                "size": args.size,
                "feature_type": "hog_motion",
                "contrast_normalize": args.contrast_normalize,
            },
        },
        output,
    )
    print(f"Saved classical features {matrix.shape} to {output}")


if __name__ == "__main__":
    main()
