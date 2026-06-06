from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.feature import hog


@dataclass(frozen=True)
class ClassicalFeatureConfig:
    orientations: int = 9
    pixels_per_cell: tuple[int, int] = (8, 8)
    cells_per_block: tuple[int, int] = (2, 2)
    motion_threshold: float = 0.08


def extract_hog_summary(
    gray_frames: np.ndarray,
    config: ClassicalFeatureConfig = ClassicalFeatureConfig(),
) -> np.ndarray:
    """Extract HOG per grayscale frame and average to one video-level vector."""
    hog_vectors = [
        hog(
            frame,
            orientations=config.orientations,
            pixels_per_cell=config.pixels_per_cell,
            cells_per_block=config.cells_per_block,
            block_norm="L2-Hys",
            feature_vector=True,
        )
        for frame in gray_frames
    ]
    return np.mean(np.stack(hog_vectors), axis=0).astype(np.float32)


def extract_motion_summary(
    gray_frames: np.ndarray,
    config: ClassicalFeatureConfig = ClassicalFeatureConfig(),
) -> np.ndarray:
    """Summarize frame-to-frame absolute differences."""
    if len(gray_frames) < 2:
        return np.zeros(5, dtype=np.float32)

    diffs = np.abs(np.diff(gray_frames, axis=0))
    frame_means = diffs.mean(axis=(1, 2))
    changed = (diffs > config.motion_threshold).mean(axis=(1, 2))
    features = np.array(
        [
            float(frame_means.mean()),
            float(frame_means.max()),
            float(frame_means.std()),
            float(changed.mean()),
            float(np.percentile(diffs, 95)),
        ],
        dtype=np.float32,
    )
    return features


def extract_classical_features(gray_frames: np.ndarray) -> np.ndarray:
    """Combine HOG shape/edge features with motion-summary features."""
    hog_features = extract_hog_summary(gray_frames)
    motion_features = extract_motion_summary(gray_frames)
    return np.concatenate([hog_features, motion_features]).astype(np.float32)
