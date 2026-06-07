from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def sample_video_frames(
    video_path: str | Path,
    num_frames: int = 16,
    resize: tuple[int, int] | None = None,
    rgb: bool = True,
) -> np.ndarray:
    """Sample evenly spaced frames from a video as uint8 array [T, H, W, C]."""
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise ValueError(f"Video has no readable frames: {video_path}")

    indices = np.linspace(0, max(frame_count - 1, 0), num_frames).astype(int)
    frames: list[np.ndarray] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        if resize is not None:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
        if rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError(f"No frames could be sampled from: {video_path}")

    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    return np.stack(frames[:num_frames]).astype(np.uint8)


def to_grayscale(frames: np.ndarray) -> np.ndarray:
    """Convert RGB frame stack to grayscale float32 in [0, 1]."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected RGB frames [T,H,W,3], got {frames.shape}")
    gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames])
    return gray.astype(np.float32) / 255.0
