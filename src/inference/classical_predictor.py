from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np

from src.data.video import sample_video_frames, to_grayscale
from src.features.classical import extract_classical_features


class ClassicalVideoRiskPredictor:
    """Serve classical HOG + motion models from deployment artifacts."""

    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.manifest = self._load_manifest()
        self.label_order = list(self.manifest["label_order"])
        self.model_version = str(self.manifest["model_version"])
        self.num_frames = int(self.manifest["input"]["num_frames"])
        resize = self.manifest["input"]["resize"]
        self.resize = (int(resize["width"]), int(resize["height"]))
        self.contrast_normalize = bool(self.manifest["input"].get("contrast_normalize", False))

        bundle = joblib.load(self.artifact_dir / self.manifest["model_artifact"])
        self.model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle

    def _load_manifest(self) -> dict[str, Any]:
        manifest_path = self.artifact_dir / "model_manifest.json"
        with manifest_path.open(encoding="utf-8") as f:
            return json.load(f)

    def predict_video(self, video_path: str | Path) -> dict[str, Any]:
        started = time.perf_counter()
        frames = sample_video_frames(
            video_path,
            num_frames=self.num_frames,
            resize=self.resize,
            rgb=True,
        )
        gray = to_grayscale(frames)
        gray = self._maybe_contrast_normalize(gray)
        features = extract_classical_features(gray)[None, :]

        pred_idx = int(self.model.predict(features)[0])
        scores = self._scores(features)
        confidence = float(scores[pred_idx])
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        return {
            "risk_level": self.label_order[pred_idx],
            "confidence": confidence,
            "probabilities": {
                label: float(scores[idx])
                for idx, label in enumerate(self.label_order)
            },
            "logits": [float(value) for value in scores],
            "model_version": self.model_version,
            "latency_ms": float(elapsed_ms),
        }

    def _maybe_contrast_normalize(self, gray_frames: np.ndarray) -> np.ndarray:
        if not self.contrast_normalize:
            return gray_frames
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        normalized = []
        for frame in gray_frames:
            frame_u8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            normalized.append(clahe.apply(frame_u8).astype(np.float32) / 255.0)
        return np.stack(normalized)

    def _scores(self, features: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(features)[0], dtype=np.float64)
        if hasattr(self.model, "decision_function"):
            decision_scores = np.asarray(self.model.decision_function(features)[0], dtype=np.float64)
            exp = np.exp(decision_scores - decision_scores.max())
            return exp / exp.sum()
        scores = np.zeros(len(self.label_order), dtype=np.float64)
        pred_idx = int(self.model.predict(features)[0])
        scores[pred_idx] = 1.0
        return scores
