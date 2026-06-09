from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.data.video import sample_video_frames
from src.features.resnet import build_resnet50_backbone, extract_resnet50_frame_embeddings
from src.models.deep_video import MeanPooledResNetHead


class VideoRiskPredictor:
    """Serve the frozen ResNet50 feature extractor plus trained temporal head."""

    def __init__(
        self,
        artifact_dir: str | Path = "model_artifacts/resnet50_mild_blur_4f",
        device: str = "cpu",
        batch_size: int = 4,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.manifest = self._load_manifest()
        self.label_order = list(self.manifest["label_order"])
        self.num_frames = int(self.manifest["input"]["num_frames"])
        self.model_version = str(self.manifest["model_version"])

        self.backbone, self.transform = build_resnet50_backbone(self.device)
        self.head = MeanPooledResNetHead(
            embedding_dim=int(self.manifest["feature_extractor"]["embedding_dim"]),
            hidden_dim=int(self.manifest["temporal_head"]["hidden_dim"]),
            num_classes=int(self.manifest["num_classes"]),
            dropout=float(self.manifest["temporal_head"]["dropout"]),
        ).to(self.device)
        checkpoint = torch.load(
            self.artifact_dir / self.manifest["model_artifact"],
            map_location=self.device,
            weights_only=False,
        )
        self.head.load_state_dict(checkpoint["model_state"])
        self.head.eval()

    def _load_manifest(self) -> dict[str, Any]:
        manifest_path = self.artifact_dir / "model_manifest.json"
        with manifest_path.open(encoding="utf-8") as f:
            return json.load(f)

    def predict_video(self, video_path: str | Path) -> dict[str, Any]:
        started = time.perf_counter()
        frames = sample_video_frames(video_path, num_frames=self.num_frames, resize=None, rgb=True)
        frames = self._apply_light_transform(frames)
        embeddings = extract_resnet50_frame_embeddings(
            frames,
            model=self.backbone,
            transform=self.transform,
            device=self.device,
            batch_size=self.batch_size,
        )

        with torch.no_grad():
            features = torch.tensor(embeddings[None, ...], dtype=torch.float32, device=self.device)
            logits_tensor = self.head(features).squeeze(0)
            probabilities_tensor = torch.softmax(logits_tensor, dim=0)

        probabilities = probabilities_tensor.detach().cpu().numpy()
        logits = logits_tensor.detach().cpu().numpy()
        pred_idx = int(probabilities.argmax())
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        return {
            "risk_level": self.label_order[pred_idx],
            "confidence": float(probabilities[pred_idx]),
            "probabilities": {
                label: float(probabilities[idx])
                for idx, label in enumerate(self.label_order)
            },
            "logits": [float(value) for value in logits],
            "model_version": self.model_version,
            "latency_ms": float(elapsed_ms),
        }

    def _apply_light_transform(self, frames: np.ndarray) -> np.ndarray:
        light_transform = self.manifest["input"].get("light_transform")
        if light_transform in (None, "none"):
            return frames
        if light_transform == "brightness_boost":
            boosted = [
                cv2.convertScaleAbs(frame, alpha=1.0, beta=30)
                for frame in frames
            ]
            return np.stack(boosted).astype(np.uint8)
        if light_transform != "mild_blur":
            raise ValueError(f"Unsupported light_transform: {light_transform}")

        blurred = [
            cv2.GaussianBlur(frame, ksize=(3, 3), sigmaX=0)
            for frame in frames
        ]
        return np.stack(blurred).astype(np.uint8)
