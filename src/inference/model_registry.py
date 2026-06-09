from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.inference.baseline_predictor import ModerateBaselinePredictor
from src.inference.classical_predictor import ClassicalVideoRiskPredictor
from src.inference.deep_predictor import VideoRiskPredictor


DEFAULT_MODEL_VERSION = "resnet50_mild_blur_4f"
MODEL_ORDER = [
    "resnet50_mild_blur_4f",
    "classical_hog_motion_svm_16f",
    "baseline_moderate",
]

MODEL_LABELS = {
    "resnet50_mild_blur_4f": "Default neural network - ResNet50 mild blur",
    "classical_hog_motion_svm_16f": "Classical model - linear SVM",
    "baseline_moderate": "Baseline - always moderate",
}


class ModelRegistry:
    """Discover deployment artifacts and lazily load selected model predictors."""

    def __init__(
        self,
        root_dir: str | Path = "model_artifacts",
        default_model_version: str = DEFAULT_MODEL_VERSION,
        device: str = "cpu",
        batch_size: int = 4,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.default_model_version = default_model_version
        self.device = device
        self.batch_size = batch_size
        self._manifests = self._discover_manifests()
        self._predictors: dict[str, Any] = {}
        if default_model_version not in self._manifests:
            raise ValueError(f"Default model artifact not found: {default_model_version}")
        self.get(default_model_version)

    def list_models(self) -> list[dict[str, Any]]:
        models = []
        for model_version in MODEL_ORDER:
            if model_version not in self._manifests:
                continue
            manifest = self._manifests[model_version]
            metrics = self._load_metrics(model_version)
            models.append(
                {
                    "model_version": model_version,
                    "label": MODEL_LABELS.get(model_version, model_version),
                    "model_type": manifest["model_type"],
                    "is_default": model_version == self.default_model_version,
                    "metrics": {
                        "accuracy": metrics.get("accuracy"),
                        "macro_f1": metrics.get("macro_f1"),
                        "high_recall": metrics.get("high_recall"),
                    },
                    "input": manifest.get("input", {}),
                }
            )
        return models

    def loaded_model_versions(self) -> list[str]:
        return sorted(self._predictors)

    def get(self, model_version: str | None = None):
        selected = model_version or self.default_model_version
        if selected not in self._manifests:
            raise KeyError(selected)
        if selected not in self._predictors:
            self._predictors[selected] = self._build_predictor(selected)
        return self._predictors[selected]

    def _discover_manifests(self) -> dict[str, dict[str, Any]]:
        manifests = {}
        for path in self.root_dir.glob("*/model_manifest.json"):
            with path.open(encoding="utf-8") as f:
                manifest = json.load(f)
            manifests[str(manifest["model_version"])] = manifest
        manifests["baseline_moderate"] = {
            "model_version": "baseline_moderate",
            "model_type": "baseline_always_moderate",
            "label_order": ["low", "moderate", "high"],
            "num_classes": 3,
            "input": {
                "num_frames": 0,
                "frame_sampling": "none",
                "uses_video_pixels": False,
            },
        }
        return manifests

    def _build_predictor(self, model_version: str):
        artifact_dir = self.root_dir / model_version
        model_type = self._manifests[model_version]["model_type"]
        if model_type == "baseline_always_moderate":
            return ModerateBaselinePredictor()
        if model_type == "frozen_resnet50_temporal_head":
            return VideoRiskPredictor(
                artifact_dir=artifact_dir,
                device=self.device,
                batch_size=self.batch_size,
            )
        if model_type.startswith("classical_cv_"):
            return ClassicalVideoRiskPredictor(artifact_dir=artifact_dir)
        raise ValueError(f"Unsupported model_type for {model_version}: {model_type}")

    def _load_metrics(self, model_version: str) -> dict[str, Any]:
        metrics_path = self.root_dir / model_version / "metrics.json"
        if not metrics_path.exists():
            return {}
        with metrics_path.open(encoding="utf-8") as f:
            return json.load(f)
