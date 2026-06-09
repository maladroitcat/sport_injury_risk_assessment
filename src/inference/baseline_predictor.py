from __future__ import annotations

from typing import Any


class ModerateBaselinePredictor:
    """Baseline predictor that always returns moderate risk."""

    model_version = "baseline_moderate"
    label_order = ["low", "moderate", "high"]

    def predict_video(self, video_path: str) -> dict[str, Any]:
        return {
            "risk_level": "moderate",
            "confidence": 1.0,
            "probabilities": {
                "low": 0.0,
                "moderate": 1.0,
                "high": 0.0,
            },
            "logits": [0.0, 1.0, 0.0],
            "model_version": self.model_version,
            "latency_ms": 0.0,
        }
