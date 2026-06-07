from __future__ import annotations

import numpy as np
import torch
from torchvision.models import ResNet50_Weights, resnet50


def build_resnet50_backbone(device: str | torch.device = "cpu") -> tuple[torch.nn.Module, object]:
    """Return pretrained ResNet50 without its classification head and its transforms."""
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, weights.transforms()


@torch.no_grad()
def extract_resnet50_frame_embeddings(
    frames: np.ndarray,
    model: torch.nn.Module,
    transform,
    device: str | torch.device = "cpu",
    batch_size: int = 8,
) -> np.ndarray:
    """Extract frame-level ResNet50 embeddings from RGB uint8 frames."""
    from PIL import Image

    embeddings: list[np.ndarray] = []
    pil_frames = [Image.fromarray(frame) for frame in frames]
    for start in range(0, len(pil_frames), batch_size):
        batch = torch.stack([transform(img) for img in pil_frames[start : start + batch_size]])
        batch = batch.to(device)
        out = model(batch).detach().cpu().numpy()
        embeddings.append(out)
    return np.concatenate(embeddings, axis=0).astype(np.float32)
