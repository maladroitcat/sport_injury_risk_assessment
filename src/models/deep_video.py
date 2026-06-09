import torch


class MeanPooledResNetHead(torch.nn.Module):
    """Small temporal head over frozen frame embeddings."""

    def __init__(self, embedding_dim: int = 2048, hidden_dim: int = 256, num_classes: int = 3, dropout: float = 0.25):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(embedding_dim),
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, frame_embeddings: torch.Tensor) -> torch.Tensor:
        """Accept [batch, frames, embedding_dim] and return [batch, classes]."""
        pooled = frame_embeddings.mean(dim=1)
        return self.classifier(pooled)
