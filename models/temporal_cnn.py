from __future__ import annotations

import torch
import torch.nn as nn


class TemporalCNNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        channels: int = 64,
        kernel_size: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(int(input_dim), int(hidden_dim))
        padding = kernel_size // 2
        self.encoder = nn.Sequential(
            nn.Conv1d(hidden_dim, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, signal: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        projected = self.input_projection(signal)
        encoded = self.encoder(projected.transpose(1, 2))
        pooled = encoded.mean(dim=2)
        return self.head(pooled).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
