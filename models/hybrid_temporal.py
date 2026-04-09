from __future__ import annotations

import torch
import torch.nn as nn


class HybridTemporalClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        channels: int = 64,
        kernel_size: int = 5,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        input_dim = int(input_dim)
        hidden_dim = int(hidden_dim)
        channels = int(channels)

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        padding = kernel_size // 2
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(hidden_dim, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )

        summary_dim = input_dim * 5
        self.summary_mlp = nn.Sequential(
            nn.Linear(summary_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(channels + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, signal: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        projected = self.input_projection(signal)
        temporal = self.temporal_encoder(projected.transpose(1, 2)).mean(dim=2)

        mean = signal.mean(dim=1)
        std = signal.std(dim=1)
        minimum = signal.min(dim=1).values
        maximum = signal.max(dim=1).values
        delta = signal[:, -1, :] - signal[:, 0, :]
        summary = torch.cat([mean, std, minimum, maximum, delta], dim=1)
        summary_rep = self.summary_mlp(summary)

        fused = torch.cat([temporal, summary_rep], dim=1)
        return self.head(fused).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
