from __future__ import annotations

import torch
import torch.nn as nn


class SequenceClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class GRUClassifier(SequenceClassifier):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.3) -> None:
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.encoder = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1),
        )

    def forward(self, signal: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        projected = self.input_projection(signal)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(projected, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, hidden = self.encoder(packed)
        else:
            _, hidden = self.encoder(projected)
        return self.classifier(hidden[-1]).squeeze(-1)


class LSTMClassifier(SequenceClassifier):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.3) -> None:
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.encoder = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1),
        )

    def forward(self, signal: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        projected = self.input_projection(signal)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(projected, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (hidden, _) = self.encoder(packed)
        else:
            _, (hidden, _) = self.encoder(projected)
        return self.classifier(hidden[-1]).squeeze(-1)
