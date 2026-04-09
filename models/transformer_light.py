from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class LightweightTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(int(input_dim), int(d_model))
        self.positional_encoding = PositionalEncoding(d_model=int(d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.classifier = nn.Sequential(
            nn.LayerNorm(int(d_model)),
            nn.Linear(int(d_model), int(d_model)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(d_model), 1),
        )

    def forward(self, signal: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input_projection(signal)
        x = self.positional_encoding(x)
        if lengths is not None:
            max_len = x.size(1)
            index = torch.arange(max_len, device=lengths.device).unsqueeze(0)
            mask = index >= lengths.unsqueeze(1)
        else:
            mask = None
        encoded = self.encoder(x, src_key_padding_mask=mask)
        if lengths is not None:
            valid = (~mask).unsqueeze(-1).float()
            pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            pooled = encoded.mean(dim=1)
        return self.classifier(pooled).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
