"""
Lightweight Transformer Trust Inference Model
==============================================
Implements the quantized transformer encoder for behavioral trust scoring
as described in the paper (Section IV.C).

Architecture:
  - Input:  behavioral sequence of shape (batch, K, 5)
  - Encoder: 4-layer multi-head self-attention, d_model=128
  - Output:  trust score in [0, 1] via sigmoid projection

Usage:
    from model.inference.transformer_model import TrustTransformer, TrustDataset
    model = TrustTransformer(config)
    score = model(sequence_tensor)  # -> tensor of shape (batch, 1)
"""

import json
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for the transformer encoder.
    Allows the model to be robust to irregular sampling intervals
    (positions encode observation index, not wall-clock time).
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer Trust Encoder
# ---------------------------------------------------------------------------

class TrustTransformer(nn.Module):
    """
    Lightweight transformer encoder for per-agent behavioral trust scoring.

    The model maps a K-step behavioral feature sequence to a scalar trust
    score in [0, 1]. Lower scores indicate higher anomaly probability.

    Key design choices (justified in paper Section V):
      - Self-attention over recurrent propagation for robustness to
        irregular acoustic sampling intervals
      - 4 layers / d_model=128 for 1.2M parameter count
      - Sigmoid output for direct trust score interpretation
    """

    def __init__(self, config: dict):
        super().__init__()
        arch = config["architecture"]
        self.input_dim = arch["input_dim"]
        self.d_model = arch["d_model"]
        self.seq_len = arch["seq_len"]
        self.num_heads = arch["num_heads"]
        self.num_layers = arch["num_layers"]
        self.d_ff = arch["d_ff"]
        self.dropout_p = arch["dropout"]

        # Input projection: 5 features -> d_model
        self.input_proj = nn.Linear(self.input_dim, self.d_model)

        # Positional encoding
        self.pos_enc = PositionalEncoding(self.d_model, self.seq_len + 1,
                                           self.dropout_p)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout_p,
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        # Aggregation + output projection
        self.pool = nn.AdaptiveAvgPool1d(1)       # mean pooling over sequence
        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Behavioral sequence tensor of shape (batch, K, input_dim).

        Returns:
            trust_score: Tensor of shape (batch, 1) with values in [0, 1].
                         Lower values indicate higher anomaly probability.
        """
        # Input projection: (batch, K, 5) -> (batch, K, d_model)
        x = self.input_proj(x)

        # Add positional encoding
        x = self.pos_enc(x)

        # Transformer encoding: (batch, K, d_model)
        x = self.transformer(x)

        # Mean pooling over sequence dimension: (batch, d_model)
        x = x.permute(0, 2, 1)           # (batch, d_model, K)
        x = self.pool(x).squeeze(-1)     # (batch, d_model)

        # Output projection: (batch, 1)
        trust_score = self.output_proj(x)
        return trust_score

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TrustDataset(Dataset):
    """
    PyTorch Dataset for behavioral sequence trust training.

    Args:
        sequences: list of dicts with keys 'sequence' (K x 5) and 'label'
        seq_len: sequence length K (pad or truncate)
    """

    def __init__(self, sequences: list, seq_len: int = 64):
        self.seq_len = seq_len
        self.data = []
        for item in sequences:
            seq = np.array(item["sequence"], dtype=np.float32)
            label = float(item["label"])
            # Pad or truncate to seq_len
            if seq.shape[0] < seq_len:
                pad = np.zeros((seq_len - seq.shape[0], seq.shape[1]),
                               dtype=np.float32)
                seq = np.vstack([seq, pad])
            else:
                seq = seq[:seq_len]
            self.data.append((seq, label))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq, label = self.data[idx]
        return torch.from_numpy(seq), torch.tensor([label], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_epoch(model: TrustTransformer, loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module) -> Tuple[float, float]:
    """Train for one epoch. Returns (mean_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for sequences, labels in loader:
        optimizer.zero_grad()
        outputs = model(sequences)           # (batch, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * sequences.size(0)
        predicted = (outputs.detach() < 0.65).float()  # below tau_min → anomalous
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model: TrustTransformer, loader: DataLoader,
             criterion: nn.Module,
             tau_min: float = 0.65) -> Tuple[float, float, float, float]:
    """Evaluate model. Returns (loss, accuracy, precision, recall)."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in loader:
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * sequences.size(0)
            preds = (outputs < tau_min).float()
            all_preds.extend(preds.numpy().flatten().tolist())
            all_labels.extend(labels.numpy().flatten().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    return total_loss / len(all_labels), accuracy, precision, recall


def train_model(config_path: str = "model/configs/transformer_config.json",
                data_path: str = "data/raw/behavioral_sequences.json",
                checkpoint_dir: str = "model/checkpoints/",
                verbose: bool = True) -> TrustTransformer:
    """
    Full training pipeline for the trust transformer.

    Args:
        config_path: Path to model config JSON.
        data_path:   Path to behavioral sequences JSON.
        checkpoint_dir: Where to save model checkpoints.
        verbose: Print training progress.

    Returns:
        Trained TrustTransformer model.
    """
    import json as _json
    with open(config_path, "r") as f:
        config = _json.load(f)

    with open(data_path, "r") as f:
        sequences = _json.load(f)

    # Reproducibility
    seed = config["training"]["random_seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Split dataset
    n = len(sequences)
    np.random.shuffle(sequences)
    n_train = int(n * config["training"]["train_split"])
    n_val = int(n * config["training"]["val_split"])
    train_data = sequences[:n_train]
    val_data = sequences[n_train: n_train + n_val]
    test_data = sequences[n_train + n_val:]

    seq_len = config["architecture"]["seq_len"]
    batch_size = config["training"]["batch_size"]

    train_loader = DataLoader(TrustDataset(train_data, seq_len),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TrustDataset(val_data, seq_len),
                            batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TrustDataset(test_data, seq_len),
                             batch_size=batch_size, shuffle=False)

    # Model
    model = TrustTransformer(config)
    if verbose:
        print(f"Model parameters: {model.count_parameters():,}")

    # Optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    criterion = nn.BCELoss()

    best_val_acc = 0.0
    patience_count = 0
    patience = config["training"]["early_stopping_patience"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_prec, val_rec = evaluate(
            model, val_loader, criterion,
            tau_min=config["trust_scoring"]["threshold_tau_min"]
        )
        scheduler.step(val_loss)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
                  f"Prec: {val_prec:.4f}, Rec: {val_rec:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_count = 0
            torch.save(model.state_dict(),
                       os.path.join(checkpoint_dir, "best_model.pt"))
        else:
            patience_count += 1
            if patience_count >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(
        os.path.join(checkpoint_dir, "best_model.pt"), weights_only=True
    ))
    test_loss, test_acc, test_prec, test_rec = evaluate(
        model, test_loader, criterion,
        tau_min=config["trust_scoring"]["threshold_tau_min"]
    )
    if verbose:
        print(f"\nTest Set Results:")
        print(f"  Accuracy:  {test_acc:.4f}")
        print(f"  Precision: {test_prec:.4f}")
        print(f"  Recall:    {test_rec:.4f}")

    return model


# ---------------------------------------------------------------------------
# Inference utility
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str,
               config_path: str = "model/configs/transformer_config.json"
               ) -> TrustTransformer:
    """Load a trained model from a checkpoint."""
    with open(config_path, "r") as f:
        config = json.load(f)
    model = TrustTransformer(config)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True,
                                      map_location="cpu"))
    model.eval()
    return model


def compute_trust_score(model: TrustTransformer,
                         sequence: np.ndarray) -> float:
    """
    Compute trust score for a single behavioral sequence.

    Args:
        model: Trained TrustTransformer.
        sequence: numpy array of shape (K, 5).

    Returns:
        trust_score: float in [0, 1]. Values below tau_min=0.65 indicate
                     anomalous behavior.
    """
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(sequence.astype(np.float32)).unsqueeze(0)  # (1, K, 5)
        score = model(x)
    return float(score.squeeze())
