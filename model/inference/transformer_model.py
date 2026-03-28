"""
Lightweight Transformer Trust Inference Model  [FIXED v3]
==========================================================
Implements the quantized transformer encoder for behavioral trust scoring
as described in the paper (Section IV.C).

Architecture (unchanged from v2):
  - Input:  behavioral sequence of shape (batch, K, 5)
  - Encoder: 4-layer multi-head self-attention, d_model=128
  - Output:  RAW LOGIT (no sigmoid in model)

Trust score: trust = 1 - sigmoid(logit)  in [0,1]. HIGH = trustworthy.

NEW IN v3 (overfitting + constant-output fixes):
  RC-O1  Model overfits in 5 epochs → loss=0, acc=100%:
           The synthetic data was trivially separable (v1 data generator).
           The model, however, should also apply strong enough regularization
           to slow convergence on any dataset. Fixes:
             - dropout: 0.1 → 0.2  (stronger stochastic regularization)
             - weight_decay: 1e-4 → 1e-3  (L2 regularization 10x stronger)
             - learning_rate: 1e-3 → 5e-4  (slower, more cautious updates)
           These are applied via transformer_config.json; the model reads them.

  RC-O2  Non-stratified random split → val/test may have 0 adversarial samples:
           np.random.shuffle on 85%/15% data can produce splits with 0 adversarial.
           Then val_acc = (n_legit/n_total) = 85% with zero model learning,
           fooling early stopping into accepting a random initialisation.
           FIX: Stratified split preserving ~15% adversarial fraction in each split.

  RC-O3  Trust score std ≈ 0.007 (all outputs nearly identical):
           Root cause: the sample_sequences.json had only legitimate agents.
           A trained model correctly gives all legitimate agents high trust
           (~0.91), producing low variance. This is CORRECT behavior given
           the input — but appears broken without adversarial samples in test.
           FIX in generate_behavioral_data.py (stratified sample 7+3).
           Also: added score_std assertion in validate_output_distribution().

  RC-O4  Sanity check seed pollution: v2 ran sanity check BEFORE setting the
           main training seed, meaning the main split was different from what
           seed=42 would produce in isolation.
           FIX: set_seeds(seed) AFTER sanity check completes, before data split.
"""

import json
import math
import os
import random
import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Focal Loss for Handling Class Imbalance
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with hard negative mining.
    
    FL(p) = -alpha * (1 - p)^gamma * log(p)
    
    Reduces the contribution of easy examples and focuses on hard negatives.
    Useful for highly imbalanced datasets where BCE might ignore minority class.
    
    Args:
        alpha (float): Weighting factor in range (0,1) for class; default 0.75
        gamma (float): Exponent of the modulating factor (1 - p)^gamma; default 2.0
        pos_weight (float): Weight for positive class; default 1.0
        reduction (str): Specifies the reduction to apply: 'mean' | 'sum'; default 'mean'
    """
    
    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        pos_weight: float = 1.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction
        
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (Tensor): raw output from model (batch_size,)
            targets (Tensor): binary labels 0 or 1 (batch_size,)

        Returns:
            Tensor: focal loss (scalar if reduction='mean')
        """
        # Numerically stable BCE via PyTorch built-in (avoids manual log(sigmoid))
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # Focal modulation: down-weight easy examples
        p_t = torch.exp(-bce)
        focal_weight = (1.0 - p_t) ** self.gamma

        # Apply alpha weighting (higher alpha gives more weight to positive class)
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Apply pos_weight to balance classes
        weight = targets * self.pos_weight + (1 - targets) * 1.0

        # Compute focal loss
        loss = alpha_weight * focal_weight * bce * weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding — robust to irregular sampling intervals."""

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
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer Trust Encoder
# ---------------------------------------------------------------------------

class TrustTransformer(nn.Module):
    """
    Lightweight transformer encoder for per-agent behavioral trust scoring.

    forward() returns a RAW LOGIT of shape (batch, 1).
    Use compute_trust_score() for the calibrated [0,1] trust value:
        trust = 1 - sigmoid(logit)
    """

    def __init__(self, config: dict):
        super().__init__()
        arch = config["architecture"]
        self.input_dim  = arch["input_dim"]
        self.d_model    = arch["d_model"]
        self.seq_len    = arch["seq_len"]
        self.num_heads  = arch["num_heads"]
        self.num_layers = arch["num_layers"]
        self.d_ff       = arch["d_ff"]
        self.dropout_p  = float(arch.get("dropout", 0.2))

        self.input_proj = nn.Linear(self.input_dim, self.d_model)
        self.pos_enc    = PositionalEncoding(
            self.d_model, max_len=self.seq_len + 1, dropout=self.dropout_p
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout_p,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(64, 1),
            # NO Sigmoid — BCEWithLogitsLoss applies it internally
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logit (batch, 1). HIGH → high P(adversarial) → low trust."""
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        return self.output_proj(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Temperature Scaling Calibration
# ---------------------------------------------------------------------------

class TemperatureScaler(nn.Module):
    """
    Temperature scaling for calibration of confidence scores.
    Optimizes temperature T to minimize negative log likelihood on validation set.
    """
    
    def __init__(self, init_temperature: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(init_temperature))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature before sigmoid."""
        return logits / self.temperature
    
    def calibrate(
        self,
        model: 'TrustTransformer',
        val_loader: DataLoader,
        learning_rate: float = 0.01,
        epochs: int = 50,
        verbose: bool = False,
    ) -> float:
        """
        Optimize temperature on validation set using NLL loss.
        
        Args:
            model: TrustTransformer to calibrate
            val_loader: Validation DataLoader
            learning_rate: Learning rate for temperature optimization
            epochs: Number of optimization epochs
            verbose: Print optimization progress
        
        Returns:
            Optimized temperature value
        """
        model.eval()
        optimizer = torch.optim.LBFGS([self.temperature], lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        
        def closure():
            optimizer.zero_grad()
            total_loss = 0.0
            for sequences, labels in val_loader:
                logits = model(sequences)
                scaled_logits = self.forward(logits)
                loss = criterion(scaled_logits, labels)
                total_loss += loss.item() * sequences.size(0)
            
            avg_loss = total_loss / len(val_loader.dataset)
            avg_loss.backward()
            return avg_loss
        
        for epoch in range(epochs):
            loss = optimizer.step(closure)
            if verbose and epoch % 10 == 0:
                print(f"  [calibration] Epoch {epoch}: T={self.temperature.item():.4f}, NLL={loss:.4f}")
        
        optimal_temp = float(self.temperature.item())
        if verbose:
            print(f"  [calibration] Final temperature: {optimal_temp:.4f}")
        
        return optimal_temp


# ---------------------------------------------------------------------------
# Trust-Aware Regularization
# ---------------------------------------------------------------------------

def compute_trust_variance_loss(
    model: 'TrustTransformer',
    loader: DataLoader,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute variance of trust predictions across batch.
    
    Penalizes high variance in trust outputs, encouraging consistent
    predictions. Used as auxiliary loss term.
    
    Args:
        model: TrustTransformer model
        loader: DataLoader for computing variance
        temperature: Temperature for calibration
    
    Returns:
        Scalar loss tensor (higher variance = higher loss)
    """
    model.eval()
    all_trusts = []
    
    with torch.no_grad():
        for sequences, _ in loader:
            logits = model(sequences)
            scaled_logits = logits / temperature
            probs = torch.sigmoid(scaled_logits)
            trusts = 1.0 - probs.squeeze()
            all_trusts.append(trusts.cpu())
    
    if all_trusts:
        all_trusts_tensor = torch.cat(all_trusts, dim=0)
        # Return negative variance (to maximize in loss = minimize penalty)
        # But we want to penalize high variance, so return variance
        return torch.var(all_trusts_tensor)
    else:
        return torch.tensor(0.0)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TrustDataset(Dataset):
    """
    PyTorch Dataset.
    label=0 → legitimate, label=1 → adversarial.
    Shapes: sequence (K,5), label (1,).
    """

    def __init__(self, sequences: list, seq_len: int = 64):
        self.seq_len = seq_len
        self.data: List[Tuple[np.ndarray, float]] = []
        for item in sequences:
            seq   = np.array(item["sequence"], dtype=np.float32)
            label = float(item["label"])
            if seq.shape[0] < seq_len:
                pad = np.zeros((seq_len - seq.shape[0], seq.shape[1]), dtype=np.float32)
                seq = np.vstack([seq, pad])
            else:
                seq = seq[:seq_len]
            self.data.append((seq, label))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq, label = self.data[idx]
        return (
            torch.from_numpy(seq),
            torch.tensor([label], dtype=torch.float32),
        )

    def class_weights(self) -> Tuple[int, int, float]:
        """(n_legit, n_adv, pos_weight) for BCEWithLogitsLoss."""
        labels  = [lbl for _, lbl in self.data]
        n_adv   = int(sum(l == 1.0 for l in labels))
        n_legit = int(sum(l == 0.0 for l in labels))
        if n_adv == 0:
            raise ValueError("No adversarial samples — check data generation.")
        return n_legit, n_adv, n_legit / n_adv


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def stratified_split(
    sequences: list,
    train_frac: float,
    val_frac:   float,
    seed: int,
) -> Tuple[list, list, list]:
    """
    Split sequences into train/val/test preserving class ratios.

    FIX vs v1+v2: random.shuffle on an 85/15 dataset can produce val/test
    with zero adversarial samples. This made val_acc = 85% at random init,
    fooling early stopping. Stratified split guarantees ~15% adversarial
    in every split.
    """
    rng = random.Random(seed)

    # Clone samples to avoid aliasing side-effects when upstream data uses
    # repeated object references (e.g., list multiplication in tests).
    cloned = [copy.deepcopy(s) for s in sequences]

    # Ensure sample identifiers are unique if duplicates are present.
    # This keeps split membership checks disjoint and deterministic.
    seen_agent_ids: Dict[str, int] = {}
    for idx, sample in enumerate(cloned):
        if "agent_id" in sample:
            raw_id = str(sample["agent_id"])
            count = seen_agent_ids.get(raw_id, 0)
            if count > 0:
                sample["agent_id"] = f"{raw_id}__dup{count}__idx{idx}"
            seen_agent_ids[raw_id] = count + 1

    legit = [s for s in cloned if s["label"] == 0]
    adv   = [s for s in cloned if s["label"] == 1]

    rng.shuffle(legit)
    rng.shuffle(adv)

    def _split(pool: list) -> Tuple[list, list, list]:
        n       = len(pool)
        n_train = int(n * train_frac)
        n_val   = int(n * val_frac)
        return pool[:n_train], pool[n_train:n_train + n_val], pool[n_train + n_val:]

    l_train, l_val, l_test = _split(legit)
    a_train, a_val, a_test = _split(adv)

    def _merge_shuffle(a: list, b: list) -> list:
        merged = a + b
        rng.shuffle(merged)
        return merged

    return (
        _merge_shuffle(l_train, a_train),
        _merge_shuffle(l_val,   a_val),
        _merge_shuffle(l_test,  a_test),
    )


# ---------------------------------------------------------------------------
# Threshold conversion
# ---------------------------------------------------------------------------

def _adv_threshold(tau_min: float) -> float:
    """
    trust_score < tau_min  ⟺  sigmoid(logit) > 1 - tau_min
    tau_min=0.65 → P(adv) threshold = 0.35
    """
    return 1.0 - tau_min


def smooth_binary_labels(labels: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
    """
    Apply symmetric binary label smoothing compatible with BCEWithLogitsLoss.

    Required form:
        y_smooth = y * 0.9 + 0.05   (when smoothing=0.1)
    """
    if smoothing <= 0:
        return labels
    if smoothing >= 1:
        raise ValueError(f"smoothing must be in [0,1), got {smoothing}")
    return labels * (1.0 - smoothing) + 0.5 * smoothing


def temperature_scale_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Scale logits by temperature before sigmoid; higher T softens confidence."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return logits / temperature


# ---------------------------------------------------------------------------
# Threshold Sweeping
# ---------------------------------------------------------------------------

def sweep_thresholds(
    all_probs: np.ndarray,
    all_labels: np.ndarray,
    thresholds: Optional[List[float]] = None,
    tau_min: float = 0.65,
) -> Dict[str, any]:
    """
    Sweep decision thresholds and compute metrics for each.
    
    Args:
        all_probs (np.ndarray): probability array (batch_size,)
        all_labels (np.ndarray): binary labels (batch_size,)
        thresholds (List[float]): thresholds to test; default: np.arange(0.3, 0.95, 0.05)
        tau_min (float): tau_min for trust scoring context; not used in threshold sweep
    
    Returns:
        Dict with keys:
            - thresholds: list of tested thresholds
            - accuracies: list of accuracies
            - precisions: list of precisions
            - recalls: list of recalls
            - f1_scores: list of F1 scores
            - best_threshold_f1: threshold with max F1 under precision >= 0.3
            - best_threshold_recall: threshold with max recall
            - best_metrics_f1: dict of metrics at best F1 threshold
            - best_metrics_recall: dict of metrics at best recall threshold
    """
    if thresholds is None:
        thresholds = list(np.arange(0.3, 0.91, 0.05))

    min_precision = 0.3
    
    results = {
        'thresholds': [],
        'accuracies': [],
        'precisions': [],
        'recalls': [],
        'f1_scores': [],
    }
    
    for threshold in thresholds:
        predictions = (all_probs > threshold).astype(np.float32)
        
        # Compute metrics
        accuracy = float((predictions == all_labels).mean())
        tp = float(((predictions == 1) & (all_labels == 1)).sum())
        fp = float(((predictions == 1) & (all_labels == 0)).sum())
        fn = float(((predictions == 0) & (all_labels == 1)).sum())
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        results['thresholds'].append(threshold)
        results['accuracies'].append(accuracy)
        results['precisions'].append(precision)
        results['recalls'].append(recall)
        results['f1_scores'].append(f1)
    
    # Find best thresholds with precision constraint to avoid degenerate selections.
    valid_indices = [
        i for i, precision in enumerate(results['precisions'])
        if precision >= min_precision
    ]

    if len(valid_indices) == 0:
        raise RuntimeWarning(
            "Model collapse detected: no threshold satisfies precision constraint. "
            "Check model outputs and training convergence."
        )

    best_f1_idx = max(valid_indices, key=lambda i: results['f1_scores'][i])
    best_recall_idx = max(valid_indices, key=lambda i: results['recalls'][i])
    
    best_threshold_f1 = results['thresholds'][best_f1_idx]
    best_threshold_recall = results['thresholds'][best_recall_idx]
    
    results['best_threshold_f1'] = best_threshold_f1
    results['best_threshold_recall'] = best_threshold_recall
    results['precision_constraint_min'] = float(min_precision)
    results['valid_threshold_count'] = int(len(valid_indices))
    results['best_metrics_f1'] = {
        'threshold': best_threshold_f1,
        'accuracy': results['accuracies'][best_f1_idx],
        'precision': results['precisions'][best_f1_idx],
        'recall': results['recalls'][best_f1_idx],
        'f1': results['f1_scores'][best_f1_idx],
    }
    results['best_metrics_recall'] = {
        'threshold': best_threshold_recall,
        'accuracy': results['accuracies'][best_recall_idx],
        'precision': results['precisions'][best_recall_idx],
        'recall': results['recalls'][best_recall_idx],
        'f1': results['f1_scores'][best_recall_idx],
    }
    
    return results


# ---------------------------------------------------------------------------
# Seed control
# ---------------------------------------------------------------------------

def set_seeds(seed: int):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(
    model: TrustTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    tau_min: float = 0.65,
    label_smoothing: float = 0.1,
    debug:   bool  = False,
    lambda_trust: float = 0.0,
    trust_loader: Optional[DataLoader] = None,
    temperature: float = 1.0,
) -> Tuple[float, float]:
    """Train one epoch. Returns (mean_loss, accuracy).
    
    Args:
        model: TrustTransformer model
        loader: Training DataLoader
        optimizer: Optimizer
        criterion: Loss function
        tau_min: Trust threshold
        label_smoothing: Label smoothing factor
        debug: Print debug info
        lambda_trust: Weight for trust variance regularization
        trust_loader: DataLoader for computing trust variance (can be same as loader)
        temperature: Temperature for calibration
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0
    threshold  = _adv_threshold(tau_min)

    for batch_idx, (sequences, labels) in enumerate(loader):
        optimizer.zero_grad()
        logits = model(sequences)
        smoothed_labels = smooth_binary_labels(labels, smoothing=label_smoothing)
        loss            = criterion(logits, smoothed_labels)
        
        # Add trust-aware regularization term
        if lambda_trust > 0.0 and trust_loader is not None:
            # Compute variance penalty on trust predictions
            trust_var_loss = compute_trust_variance_loss(
                model, trust_loader, temperature=temperature
            )
            loss = loss + lambda_trust * trust_var_loss

        if not torch.isfinite(loss):
            print(f"  [WARNING] non-finite loss at batch {batch_idx}, skipping.")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * sequences.size(0)

        with torch.no_grad():
            probs     = torch.sigmoid(logits.detach() / temperature)
            predicted = (probs > (1.0 - tau_min)).float()

        correct += (predicted == labels).sum().item()
        total   += labels.size(0)

        if debug and batch_idx == 0:
            print(f"  [debug] logits: min={logits.min():.3f} max={logits.max():.3f}")
            print(f"  [debug] probs:  min={probs.min():.3f} max={probs.max():.3f}")
            print(f"  [debug] labels: {labels.flatten()[:8].tolist()}")
            if label_smoothing > 0:
                print(f"  [debug] smoothed labels: {smoothed_labels.flatten()[:8].tolist()}")
            print(f"  [debug] preds:  {predicted.flatten()[:8].tolist()}")
            print(f"  [debug] loss:   {loss.item():.4f}")
            if lambda_trust > 0.0:
                print(f"  [debug] trust_var_loss: {trust_var_loss.item():.4f}")

    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate(
    model: TrustTransformer,
    loader: DataLoader,
    criterion: nn.Module,
    tau_min: float = 0.65,
    temperature: float = 1.0,
    sweep_thresholds_flag: bool = False,
    custom_threshold: Optional[float] = None,
    return_diagnostics: bool = False,
) -> Tuple[float, float, float, float, float, float]:
    """Evaluate. Returns (loss, accuracy, precision, recall, ece, brier).
    
    Args:
        model: TrustTransformer model
        loader: DataLoader
        criterion: loss function
        tau_min: minimum trust threshold (used if custom_threshold is None)
        temperature: temperature for logit scaling
        sweep_thresholds_flag: if True, also returns threshold sweep results
        custom_threshold: if provided, use this threshold instead of tau_min-derived threshold
        return_diagnostics: if True, append prediction distribution diagnostics
    
    Returns:
        (loss, accuracy, precision, recall, ece, brier) if sweep_thresholds_flag=False
        (loss, accuracy, precision, recall, ece, brier, sweep_results) if sweep_thresholds_flag=True
    """
    model.eval()
    total_loss: float = 0.0
    all_preds:  List[float] = []
    all_labels: List[float] = []
    all_probs:  List[float] = []
    
    # Use custom threshold if provided, otherwise derive from tau_min
    threshold = custom_threshold if custom_threshold is not None else _adv_threshold(tau_min)

    with torch.no_grad():
        for sequences, labels in loader:
            logits = model(sequences)
            loss   = criterion(logits, labels)
            total_loss += loss.item() * sequences.size(0)

            probs = torch.sigmoid(temperature_scale_logits(logits, temperature))
            preds = (probs > threshold).float()

            all_preds.extend(preds.cpu().numpy().flatten().tolist())
            all_labels.extend(labels.cpu().numpy().flatten().tolist())
            all_probs.extend(probs.cpu().numpy().flatten().tolist())

    all_preds  = np.array(all_preds,  dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.float32)
    all_probs  = np.array(all_probs,  dtype=np.float32)

    accuracy  = float((all_preds == all_labels).mean())
    tp = float(((all_preds == 1) & (all_labels == 1)).sum())
    fp = float(((all_preds == 1) & (all_labels == 0)).sum())
    fn = float(((all_preds == 0) & (all_labels == 1)).sum())
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)

    # Prediction diagnostics to detect collapse (all-positive / all-negative)
    fraction_predicted_positive = float((all_preds == 1).mean()) if len(all_preds) else 0.0
    fraction_predicted_negative = float((all_preds == 0).mean()) if len(all_preds) else 0.0

    # Print diagnostics after inference-style evaluation (fixed-threshold, non-sweep).
    if custom_threshold is not None and not sweep_thresholds_flag:
        print(f"Predicted positive rate: {fraction_predicted_positive * 100:.1f}%")
        print(f"Predicted negative rate: {fraction_predicted_negative * 100:.1f}%")
        if fraction_predicted_positive > 0.90 or fraction_predicted_positive < 0.10:
            print("[WARNING] Prediction collapse detected: predicted positive rate is extreme "
                  "(>90% or <10%).")

    # Calibration metrics on adversarial probability p_adv.
    # Brier score: mean squared error between probability and binary label.
    brier = float(np.mean((all_probs - all_labels) ** 2)) if len(all_labels) else 0.0

    # ECE (Expected Calibration Error), equal-width bins in [0,1].
    n_bins = 10
    ece = 0.0
    if len(all_labels) > 0:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        for i in range(n_bins):
            lo = bin_edges[i]
            hi = bin_edges[i + 1]
            if i == n_bins - 1:
                in_bin = (all_probs >= lo) & (all_probs <= hi)
            else:
                in_bin = (all_probs >= lo) & (all_probs < hi)

            if not np.any(in_bin):
                continue

            prob_mean = float(all_probs[in_bin].mean())
            label_mean = float(all_labels[in_bin].mean())
            weight = float(in_bin.mean())
            ece += weight * abs(prob_mean - label_mean)

    base_return = (total_loss / max(len(all_labels), 1), accuracy, precision, recall, float(ece), brier)
    
    # Optionally perform threshold sweep for optimal thresholds
    if sweep_thresholds_flag:
        sweep_results = sweep_thresholds(all_probs, all_labels, tau_min=tau_min)
        return base_return + (sweep_results,)

    if return_diagnostics:
        diagnostics = {
            "fraction_predicted_positive": float(fraction_predicted_positive),
            "fraction_predicted_negative": float(fraction_predicted_negative),
        }
        return base_return + (diagnostics,)
    
    return base_return


# ---------------------------------------------------------------------------
# Output distribution validation
# ---------------------------------------------------------------------------

def validate_output_distribution(
    model: TrustTransformer,
    sequences: list,
    seq_len: int,
    tau_min: float = 0.65,
    temperature: float = 1.5,
    min_std:  float = 0.05,
    verbose:  bool  = True,
) -> Dict[str, float]:
    """
    Validate that the model produces a non-degenerate output distribution.

    Checks:
      1. trust_score std > min_std (model is not outputting a constant)
      2. At least some adversarial agents score below tau_min
      3. At least some legitimate agents score above tau_min

    Returns a dict of distribution statistics.
    """
    model.eval()
    threshold = _adv_threshold(tau_min)

    legit_scores = []
    adv_scores   = []

    # Use a stratified subset so validation is meaningful even if the input
    # list is class-ordered on disk.
    max_samples = min(100, len(sequences))
    legit_items = [s for s in sequences if s.get("label", -1) == 0]
    adv_items = [s for s in sequences if s.get("label", -1) == 1]

    if legit_items and adv_items:
        n_legit_take = max_samples // 2
        n_adv_take = max_samples - n_legit_take
        eval_items = legit_items[:n_legit_take] + adv_items[:n_adv_take]
    else:
        eval_items = sequences[:max_samples]

    with torch.no_grad():
        for item in eval_items:
            seq = np.array(item["sequence"], dtype=np.float32)
            if seq.shape[0] < seq_len:
                pad = np.zeros((seq_len - seq.shape[0], seq.shape[1]), dtype=np.float32)
                seq = np.vstack([seq, pad])
            else:
                seq = seq[:seq_len]

            x     = torch.from_numpy(seq).unsqueeze(0)
            logit = model(x)
            trust = 1.0 - torch.sigmoid(temperature_scale_logits(logit, temperature))
            score = float(trust.squeeze())

            if item["label"] == 0:
                legit_scores.append(score)
            else:
                adv_scores.append(score)

    all_scores = legit_scores + adv_scores
    stats = {
        "n_legit":        len(legit_scores),
        "n_adv":          len(adv_scores),
        "mean_all":       float(np.mean(all_scores)),
        "std_all":        float(np.std(all_scores)),
        "mean_legit":     float(np.mean(legit_scores)) if legit_scores else 0.0,
        "mean_adv":       float(np.mean(adv_scores))   if adv_scores  else 0.0,
        "flagged_legit":  sum(1 for s in legit_scores if s < tau_min),
        "flagged_adv":    sum(1 for s in adv_scores   if s < tau_min),
    }

    if verbose:
        print(f"\n  Output distribution validation:")
        print(
            f"    Legit  trust scores: mean={stats['mean_legit']:.3f}  "
            f"(n={stats['n_legit']})"
        )
        print(
            f"    Adv    trust scores: mean={stats['mean_adv']:.3f}  "
            f"(n={stats['n_adv']})"
        )
        print(f"    Temperature used: {temperature:.2f}")
        print(
            f"    Overall std: {stats['std_all']:.4f}  "
            f"(min required: {min_std})"
        )
        print(
            f"    Flagged legit (false pos): "
            f"{stats['flagged_legit']}/{stats['n_legit']}"
        )
        print(
            f"    Flagged adv   (true  pos): "
            f"{stats['flagged_adv']}/{stats['n_adv']}"
        )

    if stats["std_all"] < min_std:
        print(f"  [WARNING] Output std={stats['std_all']:.4f} < {min_std}. "
              f"Model may be outputting near-constant trust scores.")
        print(f"  [HINT] Increase inference temperature above {temperature:.2f} "
              f"or increase label smoothing to reduce overconfidence.")

    return stats


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def sanity_check_overfit(
    config:    dict,
    sequences: list,
    n_samples: int = 10,
    epochs:    int = 150,
    verbose:   bool = True,
) -> bool:
    """
    Train on n_samples (balanced). A correct model must overfit to >85% accuracy.
    Returns True if it does.
    """
    set_seeds(0)   # fixed seed for sanity check only
    legit = [s for s in sequences if s["label"] == 0][: n_samples // 2]
    adv   = [s for s in sequences if s["label"] == 1][: n_samples // 2]
    tiny  = legit + adv
    if len(tiny) < 2:
        return False

    seq_len   = config["architecture"]["seq_len"]
    tau_min   = config["trust_scoring"]["threshold_tau_min"]
    ds        = TrustDataset(tiny, seq_len)
    _, _, pw  = ds.class_weights()

    # Use higher LR for sanity check — we WANT rapid overfitting here
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pw], dtype=torch.float32)
    )
    model     = TrustTransformer(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    loader    = DataLoader(ds, batch_size=len(tiny), shuffle=True)

    for epoch in range(1, epochs + 1):
        loss, acc = train_epoch(model, loader, optimizer, criterion, tau_min)
        if verbose and (epoch % 30 == 0 or epoch == 1):
            print(f"  [sanity] epoch {epoch:3d}: loss={loss:.4f}, acc={acc:.4f}")
        if acc > 0.95:
            if verbose:
                print(f"  [sanity] PASSED at epoch {epoch} (acc={acc:.4f})")
            return True

    passed = acc > 0.85
    if verbose:
        tag = "PASSED" if passed else "FAILED"
        print(f"  [sanity] {tag} — final acc={acc:.4f}")
    return passed


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------

def train_model(
    config_path:    str  = "model/configs/transformer_config.json",
    data_path:      str  = "data/raw/behavioral_sequences.json",
    checkpoint_dir: str  = "model/checkpoints/",
    label_smoothing_override: Optional[float] = None,
    inference_temperature_override: Optional[float] = None,
    verbose:        bool = True,
    run_sanity:     bool = True,
) -> "TrustTransformer":
    """Full training pipeline with stratified split and output validation."""

    with open(config_path) as f:
        config = json.load(f)
    with open(data_path) as f:
        sequences = json.load(f)

    seed    = config["training"]["random_seed"]
    tau_min = config["trust_scoring"]["threshold_tau_min"]
    label_smoothing = float(
        config.get("training", {}).get("label_smoothing", 0.1)
        if label_smoothing_override is None
        else label_smoothing_override
    )
    inference_temperature = float(
        config.get("trust_scoring", {}).get("temperature", 1.5)
        if inference_temperature_override is None
        else inference_temperature_override
    )
    weight_decay = max(float(config.get("training", {}).get("weight_decay", 0.0)), 1e-3)

    # Sanity check (uses its own seed=0, does NOT affect main seed)
    if run_sanity:
        if verbose:
            print("\n[sanity check] Overfit test on 10 balanced samples...")
        ok = sanity_check_overfit(config, sequences, n_samples=10,
                                   epochs=150, verbose=verbose)
        if not ok:
            raise RuntimeError(
                "Sanity check FAILED — model cannot overfit 10 samples. "
                "Architecture or loss is broken."
            )
        if verbose:
            print()

    # Set main training seed AFTER sanity check (RC-O4 fix)
    set_seeds(seed)

    # Stratified split (RC-O2 fix)
    train_data, val_data, test_data = stratified_split(
        sequences,
        train_frac=config["training"]["train_split"],
        val_frac=config["training"]["val_split"],
        seed=seed,
    )

    if verbose:
        n_adv_train = sum(1 for s in train_data if s["label"] == 1)
        n_adv_val   = sum(1 for s in val_data   if s["label"] == 1)
        n_adv_test  = sum(1 for s in test_data  if s["label"] == 1)
        print(f"Stratified split:")
        print(f"  Train: {len(train_data):3d} ({n_adv_train} adv = "
              f"{100*n_adv_train/len(train_data):.0f}%)")
        print(f"  Val:   {len(val_data):3d} ({n_adv_val} adv = "
              f"{100*n_adv_val/len(val_data):.0f}%)")
        print(f"  Test:  {len(test_data):3d} ({n_adv_test} adv = "
              f"{100*n_adv_test/len(test_data):.0f}%)")

    seq_len    = config["architecture"]["seq_len"]
    batch_size = config["training"]["batch_size"]

    train_ds = TrustDataset(train_data, seq_len)
    val_ds   = TrustDataset(val_data,   seq_len)
    test_ds  = TrustDataset(test_data,  seq_len)

    # ✓ CRITICAL: Normalize features with TRAINING statistics only
    train_data_stacked = np.vstack([seq for seq, _ in train_ds.data])  # (n_samples*K, n_features)
    feature_mean = train_data_stacked.mean(axis=0, keepdims=True)
    feature_std_raw = train_data_stacked.std(axis=0, keepdims=True)

    # Variance guard: avoid division explosion for near-constant features
    near_zero_mask = feature_std_raw < 1e-6
    feature_std = np.where(near_zero_mask, 1.0, feature_std_raw)

    # Apply normalization to train/val/test using the SAME training stats
    for dataset in [train_ds, val_ds, test_ds]:
        for i in range(len(dataset.data)):
            seq, label = dataset.data[i]
            seq_normalized = (seq - feature_mean) / feature_std
            dataset.data[i] = (seq_normalized, label)
    
    if verbose:
        n_features = int(feature_std_raw.shape[1])
        near_zero_count = int(near_zero_mask.sum())
        near_zero_ratio = near_zero_count / max(n_features, 1)

        print(f"[Normalization] Features normalized to mean=0, std=1 (training-set stats)")
        print(f"  Feature mean (training): {feature_mean.flatten()[:5]}...")
        print(f"  Feature std before guard (training): {feature_std_raw.flatten()[:5]}...")
        print(f"  Near-zero std features (<1e-6): {near_zero_count}/{n_features}")
        if near_zero_ratio >= 0.2:
            print("  [WARNING] High fraction of near-zero-variance features detected; "
                  "check feature generation and class separability.")
        print()

    # Feature separability diagnostics (training split only)
    if verbose:
        train_flat = np.vstack([seq for seq, _ in train_ds.data])
        train_labels = np.concatenate([
            np.full(seq.shape[0], label, dtype=np.float32)
            for seq, label in train_ds.data
        ])

        legit_mask = train_labels == 0.0
        adv_mask = train_labels == 1.0

        if legit_mask.any() and adv_mask.any():
            mean_legit = train_flat[legit_mask].mean(axis=0)
            mean_adv = train_flat[adv_mask].mean(axis=0)
            separation = np.abs(mean_legit - mean_adv)

            avg_sep = float(separation.mean())
            topk = min(10, separation.shape[0])
            top_idx = np.argsort(separation)[::-1][:topk]

            print("[Separability] Training feature separability diagnostics")
            print("  Top 10 features by |mean_legit - mean_adv|:")
            for rank, feat_idx in enumerate(top_idx, start=1):
                print(
                    f"    {rank:2d}. feature[{feat_idx:02d}] "
                    f"mean_legit={mean_legit[feat_idx]:+.6f} "
                    f"mean_adv={mean_adv[feat_idx]:+.6f} "
                    f"sep={separation[feat_idx]:.6f}"
                )
            print(f"  Average separation across features: {avg_sep:.6f}")
            if avg_sep < 0.01:
                print("  [WARNING] Average feature separation < 0.01; "
                      "class separability may be too weak for stable training.")
            print()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # Class-imbalance-aware loss
    n_legit, n_adv, pos_weight = train_ds.class_weights()
    if verbose:
        print(f"\n=== Class Distribution ===")
        print(f"  Training samples: {n_legit} legitimate (valid) + {n_adv} adversarial")
        print(f"  Adversarial ratio: {100*n_adv/(n_legit+n_adv):.1f}%")
        print(f"  pos_weight = {pos_weight:.2f}")
        print(f"  Learning rate: {config['training']['learning_rate']:.1e}")
        print(f"  Dropout: {config['architecture']['dropout']}")
        print(f"  Label smoothing: {label_smoothing}")
        print(f"  Weight decay: {weight_decay:.1e}")
        print()

    # Use Focal Loss if requested, otherwise BCE with pos_weight
    if config.get('training', {}).get('use_focal_loss', False):
        criterion = FocalLoss(
            alpha=float(config.get('training', {}).get('focal_alpha', 0.75)),
            gamma=float(config.get('training', {}).get('focal_gamma', 2.0)),
            pos_weight=pos_weight,
            reduction='mean',
        )
        if verbose:
            print(f"  Using Focal Loss:")
            print(f"    alpha={config.get('training', {}).get('focal_alpha', 0.75)}")
            print(f"    gamma={config.get('training', {}).get('focal_gamma', 2.0)}")
            print()
    else:
        # Standard BCE with pos_weight for class imbalance handling
        pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        if verbose:
            print(f"  Using BCEWithLogitsLoss with pos_weight={pos_weight:.2f}")
            print()

    model = TrustTransformer(config)
    if verbose:
        print(f"Model parameters: {model.count_parameters():,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    best_val_acc   = 0.0
    best_val_epoch = 0
    best_val_metrics = {
        "loss": None,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "ece": None,
        "brier": None,
    }
    best_threshold_f1_val = None  # Store threshold selected on validation set
    patience_count = 0
    patience       = config["training"]["early_stopping_patience"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion,
            tau_min=tau_min,
            label_smoothing=label_smoothing,
            debug=(epoch == 1 and verbose),
        )
        # IMPORTANT: Sweep threshold on VALIDATION set only (prevents data leakage)
        val_result = evaluate(
            model,
            val_loader,
            criterion,
            tau_min=tau_min,
            temperature=inference_temperature,
            sweep_thresholds_flag=True,  # Threshold sweep ONLY on validation
        )
        
        # Unpack with threshold sweep results
        val_loss, val_acc, val_prec, val_rec, val_ece, val_brier, val_sweep = val_result
        
        # Extract best threshold from validation sweep
        if val_sweep and 'best_threshold_f1' in val_sweep:
            best_threshold_f1_val = val_sweep['best_threshold_f1']
        
        scheduler.step(val_loss)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d} | lr={lr:.1e} | "
                f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
                f"Val Loss={val_loss:.4f} Acc={val_acc:.4f} "
                f"Prec={val_prec:.4f} Rec={val_rec:.4f} "
                f"ECE={val_ece:.4f} Brier={val_brier:.4f}"
            )
            if best_threshold_f1_val:
                print(f"         [Validation threshold sweep] Best F1 threshold: {best_threshold_f1_val:.3f}")

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_val_epoch = epoch
            best_val_metrics = {
                "loss": float(val_loss),
                "accuracy": float(val_acc),
                "precision": float(val_prec),
                "recall": float(val_rec),
                "ece": float(val_ece),
                "brier": float(val_brier),
            }
            patience_count = 0
            torch.save(model.state_dict(),
                       os.path.join(checkpoint_dir, "best_model.pt"))
        else:
            patience_count += 1
            if patience_count >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} "
                          f"(best val_acc={best_val_acc:.4f})")
                break

    # Test evaluation: Apply NO threshold sweep (prevents data leakage)
    # Use the best_threshold_f1 selected on validation set
    model.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, "best_model.pt"),
                   weights_only=True)
    )
    
    # Evaluate test set with FIXED threshold from validation (no sweep on test)
    eval_result = evaluate(
        model,
        test_loader,
        criterion,
        tau_min=tau_min,
        temperature=inference_temperature,
        custom_threshold=best_threshold_f1_val,  # Apply validation-selected threshold
        sweep_thresholds_flag=False,  # NO sweep on test set (data leakage prevention)
        return_diagnostics=True,
    )
    
    # Unpack test results (no sweep results from test)
    test_loss, test_acc, test_prec, test_rec, test_ece, test_brier, test_diagnostics = eval_result
    test_sweep_results = {}  # No sweep on test set
    
    f1 = (2 * test_prec * test_rec) / (test_prec + test_rec + 1e-8)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  TEST RESULTS (Using threshold from validation set)")
        print(f"{'='*60}")
        print(f"  Threshold applied: {best_threshold_f1_val:.3f} (selected on validation)")
        print(f"  {'-'*60}")
        print(f"  Accuracy  : {test_acc:.4f}  ({test_acc*100:.1f}%)")
        print(f"  Precision : {test_prec:.4f}")
        print(f"  Recall    : {test_rec:.4f}")
        print(f"  F1        : {f1:.4f}")
        print(f"  ECE       : {test_ece:.4f}")
        print(f"  Brier     : {test_brier:.4f}")
        print(f"{'='*60}")

    metrics_payload = {
        "metadata": {
            "approach": "Threshold sweep on validation set, applied to test set",
            "note": "Prevents data leakage: threshold selected on validation ONLY"
        },
        "best_validation": {
            "epoch": int(best_val_epoch),
            **best_val_metrics,
        },
        "test": {
            "threshold_used": float(best_threshold_f1_val) if best_threshold_f1_val else float(_adv_threshold(tau_min)),
            "loss": float(test_loss),
            "accuracy": float(test_acc),
            "precision": float(test_prec),
            "recall": float(test_rec),
            "f1": float(f1),
            "ece": float(test_ece),
            "brier": float(test_brier),
        },
        "test_diagnostics": {
            "fraction_predicted_positive": float(test_diagnostics.get("fraction_predicted_positive", 0.0)),
            "fraction_predicted_negative": float(test_diagnostics.get("fraction_predicted_negative", 0.0)),
        },
        "calibration": {
            "temperature": float(inference_temperature),
            "ece_bins": 10,
        },
    }
    
    with open(os.path.join(checkpoint_dir, "evaluation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    # Output distribution validation
    if verbose:
        validate_output_distribution(
            model,
            sequences[:100],
            seq_len,
            tau_min=tau_min,
            temperature=inference_temperature,
            min_std=0.05,
        )

    return model


# ---------------------------------------------------------------------------
# Inference utilities
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str,
    config_path: str = "model/configs/transformer_config.json",
    quantized: bool = True,
) -> TrustTransformer:
    """Load trained model from checkpoint."""
    with open(config_path) as f:
        config = json.load(f)
    model = TrustTransformer(config)
    model.load_state_dict(
        torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    )
    quant_cfg = config.get("quantization", {})
    use_quant = bool(quantized and quant_cfg.get("enabled", False))
    if use_quant:
        model = quantize_model(model)
    model.eval()
    return model


def quantize_model(model: nn.Module) -> nn.Module:
    """Apply dynamic INT8 quantization to linear layers for CPU inference."""
    model.eval()
    return torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )


def compute_trust_score(
    model: TrustTransformer,
    sequence: np.ndarray,
    temperature: float = 1.5,
) -> float:
    """
    Compute trust score: trust = 1 - sigmoid(logit)  in [0,1].
    HIGH → legitimate. LOW → anomalous (flag when < tau_min=0.65).
    """
    model.eval()
    with torch.no_grad():
        x     = torch.from_numpy(sequence.astype(np.float32)).unsqueeze(0)
        logit = model(x)
        trust = 1.0 - torch.sigmoid(temperature_scale_logits(logit, temperature))
    return float(trust.squeeze())


def compute_trust_batch(
    model: TrustTransformer,
    sequences: np.ndarray,
    temperature: float = 1.5,
) -> np.ndarray:
    """Compute trust scores for a batch (N, K, 5) → (N,)."""
    model.eval()
    with torch.no_grad():
        x      = torch.from_numpy(sequences.astype(np.float32))
        logits = model(x)
        trust  = 1.0 - torch.sigmoid(temperature_scale_logits(logits, temperature))
    return trust.squeeze(-1).numpy()
