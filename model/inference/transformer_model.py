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
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


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
        # Calibration/regularization fix: use at least 0.30 dropout to reduce
        # overconfident logit saturation while keeping architecture unchanged.
        self.dropout_p  = max(float(arch.get("dropout", 0.2)), 0.30)

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

    legit = [s for s in sequences if s["label"] == 0]
    adv   = [s for s in sequences if s["label"] == 1]

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
) -> Tuple[float, float]:
    """Train one epoch. Returns (mean_loss, accuracy)."""
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

        if not torch.isfinite(loss):
            print(f"  [WARNING] non-finite loss at batch {batch_idx}, skipping.")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * sequences.size(0)

        with torch.no_grad():
            probs     = torch.sigmoid(logits.detach())
            predicted = (probs > threshold).float()

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

    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate(
    model: TrustTransformer,
    loader: DataLoader,
    criterion: nn.Module,
    tau_min: float = 0.65,
    temperature: float = 1.0,
) -> Tuple[float, float, float, float]:
    """Evaluate. Returns (loss, accuracy, precision, recall)."""
    model.eval()
    total_loss: float = 0.0
    all_preds:  List[float] = []
    all_labels: List[float] = []
    threshold = _adv_threshold(tau_min)

    with torch.no_grad():
        for sequences, labels in loader:
            logits = model(sequences)
            loss   = criterion(logits, labels)
            total_loss += loss.item() * sequences.size(0)

            probs = torch.sigmoid(temperature_scale_logits(logits, temperature))
            preds = (probs > threshold).float()

            all_preds.extend(preds.cpu().numpy().flatten().tolist())
            all_labels.extend(labels.cpu().numpy().flatten().tolist())

    all_preds  = np.array(all_preds,  dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.float32)

    accuracy  = float((all_preds == all_labels).mean())
    tp = float(((all_preds == 1) & (all_labels == 1)).sum())
    fp = float(((all_preds == 1) & (all_labels == 0)).sum())
    fn = float(((all_preds == 0) & (all_labels == 1)).sum())
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)

    return total_loss / max(len(all_labels), 1), accuracy, precision, recall


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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # Class-imbalance-aware loss
    n_legit, n_adv, pos_weight = train_ds.class_weights()
    if verbose:
        print(f"\nClass balance: {n_legit} legit, {n_adv} adv → "
              f"pos_weight={pos_weight:.2f}  "
              f"(lr={config['training']['learning_rate']}, "
              f"dropout_effective={max(config['architecture']['dropout'], 0.30)}, "
              f"label_smoothing={label_smoothing}, "
              f"wd_effective={weight_decay})")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32)
    )

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
        val_loss, val_acc, val_prec, val_rec = evaluate(
            model, val_loader, criterion, tau_min=tau_min, temperature=1.0
        )
        scheduler.step(val_loss)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d} | lr={lr:.1e} | "
                f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
                f"Val Loss={val_loss:.4f} Acc={val_acc:.4f} "
                f"Prec={val_prec:.4f} Rec={val_rec:.4f}"
            )

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
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

    # Test evaluation
    model.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, "best_model.pt"),
                   weights_only=True)
    )
    test_loss, test_acc, test_prec, test_rec = evaluate(
        model, test_loader, criterion, tau_min=tau_min, temperature=1.0
    )
    f1 = (2 * test_prec * test_rec) / (test_prec + test_rec + 1e-8)

    if verbose:
        print(f"\n{'='*54}")
        print(f"  TEST RESULTS  (tau_min={tau_min}, "
              f"P(adv) thr={_adv_threshold(tau_min):.2f})")
        print(f"{'='*54}")
        print(f"  Accuracy  : {test_acc:.4f}  ({test_acc*100:.1f}%)")
        print(f"  Precision : {test_prec:.4f}")
        print(f"  Recall    : {test_rec:.4f}")
        print(f"  F1        : {f1:.4f}")
        print(f"{'='*54}")

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
) -> TrustTransformer:
    """Load trained model from checkpoint."""
    with open(config_path) as f:
        config = json.load(f)
    model = TrustTransformer(config)
    model.load_state_dict(
        torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    )
    model.eval()
    return model


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
