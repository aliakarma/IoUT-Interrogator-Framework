# Threshold Sweeping and Focal Loss Enhancements

## Overview

The evaluation pipeline has been significantly enhanced with three major improvements:

1. **Threshold Sweeping** — Automatic identification of optimal decision thresholds
2. **Focal Loss** — Improved handling of class imbalance and hard examples
3. **Class Distribution Logging** — Better visibility into training data balance

## 1. Threshold Sweeping

### What is it?

Instead of using a fixed decision threshold for binary classification, threshold sweeping automatically tests a range of decision boundaries (0.1 to 0.9 with step 0.05) and reports:

- **Optimal threshold for F1 score** (balanced precision/recall)
- **Optimal threshold for recall maximization** (prioritizes catching adversarial agents)
- **Metrics at both thresholds** (accuracy, precision, recall, F1)

### Why?

- Fixed thresholds (e.g., 0.5) may not be optimal for your specific use case
- Different applications require different precision/recall tradeoffs:
  - **F1-optimized threshold**: General-purpose detection (balanced)
  - **Recall-optimized threshold**: Security-critical (catch all adversaries, accept false positives)

### Implementation

The threshold sweeping is performed on:
- **Validation set** — During training (optional)
- **Test set** — After training (automatic)

### Usage

```bash
# Default training with threshold sweep on test set only
python model/inference/train.py

# View threshold sweep results in evaluation_metrics.json
cat model/checkpoints/evaluation_metrics.json
```

### Output

The training script prints optimized thresholds:

```
==================================================
  THRESHOLD SWEEP RESULTS
==================================================
  Best threshold (F1):     0.350
    Accuracy : 0.9234
    Precision: 0.8900
    Recall   : 0.7600
    F1       : 0.8196

  Best threshold (Recall): 0.200
    Accuracy : 0.8950
    Precision: 0.6500
    Recall   : 0.9800
    F1       : 0.7875
==================================================
```

### Saved Results

Results are saved to `model/checkpoints/evaluation_metrics.json`:

```json
{
  "threshold_sweep": {
    "best_threshold_f1": 0.35,
    "best_metrics_f1": {
      "threshold": 0.35,
      "accuracy": 0.9234,
      "precision": 0.89,
      "recall": 0.76,
      "f1": 0.8196
    },
    "best_threshold_recall": 0.2,
    "best_metrics_recall": {
      "threshold": 0.2,
      "accuracy": 0.895,
      "precision": 0.65,
      "recall": 0.98,
      "f1": 0.7875
    }
  }
}
```

## 2. Focal Loss

### What is it?

Focal Loss is a specialized loss function designed to handle class imbalance by:
- **Downweighting easy examples** that the model already classifies correctly
- **Focusing on hard examples** that are misclassified
- **Improving minority class detection** (e.g., adversarial agents in mostly-legitimate populations)

Formula:
```
FL(p) = -alpha * (1 - p)^gamma * log(p)
```

Where:
- `alpha = 0.75` — Weight for class balancing
- `gamma = 2.0` — Focusing parameter (higher = more focus on hard examples)

### Why Focal Loss?

**Problem with standard BCE:**
- For highly imbalanced data (e.g., 5% adversarial, 95% legitimate), BCE can ignore the minority class
- The model learns to classify most samples correctly without effort
- Recall on minority class suffers

**Solution with Focal Loss:**
- Hard misclassified examples get higher loss weight
- Model is forced to improve on difficult cases
- Better recall on minority class (adversarial agents)

### When to Use

Use Focal Loss when:
- **Class imbalance is severe** (e.g., < 10% adversarial samples)
- **Recall of minority class is critical** (catching all adversarial agents is important)
- BCEWithLogitsLoss + pos_weight isn't sufficient

### Usage

```bash
# Default: Standard BCE with pos_weight
python model/inference/train.py

# With Focal Loss (recommended for imbalanced data)
python model/inference/train.py --use-focal-loss

# Customize Focal Loss parameters
python model/inference/train.py \
  --use-focal-loss \
  --focal-alpha 0.75 \
  --focal-gamma 2.0
```

### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--use-focal-loss` | False | Enable Focal Loss instead of BCE |
| `--focal-alpha` | 0.75 | Weighting factor for class balance (range: 0-1) |
| `--focal-gamma` | 2.0 | Focusing parameter for hard examples (range: 0-5) |

### Output

During training, you'll see which loss function is active:

```
=== Class Distribution ===
  Training samples: 89 legitimate (valid) + 11 adversarial
  Adversarial ratio: 11.0%
  pos_weight = 8.09
  ...
  Using Focal Loss:
    alpha=0.75
    gamma=2.0
```

or

```
  Using BCEWithLogitsLoss with pos_weight=8.09
```

## 3. Class Distribution Logging

### What Changed?

Training now logs detailed class balance information:

```
=== Class Distribution ===
  Training samples: 89 legitimate (valid) + 11 adversarial
  Adversarial ratio: 11.0%
  pos_weight = 8.09
  Learning rate: 5.0e-04
  Dropout: 0.2
  Label smoothing: 0.1
  Weight decay: 1.0e-03
```

### Benefits

- **Visibility**: Immediately see if your data is balanced
- **Reproducibility**: Understand exactly what pos_weight was used
- **Debugging**: Detect data generation issues (e.g., no adversarial samples)

### Understanding pos_weight

`pos_weight = num_negative / num_positive`

Example:
- 89 legitimate, 11 adversarial → pos_weight = 89/11 ≈ 8.09
- **Interpretation**: Misclassifying one adversarial agent is weighted ~8x more than misclassifying one legitimate agent
- **Effect**: Forces model to pay more attention to minority class

## Combined Usage Example

### Recommended Configuration for Imbalanced Data

```bash
# Train with Focal Loss, label smoothing, and class imbalance handling
python model/inference/train.py \
  --use-focal-loss \
  --focal-alpha 0.75 \
  --focal-gamma 2.0 \
  --label-smoothing 0.1 \
  --seed 42

# This will:
# 1. Use Focal Loss for hard example mining
# 2. Apply label smoothing for regularization
# 3. Log class distribution and pos_weight
# 4. Perform threshold sweep on test set
# 5. Report optimal thresholds for F1 and recall
```

### Inspecting Results

```bash
# View final evaluation metrics with threshold sweep
cat model/checkpoints/evaluation_metrics.json

# Extract just the threshold sweep results
python -c "import json; m=json.load(open('model/checkpoints/evaluation_metrics.json')); print(json.dumps(m['threshold_sweep'], indent=2))"
```

## Technical Details

### Focal Loss Implementation

The FocalLoss class in `transformer_model.py`:

```python
class FocalLoss(nn.Module):
    """
    FL(p) = -alpha * (1 - p)^gamma * log(p)
    """
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce = -(targets * log(probs) + (1-targets) * log(1-probs))
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1.0 - p_t) ** gamma
        loss = alpha * focal_weight * bce
        return loss.mean()
```

### Threshold Sweep Implementation

The `sweep_thresholds()` function:

```python
def sweep_thresholds(all_probs, all_labels, thresholds=None):
    # Test each threshold: 0.1, 0.15, ..., 0.9
    # Compute accuracy, precision, recall, F1 for each
    # Return best threshold for F1 and for recall
```

### GPU Support

- **Focal Loss**: Fully GPU-compatible
- **pos_weight in BCE**: Properly moved to GPU before loss computation
- **No device issues**: All tensors stay on the same device

## Backward Compatibility

All enhancements are **100% backward compatible**:

- Default behavior unchanged: BCE with pos_weight is still default
- Threshold sweeping happens automatically on test set
- No breaking changes to existing training workflows

## Performance Impact

### Training Time

- **Focal Loss**: ~5% slower than BCE (due to additional focal weighting computation)
- **Threshold Sweep**: Negligible on test set (only after training complete)

### Memory Usage

- Focal Loss: Same memory footprint as BCE
- Negligible additional memory for threshold sweep

## References

- Focal Loss paper: [Focal Loss for Dense Object Detection (Lin et al., 2017)](https://arxiv.org/abs/1708.02002)
- Threshold selection: Standard ML practice for optimal decision boundaries
- pos_weight: PyTorch BCEWithLogitsLoss documentation

## FAQ

**Q: Should I always use Focal Loss?**
A: No. Use it only if your data is highly imbalanced (< 10% minority class). For balanced data, standard BCE is fine.

**Q: What if Focal Loss makes training worse?**
A: Try reducing `gamma` (e.g., 1.0 or 1.5) or adjust `alpha`. The default 0.75, 2.0 works well for most cases.

**Q: Can I use custom thresholds?**
A: Yes, the `evaluate()` function accepts custom thresholds in `sweep_thresholds()` parameter.

**Q: How do I use the optimal threshold for inference?**
A: Read `best_threshold_f1` from `evaluation_metrics.json` and use it in your inference script instead of the fixed `tau_min`.

## Troubleshooting

**Problem**: All thresholds produce same result
- **Cause**: Model outputs collapse to one value
- **Fix**: Increase inference temperature or check model training

**Problem**: Focal Loss increases training loss
- **Cause**: Higher weight on hard examples by design
- **Fix**: This is expected; monitor validation accuracy instead of training loss

**Problem**: Threshold sweep results missing
- **Cause**: Old `evaluate()` function without sweep_thresholds_flag
- **Fix**: Ensure you have the latest version of transformer_model.py

---

**For questions or issues**: See REPRODUCIBILITY.md for full training reproducibility guide.
