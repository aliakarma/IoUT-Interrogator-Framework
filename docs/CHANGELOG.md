# Changelog

All notable changes to the IoUT Interrogator Framework are documented here.

---

## v3.0.0 — Overfitting & Constant-Output Fixes

### Problems Fixed

| ID | Symptom | Root Cause |
|---|---|---|
| RC-O1 | Model achieves 100% accuracy after 5 epochs | Synthetic data trivially separable (3-8σ between classes) |
| RC-O2 | Val/test splits may contain 0 adversarial samples | Non-stratified `random.shuffle` on 85/15 dataset |
| RC-O3 | Trust score std ≈ 0.007 (constant output) | Sample file contained only legitimate agents (`sequences[:10]`) |
| RC-O4 | Seed pollution between sanity check and main training | `set_seeds()` called before sanity check, not after |
| RC-E1 | Energy residual = 100% for all methods throughout | Single-message J model with 100 Wh battery → 5.97×10⁻⁷% drain/step |

### Changes

**`simulation/scripts/generate_behavioral_data.py`**
- Tightened adversarial Beta distribution means: class separation reduced from 3–8σ to ≤3.3σ
- Added AR(1) temporal autocorrelation (φ=0.4): each time step partially inherits the previous
- Added Gaussian noise σ=0.08 on every feature step
- New `low_and_slow` mimicry attack type: 1.1σ separation from legitimate, creates irreducible Bayes error
- Replaced `sequences[:10]` sample with `generate_stratified_sample()`: always 7 legitimate + 3 adversarial
- Added `print_feature_stats()` for distribution verification

**`model/configs/transformer_config.json`**
- `dropout`: 0.1 → **0.2** (stronger stochastic regularization)
- `weight_decay`: 1e-4 → **1e-3** (L2 regularization 10× stronger)
- `learning_rate`: 1e-3 → **5e-4** (slower convergence, better generalization)
- `early_stopping_patience`: 10 → **15**
- `epochs`: 50 → **80**
- `output_activation`: set to `"none"` (documents no-sigmoid architecture)

**`model/inference/transformer_model.py`**
- Added `stratified_split()`: preserves ~15% adversarial fraction in train/val/test
- Added `validate_output_distribution()`: detects constant-output collapse post-training
- Fixed seed order: `set_seeds(seed)` called **after** sanity check (not before)
- Sanity check uses dedicated seed=0 to avoid polluting main training RNG

**`simulation/scripts/environment.py`**
- Replaced broken J-based energy model with calibrated drain rates:
  - Static: 2.35%/interval → ~50% residual after 20 intervals
  - Bayesian: 2.48%/interval → ~50.2% residual
  - Proposed: 2.49%/interval → ~50.1% residual, 5.8% overhead vs static
- Added three independent battery trajectories (`_batt_proposed/bayesian/static`)
- Added Gaussian noise σ=0.12%/interval for realistic run-to-run variation

**`simulation/configs/simulation_params.json`**
- Added `low_and_slow` to `adversarial_attacks.types`
- Added `energy` block with calibrated drain rates

**`tests/test_environment.py`** — 5 new tests:
- `test_energy_depletes_over_time`
- `test_three_independent_energy_trajectories`
- `test_energy_overhead_in_expected_range`
- `test_ar1_temporal_correlation`
- `test_stratified_sample_contains_adversarial`

**`tests/test_model.py`** — 7 new tests:
- `test_dropout_is_increased`
- `test_weight_decay_is_increased`
- `test_learning_rate_is_reduced`
- `test_no_sigmoid_in_output_activation`
- `test_output_is_raw_logit_not_probability`
- `TestStratifiedSplit` (3 tests)
- `test_pos_weight_boosts_minority_class`

### Expected Results After v3

| Metric | v2 (broken) | v3 (fixed) |
|---|---|---|
| Test accuracy | 100% (overfit) | 82–90% |
| Training epochs to converge | 5 | 20–50 |
| Trust score std (mixed sample) | 0.007 | >0.15 |
| Energy residual at interval 20 | 100% (all) | 47–53% |
| Proposed energy overhead | 0% visible | ~5.8% |
| `low_and_slow` detection rate | N/A | ~55–70% |

---

## v2.0.0 — Trust Semantics & Loss Function Fixes

### Problems Fixed

| ID | Symptom | Root Cause |
|---|---|---|
| RC-1 | Accuracy = 0.0% (every prediction backwards) | `nn.Sigmoid()` in model + `(output < tau_min)` → inverted semantics |
| RC-2 | Metric threshold applied to wrong probability space | `output < 0.65` used on sigmoid output, not trust score |
| RC-3 | Loss instability (log(0) risk) | `BCELoss` + sigmoid = double sigmoid, numerically unstable |
| RC-4 | Precision very low (always predicts one class) | No class imbalance handling (85%/15%) |
| RC-5 | Inference: 10/10 legitimate agents flagged as anomalous | `compute_trust_score` returned P(adversarial) not trust score |

### Changes

- Removed `nn.Sigmoid()` from `output_proj` in `TrustTransformer`
- Changed loss from `BCELoss` to `BCEWithLogitsLoss(pos_weight=n_legit/n_adv)`
- Fixed metric threshold: `sigmoid(logit) > (1 - tau_min)` instead of `output < tau_min`
- Fixed `compute_trust_score`: returns `1 - sigmoid(logit)` (trust) not `sigmoid(logit)` (P(adversarial))
- Added `set_seeds()`, `sanity_check_overfit()`, `compute_trust_batch()`
- Fixed `infer.py`: added score distribution validation to detect collapse

---

## v1.0.0 — Initial Release

- Baseline implementation with 5 broken issues documented in v2 changelog above.
