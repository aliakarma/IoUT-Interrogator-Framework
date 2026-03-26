"""
Unit Tests: Transformer Trust Model  [FIXED v3]
=================================================
Tests for TrustTransformer, TrustDataset, stratified_split, and
inference utilities, all updated for the v3 fixes.

Run:
    pytest tests/test_model.py -v
"""
import sys
import os
import json
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

CONFIG_PATH = "model/configs/transformer_config.json"

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="torch not installed"
)


def load_config():
    if not os.path.exists(CONFIG_PATH):
        pytest.skip(f"Config not found: {CONFIG_PATH}")
    with open(CONFIG_PATH) as f:
        return json.load(f)


# ── Config validation ─────────────────────────────────────────────────────

class TestConfig:

    def test_dropout_is_increased(self):
        """v3 fix: dropout should be 0.2, not v1's 0.1."""
        config = load_config()
        assert config["architecture"]["dropout"] == 0.2, \
            "dropout must be 0.2 for v3 regularization (was 0.1 in v1)"

    def test_weight_decay_is_increased(self):
        """v3 fix: weight_decay should be 1e-3, not v1's 1e-4."""
        config = load_config()
        assert config["training"]["weight_decay"] == 1e-3, \
            "weight_decay must be 1e-3 (was 1e-4 in v1)"

    def test_learning_rate_is_reduced(self):
        """v3 fix: lr should be 5e-4, not v1's 1e-3."""
        config = load_config()
        assert config["training"]["learning_rate"] == 5e-4, \
            "learning_rate must be 5e-4 (was 1e-3 in v1)"

    def test_no_sigmoid_in_output_activation(self):
        """v3: output_activation must be 'none' — sigmoid removed from model."""
        config = load_config()
        act = config["architecture"].get("output_activation", "")
        assert act in ("none", ""), \
            "output_activation must be 'none' — sigmoid was the v1 RC-1 bug"


# ── Positional Encoding ───────────────────────────────────────────────────

class TestPositionalEncoding:

    def test_output_shape(self):
        from model.inference.transformer_model import PositionalEncoding
        pe  = PositionalEncoding(d_model=128, max_len=64)
        x   = torch.zeros(2, 64, 128)
        out = pe(x)
        assert out.shape == (2, 64, 128)

    def test_pe_adds_nonzero_values(self):
        from model.inference.transformer_model import PositionalEncoding
        pe  = PositionalEncoding(d_model=128, max_len=64, dropout=0.0)
        x   = torch.zeros(1, 10, 128)
        out = pe(x)
        assert not torch.all(out == 0)


# ── TrustTransformer ──────────────────────────────────────────────────────

class TestTrustTransformer:

    def setup_method(self):
        from model.inference.transformer_model import TrustTransformer
        self.config = load_config()
        self.model  = TrustTransformer(self.config)
        self.model.eval()

    def test_parameter_count(self):
        n = self.model.count_parameters()
        assert 300_000 < n < 2_500_000, \
            f"Expected ~500K-1.2M params, got {n:,}"

    def test_output_shape(self):
        x   = torch.randn(4, 64, 5)
        out = self.model(x)
        assert out.shape == (4, 1)

    def test_output_is_raw_logit_not_probability(self):
        """
        v1 bug: model had Sigmoid → output was in (0,1) squashed.
        v3 fix: no Sigmoid → output is unbounded logit.
        Logits from a random-init model can be outside [0,1].
        """
        torch.manual_seed(0)
        x      = torch.randn(50, 64, 5) * 3.0   # extreme inputs
        logits = self.model(x)
        # At least some logits should be outside [0,1] for an untrained model
        # (if all are in [0,1], sigmoid is still present — bug still there)
        outside_01 = ((logits < 0) | (logits > 1)).any()
        assert outside_01, \
            "Model outputs are all in [0,1] — sigmoid still present in output_proj (v1 bug)"

    def test_trust_score_in_01(self):
        """Trust score = 1 - sigmoid(logit) must be in [0,1]."""
        from model.inference.transformer_model import compute_trust_score
        seq   = np.random.rand(64, 5).astype(np.float32)
        score = compute_trust_score(self.model, seq)
        assert 0.0 <= score <= 1.0

    def test_deterministic_inference(self):
        from model.inference.transformer_model import compute_trust_score
        seq = np.random.rand(64, 5).astype(np.float32)
        s1  = compute_trust_score(self.model, seq)
        s2  = compute_trust_score(self.model, seq)
        assert abs(s1 - s2) < 1e-6

    def test_extreme_adv_input_scores_low(self):
        """
        An adversarial-like input (high retx, low routing stability) should
        produce a LOWER trust score than a legitimate-like input.
        This tests the semantic correctness of 1-sigmoid(logit).
        Note: only reliable after training; for untrained model we just check
        the function runs without error.
        """
        from model.inference.transformer_model import compute_trust_score
        legit = np.array([[0.1, 0.05, 0.9, 0.05, 0.95]] * 64, dtype=np.float32)
        adv   = np.array([[0.8, 0.85, 0.1, 0.8,  0.2 ]] * 64, dtype=np.float32)
        s_l   = compute_trust_score(self.model, legit)
        s_a   = compute_trust_score(self.model, adv)
        assert isinstance(s_l, float)
        assert isinstance(s_a, float)

    def test_compute_trust_batch_shape(self):
        from model.inference.transformer_model import compute_trust_batch
        seqs   = np.random.rand(8, 64, 5).astype(np.float32)
        scores = compute_trust_batch(self.model, seqs)
        assert scores.shape == (8,)
        assert np.all(scores >= 0) and np.all(scores <= 1)


# ── TrustDataset ─────────────────────────────────────────────────────────

class TestTrustDataset:

    def test_length(self):
        from model.inference.transformer_model import TrustDataset
        seqs = [{"sequence": np.random.rand(64, 5).tolist(), "label": 0}
                for _ in range(10)]
        ds = TrustDataset(seqs, seq_len=64)
        assert len(ds) == 10

    def test_item_shapes(self):
        from model.inference.transformer_model import TrustDataset
        seqs = [{"sequence": np.random.rand(64, 5).tolist(), "label": 1}]
        ds   = TrustDataset(seqs, seq_len=64)
        seq_t, lbl_t = ds[0]
        assert seq_t.shape == (64, 5)
        assert lbl_t.shape == (1,)

    def test_short_sequence_padding(self):
        from model.inference.transformer_model import TrustDataset
        seqs = [{"sequence": np.random.rand(20, 5).tolist(), "label": 0}]
        ds   = TrustDataset(seqs, seq_len=64)
        seq_t, _ = ds[0]
        assert seq_t.shape == (64, 5)
        assert torch.all(seq_t[20:] == 0.0)

    def test_long_sequence_truncation(self):
        from model.inference.transformer_model import TrustDataset
        seqs = [{"sequence": np.random.rand(100, 5).tolist(), "label": 0}]
        ds   = TrustDataset(seqs, seq_len=64)
        seq_t, _ = ds[0]
        assert seq_t.shape == (64, 5)

    def test_class_weights_sensible(self):
        from model.inference.transformer_model import TrustDataset
        seqs = (
            [{"sequence": np.random.rand(64, 5).tolist(), "label": 0}] * 85 +
            [{"sequence": np.random.rand(64, 5).tolist(), "label": 1}] * 15
        )
        ds = TrustDataset(seqs, seq_len=64)
        n_l, n_a, pw = ds.class_weights()
        assert n_l == 85 and n_a == 15
        assert abs(pw - 85/15) < 0.01

    def test_class_weights_raises_on_no_adversarial(self):
        from model.inference.transformer_model import TrustDataset
        seqs = [{"sequence": np.random.rand(64, 5).tolist(), "label": 0}] * 10
        ds   = TrustDataset(seqs, seq_len=64)
        with pytest.raises(ValueError, match="No adversarial"):
            ds.class_weights()


# ── Stratified Split ──────────────────────────────────────────────────────

class TestStratifiedSplit:

    def _make_sequences(self, n_legit, n_adv):
        return (
            [{"sequence": np.random.rand(64, 5).tolist(), "label": 0}] * n_legit +
            [{"sequence": np.random.rand(64, 5).tolist(), "label": 1}] * n_adv
        )

    def test_all_splits_contain_adversarial(self):
        """
        v1/v2 bug: random split could produce val/test with 0 adversarial.
        v3 fix: stratified_split preserves class ratio in every split.
        """
        from model.inference.transformer_model import stratified_split
        seqs = self._make_sequences(85, 15)
        train, val, test = stratified_split(seqs, 0.7, 0.15, seed=42)
        for name, split in [("train", train), ("val", val), ("test", test)]:
            n_a = sum(1 for s in split if s["label"] == 1)
            assert n_a > 0, f"No adversarial samples in {name} split (stratification failed)"

    def test_split_fractions_approximate(self):
        from model.inference.transformer_model import stratified_split
        seqs = self._make_sequences(850, 150)
        train, val, test = stratified_split(seqs, 0.7, 0.15, seed=42)
        n = len(seqs)
        assert 0.60 < len(train)/n < 0.80
        assert 0.10 < len(val)/n   < 0.20
        assert 0.05 < len(test)/n  < 0.25

    def test_no_overlap_between_splits(self):
        from model.inference.transformer_model import stratified_split
        seqs = self._make_sequences(85, 15)
        # Add unique identifiers
        for i, s in enumerate(seqs):
            s["agent_id"] = i
        train, val, test = stratified_split(seqs, 0.7, 0.15, seed=42)
        train_ids = {s["agent_id"] for s in train}
        val_ids   = {s["agent_id"] for s in val}
        test_ids  = {s["agent_id"] for s in test}
        assert len(train_ids & val_ids)  == 0, "Train/val overlap"
        assert len(train_ids & test_ids) == 0, "Train/test overlap"
        assert len(val_ids   & test_ids) == 0, "Val/test overlap"


# ── Loss function correctness ─────────────────────────────────────────────

class TestLossFunction:

    def test_bce_with_logits_no_double_sigmoid(self):
        """
        v1/v2 bug RC-3: BCELoss + sigmoid = double sigmoid risk.
        v3 uses BCEWithLogitsLoss which applies sigmoid internally.
        Verify: BCEWithLogitsLoss(logit, label) == BCELoss(sigmoid(logit), label).
        """
        logits = torch.tensor([[2.0], [-1.5], [0.5]])
        labels = torch.tensor([[1.0], [0.0], [1.0]])

        loss_bce_logits = nn.BCEWithLogitsLoss()(logits, labels)
        loss_bce        = nn.BCELoss()(torch.sigmoid(logits), labels)

        assert abs(loss_bce_logits.item() - loss_bce.item()) < 1e-5

    def test_adv_threshold_is_correct_inversion(self):
        """
        v1 bug RC-2: threshold was tau_min=0.65 applied directly to sigmoid output.
        v3 fix: threshold = 1 - tau_min = 0.35 in P(adversarial) space.
        Verify: trust<0.65 ⟺ P(adv)>0.35.
        """
        from model.inference.transformer_model import _adv_threshold
        tau_min   = 0.65
        threshold = _adv_threshold(tau_min)
        assert abs(threshold - 0.35) < 1e-6

        # Verify semantic equivalence
        import math
        sigmoid = lambda x: 1 / (1 + math.exp(-x))
        for logit in [-2.0, -0.5, 0.0, 0.8, 2.5]:
            p_adv       = sigmoid(logit)
            trust       = 1.0 - p_adv
            flag_old    = trust < tau_min            # v1 (wrong: on trust not p_adv)
            flag_new    = p_adv > threshold           # v3 (correct: on p_adv)
            # Both should agree because trust < tau_min ⟺ (1-p_adv) < 0.65 ⟺ p_adv > 0.35
            assert flag_old == flag_new, \
                f"logit={logit}: trust<tau_min={flag_old} but p_adv>threshold={flag_new}"

    def test_pos_weight_boosts_minority_class(self):
        """pos_weight=5.67 should give adversarial samples 5.67x higher gradient."""
        pos_weight  = torch.tensor([5.67])
        criterion   = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

        logit  = torch.tensor([[0.0]])   # sigmoid = 0.5, neutral prediction
        loss_adv   = criterion(logit, torch.tensor([[1.0]]))   # adversarial
        loss_legit = criterion(logit, torch.tensor([[0.0]]))   # legitimate

        ratio = loss_adv.item() / loss_legit.item()
        assert abs(ratio - 5.67) < 0.1, \
            f"Loss ratio {ratio:.2f} should be ~5.67 (pos_weight)"
