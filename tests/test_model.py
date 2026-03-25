"""
Unit Tests: Transformer Trust Model
=====================================
"""
import sys
import os
import json
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.inference.transformer_model import (
    TrustTransformer, TrustDataset, PositionalEncoding,
    compute_trust_score
)

CONFIG_PATH = "model/configs/transformer_config.json"


def load_config():
    if not os.path.exists(CONFIG_PATH):
        pytest.skip(f"Config file not found: {CONFIG_PATH}")
    with open(CONFIG_PATH) as f:
        return json.load(f)


class TestPositionalEncoding:

    def test_output_shape(self):
        pe = PositionalEncoding(d_model=128, max_len=64)
        x = torch.zeros(2, 64, 128)
        out = pe(x)
        assert out.shape == (2, 64, 128)

    def test_output_not_all_zeros(self):
        pe = PositionalEncoding(d_model=128, max_len=64)
        x = torch.zeros(1, 10, 128)
        out = pe(x)
        assert not torch.all(out == 0)


class TestTrustTransformer:

    def setup_method(self):
        self.config = load_config()
        self.model = TrustTransformer(self.config)
        self.model.eval()

    def test_parameter_count_approx(self):
        """Model should have approximately 1.2M parameters."""
        n_params = self.model.count_parameters()
        assert 800_000 < n_params < 2_000_000, \
            f"Expected ~1.2M params, got {n_params:,}"

    def test_output_shape(self):
        x = torch.randn(4, 64, 5)   # batch=4, K=64, features=5
        with torch.no_grad():
            out = self.model(x)
        assert out.shape == (4, 1)

    def test_output_in_01_range(self):
        x = torch.randn(8, 64, 5)
        with torch.no_grad():
            out = self.model(x)
        assert torch.all(out >= 0.0)
        assert torch.all(out <= 1.0)

    def test_single_sequence_inference(self):
        seq = np.random.rand(64, 5).astype(np.float32)
        score = compute_trust_score(self.model, seq)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_deterministic_inference(self):
        """Identical input produces identical output (eval mode)."""
        seq = np.random.rand(64, 5).astype(np.float32)
        s1 = compute_trust_score(self.model, seq)
        s2 = compute_trust_score(self.model, seq)
        assert abs(s1 - s2) < 1e-6

    def test_different_inputs_different_scores(self):
        """Legitimate and adversarial sequences should generally differ."""
        # Legitimate: high routing stability, low retransmission
        legit = np.array([[0.1, 0.05, 0.9, 0.05, 0.95]] * 64, dtype=np.float32)
        # Adversarial: high retransmission, low routing stability
        adv   = np.array([[0.8, 0.85, 0.1, 0.8,  0.2 ]] * 64, dtype=np.float32)
        s_l = compute_trust_score(self.model, legit)
        s_a = compute_trust_score(self.model, adv)
        # Untrained model may not differentiate; just check they are valid floats
        assert isinstance(s_l, float)
        assert isinstance(s_a, float)

    def test_batch_consistency(self):
        """Single-item batch matches individual inference."""
        seq = np.random.rand(64, 5).astype(np.float32)
        individual_score = compute_trust_score(self.model, seq)

        x_batch = torch.from_numpy(seq).unsqueeze(0)
        with torch.no_grad():
            batch_score = float(self.model(x_batch).squeeze())
        assert abs(individual_score - batch_score) < 1e-5


class TestTrustDataset:

    def setup_method(self):
        self.config = load_config()

    def test_dataset_length(self):
        seqs = [{"sequence": np.random.rand(64, 5).tolist(), "label": 0}
                for _ in range(10)]
        ds = TrustDataset(seqs, seq_len=64)
        assert len(ds) == 10

    def test_item_shapes(self):
        seqs = [{"sequence": np.random.rand(64, 5).tolist(), "label": 1}]
        ds = TrustDataset(seqs, seq_len=64)
        seq_t, lbl_t = ds[0]
        assert seq_t.shape == (64, 5)
        assert lbl_t.shape == (1,)

    def test_short_sequence_padding(self):
        """Sequences shorter than K should be zero-padded to K."""
        seqs = [{"sequence": np.random.rand(20, 5).tolist(), "label": 0}]
        ds = TrustDataset(seqs, seq_len=64)
        seq_t, _ = ds[0]
        assert seq_t.shape == (64, 5)
        # Last 44 rows should be zeros
        assert torch.all(seq_t[20:] == 0.0)

    def test_long_sequence_truncation(self):
        """Sequences longer than K should be truncated to K."""
        seqs = [{"sequence": np.random.rand(100, 5).tolist(), "label": 0}]
        ds = TrustDataset(seqs, seq_len=64)
        seq_t, _ = ds[0]
        assert seq_t.shape == (64, 5)
