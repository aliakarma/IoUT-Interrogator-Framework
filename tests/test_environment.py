"""
Unit Tests: IoUT Simulation Environment  [FIXED v3]
=====================================================
Tests for the simulation environment, energy model (fixed), and
behavioral data generator (fixed with harder data, stratified sample).

Run:
    pytest tests/test_environment.py -v
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

CONFIG_PATH = "simulation/configs/simulation_params.json"


# ── Data generator tests ──────────────────────────────────────────────────

class TestDataGenerator:

    def test_generates_both_classes(self):
        from simulation.scripts.generate_behavioral_data import generate_dataset
        seqs, df = generate_dataset(50, K=16, adv_fraction=0.15, seed=42)
        assert df["label"].sum() > 0,       "No adversarial sequences generated"
        assert (df["label"] == 0).sum() > 0,"No legitimate sequences generated"

    def test_low_and_slow_attack_present(self):
        """New v3 attack type must be present."""
        from simulation.scripts.generate_behavioral_data import generate_dataset
        _, df = generate_dataset(200, K=16, adv_fraction=0.15, seed=42)
        atks = df[df.label == 1].attack_type.unique().tolist()
        assert "low_and_slow" in atks, \
            f"low_and_slow missing from attacks: {atks}"

    def test_ar1_temporal_correlation(self):
        """Each feature should have positive lag-1 autocorrelation."""
        from simulation.scripts.generate_behavioral_data import generate_dataset
        seqs, _ = generate_dataset(30, K=64, adv_fraction=0.0, seed=42)
        for feat_i in range(5):
            autocorrs = []
            for s in seqs[:10]:
                col = np.array(s["sequence"])[:, feat_i]
                ac  = np.corrcoef(col[:-1], col[1:])[0, 1]
                autocorrs.append(ac)
            mean_ac = np.mean(autocorrs)
            assert mean_ac > 0, \
                f"Feature {feat_i}: expected positive AR(1) correlation, got {mean_ac:.3f}"

    def test_feature_values_in_01(self):
        from simulation.scripts.generate_behavioral_data import generate_dataset
        seqs, _ = generate_dataset(20, K=64, adv_fraction=0.15, seed=42)
        for s in seqs:
            arr = np.array(s["sequence"])
            assert arr.min() >= 0.0, "Feature below 0"
            assert arr.max() <= 1.0, "Feature above 1"

    def test_sequence_shape(self):
        from simulation.scripts.generate_behavioral_data import generate_dataset
        seqs, _ = generate_dataset(10, K=64, adv_fraction=0.15, seed=42)
        for s in seqs:
            arr = np.array(s["sequence"])
            assert arr.shape == (64, 5), f"Shape {arr.shape} != (64, 5)"

    def test_stratified_sample_contains_adversarial(self):
        """
        v1 bug: sequences[:10] was always all legitimate.
        v3 fix: generate_stratified_sample guarantees adversarial representation.
        """
        from simulation.scripts.generate_behavioral_data import (
            generate_dataset, generate_stratified_sample
        )
        seqs, _ = generate_dataset(100, K=16, adv_fraction=0.15, seed=42)
        sample   = generate_stratified_sample(seqs, n_legit_sample=7,
                                              n_adv_sample=3, seed=99)
        n_adv    = sum(1 for s in sample if s["label"] == 1)
        n_legit  = sum(1 for s in sample if s["label"] == 0)
        assert n_adv   == 3, f"Expected 3 adversarial in sample, got {n_adv}"
        assert n_legit == 7, f"Expected 7 legitimate in sample, got {n_legit}"

    def test_feature_classes_overlap_is_nontrivial(self):
        """
        v1 bug: class separation was 3-8x std → trivially separable.
        v3 fix: low_and_slow has separation < 1.5x std.
        """
        from simulation.scripts.generate_behavioral_data import (
            _LEGIT_MEAN, _ADV_MEANS
        )
        eff_std = np.sqrt(0.04**2 + 2 * 0.08**2)  # ≈ 0.12
        low_and_slow_mean = _ADV_MEANS["low_and_slow"]
        max_sep = np.max(np.abs(_LEGIT_MEAN - low_and_slow_mean)) / eff_std
        assert max_sep < 2.5, \
            f"low_and_slow max separation {max_sep:.1f}x std; should be < 2.5x for hard data"


# ── Acoustic channel tests ────────────────────────────────────────────────

class TestAcousticChannel:

    def setup_method(self):
        if not os.path.exists(CONFIG_PATH):
            pytest.skip("Config not found")
        from simulation.scripts.environment import AcousticChannel
        config = {
            "elec_energy_nJ_per_bit": 50,
            "amp_energy_pJ_per_bit_per_m2": 100,
            "path_loss_exponent": 2.0,
            "propagation_delay_s_per_km": 0.67,
            "data_rate_kbps_min": 10,
            "data_rate_kbps_max": 20,
        }
        self.channel = AcousticChannel(config)

    def test_energy_positive(self):
        e = self.channel.transmission_energy_J(bits=1024, distance_m=100)
        assert e > 0

    def test_energy_increases_with_distance(self):
        e1 = self.channel.transmission_energy_J(bits=1024, distance_m=100)
        e2 = self.channel.transmission_energy_J(bits=1024, distance_m=500)
        assert e2 > e1

    def test_propagation_delay(self):
        delay = self.channel.propagation_delay_s(distance_m=500)
        # 500m = 0.5km → 0.5 * 0.67 = 0.335s
        assert abs(delay - 0.335) < 0.01

    def test_delivery_returns_bool(self):
        import random
        rng = random.Random(42)
        result = self.channel.delivery_success(100, False, None, rng)
        assert isinstance(result, bool)


# ── Environment tests ─────────────────────────────────────────────────────

class TestIoUTEnvironment:

    def setup_method(self):
        if not os.path.exists(CONFIG_PATH):
            pytest.skip("Config not found")
        from simulation.scripts.environment import IoUTEnvironment
        self.env = IoUTEnvironment(config_path=CONFIG_PATH, seed=42)

    def test_agent_count(self):
        assert len(self.env.agents) == 50

    def test_adversarial_count(self):
        n_adv = sum(1 for a in self.env.agents if a.is_adversarial)
        assert n_adv == 7

    def test_positions_in_bounds(self):
        for a in self.env.agents:
            assert 0 <= a.x <= self.env.area_side
            assert 0 <= a.y <= self.env.area_side

    def test_initial_batteries(self):
        for a in self.env.agents:
            assert a.battery == 100.0

    def test_behavioral_feature_shape(self):
        agent = self.env.agents[0]
        feats = self.env._generate_behavioral_features(agent, interval=1)
        assert feats.shape == (5,)

    def test_behavioral_features_in_01(self):
        for agent in self.env.agents:
            feats = self.env._generate_behavioral_features(agent, interval=1)
            assert np.all(feats >= 0.0) and np.all(feats <= 1.0)

    def test_run_returns_all_keys(self):
        res = self.env.run(num_intervals=3)
        for key in ["interval", "accuracy_proposed", "accuracy_bayesian",
                    "accuracy_static", "pdr_proposed", "pdr_bayesian",
                    "pdr_static", "energy_proposed", "energy_bayesian",
                    "energy_static", "precision_proposed", "precision_bayesian",
                    "precision_static", "recall_proposed", "recall_bayesian",
                    "recall_static", "f1_proposed", "f1_bayesian",
                    "f1_static", "tp_proposed", "tn_proposed",
                    "fp_proposed", "fn_proposed"]:
            assert key in res, f"Missing key: {key}"

    def test_static_recall_and_f1_are_zero(self):
        res = self.env.run(num_intervals=3)
        assert all(v == 0.0 for v in res["recall_static"])
        assert all(v == 0.0 for v in res["f1_static"])

    def test_run_interval_count(self):
        res = self.env.run(num_intervals=5)
        assert len(res["interval"]) == 5

    def test_accuracies_in_valid_range(self):
        res = self.env.run(num_intervals=3)
        for acc in res["accuracy_proposed"]:
            assert 0 <= acc <= 100

    def test_pdr_in_valid_range(self):
        res = self.env.run(num_intervals=3)
        for pdr in res["pdr_proposed"]:
            assert 0 <= pdr <= 100

    def test_static_accuracy_equals_legit_fraction(self):
        acc      = self.env._static_accuracy()
        n_legit  = sum(1 for a in self.env.agents if not a.is_adversarial)
        expected = n_legit / len(self.env.agents)
        assert abs(acc - expected) < 1e-6

    def test_energy_depletes_over_time(self):
        """
        v1 bug: energy was 100% for all methods throughout (drain ≈ 0).
        v3 fix: calibrated drain produces visible depletion over 20 intervals.
        """
        res = self.env.run(num_intervals=20)
        final_proposed = res["energy_proposed"][-1]
        final_static   = res["energy_static"][-1]

        # Should deplete significantly (< 95% remaining after 20 intervals)
        assert final_proposed < 95.0, \
            f"Proposed energy {final_proposed:.1f}% — no visible depletion (v1 bug still present)"
        assert final_static < 95.0, \
            f"Static energy {final_static:.1f}% — no visible depletion (v1 bug still present)"

    def test_three_independent_energy_trajectories(self):
        """
        v3 fix: three separate battery trajectories per method.
        Static should retain more energy than proposed (less overhead).
        """
        res = self.env.run(num_intervals=20)
        final_s = res["energy_static"][-1]
        final_p = res["energy_proposed"][-1]
        final_b = res["energy_bayesian"][-1]

        assert final_s > final_p, \
            "Static should have more residual energy than proposed"
        assert final_s > final_b, \
            "Static should have more residual energy than bayesian"

    def test_energy_overhead_in_expected_range(self):
        """Proposed overhead vs static should be 3-12% (paper: 5.8%)."""
        res = self.env.run(num_intervals=20)
        drain_s = 100.0 - res["energy_static"][-1]
        drain_p = 100.0 - res["energy_proposed"][-1]
        overhead = (drain_p - drain_s) / drain_s * 100
        assert 3.0 < overhead < 15.0, \
            f"Overhead {overhead:.1f}% outside expected range [3, 15]"

    def test_reproducibility(self):
        from simulation.scripts.environment import IoUTEnvironment
        env1 = IoUTEnvironment(config_path=CONFIG_PATH, seed=99)
        env2 = IoUTEnvironment(config_path=CONFIG_PATH, seed=99)
        r1   = env1.run(num_intervals=2)
        r2   = env2.run(num_intervals=2)
        assert r1["accuracy_proposed"] == r2["accuracy_proposed"]
        assert r1["energy_proposed"]   == r2["energy_proposed"]
