"""
Unit Tests: IoUT Simulation Environment
========================================
"""
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from simulation.scripts.environment import IoUTEnvironment, AcousticChannel


CONFIG_PATH = "simulation/configs/simulation_params.json"


class TestAcousticChannel:

    def setup_method(self):
        config = {
            "elec_energy_nJ_per_bit": 50,
            "amp_energy_pJ_per_bit_per_m2": 100,
            "path_loss_exponent": 2.0,
            "propagation_delay_s_per_km": 0.67,
            "data_rate_kbps_min": 10,
            "data_rate_kbps_max": 20,
        }
        self.channel = AcousticChannel(config)

    def test_transmission_energy_positive(self):
        energy = self.channel.transmission_energy_J(bits=1024, distance_m=100)
        assert energy > 0

    def test_transmission_energy_increases_with_distance(self):
        e1 = self.channel.transmission_energy_J(bits=1024, distance_m=100)
        e2 = self.channel.transmission_energy_J(bits=1024, distance_m=500)
        assert e2 > e1

    def test_propagation_delay_positive(self):
        delay = self.channel.propagation_delay_s(distance_m=500)
        assert delay > 0
        # 500 m = 0.5 km → 0.5 * 0.67 = 0.335 s
        assert abs(delay - 0.335) < 0.01

    def test_delivery_success_returns_bool(self):
        import random
        rng = random.Random(42)
        result = self.channel.delivery_success(100, False, None, rng)
        assert isinstance(result, bool)


class TestIoUTEnvironment:

    def setup_method(self):
        if not os.path.exists(CONFIG_PATH):
            pytest.skip(f"Config file not found: {CONFIG_PATH}")
        self.env = IoUTEnvironment(config_path=CONFIG_PATH, seed=42)

    def test_agent_count(self):
        assert len(self.env.agents) == 50

    def test_adversarial_count(self):
        n_adv = sum(1 for a in self.env.agents if a.is_adversarial)
        assert n_adv == 7   # 15% of 50 = 7.5 → 7

    def test_agent_positions_in_bounds(self):
        for agent in self.env.agents:
            assert 0 <= agent.x <= self.env.area_side
            assert 0 <= agent.y <= self.env.area_side

    def test_agent_batteries_start_full(self):
        for agent in self.env.agents:
            assert agent.battery == 100.0

    def test_behavioral_features_shape(self):
        agent = self.env.agents[0]
        feats = self.env._generate_behavioral_features(agent, interval=1)
        assert feats.shape == (5,)

    def test_behavioral_features_range(self):
        for agent in self.env.agents:
            feats = self.env._generate_behavioral_features(agent, interval=1)
            assert np.all(feats >= 0.0)
            assert np.all(feats <= 1.0)

    def test_run_returns_expected_keys(self):
        results = self.env.run(num_intervals=3)
        expected_keys = [
            "interval",
            "accuracy_proposed", "accuracy_bayesian", "accuracy_static",
            "pdr_proposed", "pdr_bayesian", "pdr_static",
            "energy_proposed", "energy_bayesian", "energy_static",
        ]
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"

    def test_run_interval_count(self):
        results = self.env.run(num_intervals=5)
        assert len(results["interval"]) == 5

    def test_accuracy_in_valid_range(self):
        results = self.env.run(num_intervals=3)
        for acc in results["accuracy_proposed"]:
            assert 0 <= acc <= 100

    def test_pdr_in_valid_range(self):
        results = self.env.run(num_intervals=3)
        for pdr in results["pdr_proposed"]:
            assert 0 <= pdr <= 100

    def test_static_accuracy_equals_legit_fraction(self):
        """Static trust accuracy = fraction of legitimate agents."""
        acc = self.env._static_accuracy()
        n_legit = sum(1 for a in self.env.agents if not a.is_adversarial)
        expected = n_legit / len(self.env.agents)
        assert abs(acc - expected) < 1e-6

    def test_reproducibility(self):
        """Same seed produces identical results."""
        env1 = IoUTEnvironment(config_path=CONFIG_PATH, seed=99)
        env2 = IoUTEnvironment(config_path=CONFIG_PATH, seed=99)
        r1 = env1.run(num_intervals=2)
        r2 = env2.run(num_intervals=2)
        assert r1["accuracy_proposed"] == r2["accuracy_proposed"]
