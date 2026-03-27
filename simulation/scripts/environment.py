"""
IoUT Multi-Agent Simulation Environment
========================================
Lightweight Python simulation of the Internet of Underwater Things (IoUT)
multi-agent environment described in the paper. Replaces NS-3 Aqua-Sim for
reproducibility purposes; core acoustic channel physics and agent behavioral
metadata generation are preserved.

Usage:
    from simulation.scripts.environment import IoUTEnvironment
    env = IoUTEnvironment(config_path="simulation/configs/simulation_params.json")
    results = env.run(num_intervals=20)
"""

import json
import math
import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

from simulation.scripts.generate_behavioral_data import (
    _ADV_MEANS,
    _LEGIT_MEAN,
    AR_PHI,
    NOISE_SIGMA,
)
from blockchain.scripts.mock_ledger import PBFTConsortium, TrustRecord


@dataclass
class Agent:
    """Represents a single IoUT agent (sensor node or AUV)."""
    agent_id: int
    agent_type: str          # "sensor" or "auv"
    x: float                 # position x (metres)
    y: float                 # position y (metres)
    is_adversarial: bool = False
    battery: float = 100.0   # residual energy percentage
    attack_type: Optional[str] = None

    # Behavioral state
    retransmit_count: int = 0
    packets_sent: int = 0
    packets_dropped: int = 0
    route_changes: int = 0
    neighbor_changes: int = 0
    protocol_violations: int = 0


class AcousticChannel:
    """
    Models the underwater acoustic channel using the Thorp attenuation model.
    Reference: Akyildiz et al. (2005), Ad Hoc Networks.
    """

    def __init__(self, config: dict):
        self.elec_nJ = config["elec_energy_nJ_per_bit"]
        self.amp_pJ = config["amp_energy_pJ_per_bit_per_m2"]
        self.k = config["path_loss_exponent"]
        self.delay_s_km = config["propagation_delay_s_per_km"]
        self.rate_kbps_min = config["data_rate_kbps_min"]
        self.rate_kbps_max = config["data_rate_kbps_max"]

    def transmission_energy_J(self, bits: int, distance_m: float) -> float:
        """E_tx(l, d) = l * E_elec + l * eps_amp * d^k  (Equation 2 in paper)."""
        e_elec = bits * self.elec_nJ * 1e-9
        e_amp = bits * self.amp_pJ * 1e-12 * (distance_m ** self.k)
        return e_elec + e_amp

    def propagation_delay_s(self, distance_m: float) -> float:
        return (distance_m / 1000.0) * self.delay_s_km

    def delivery_success(self, distance_m: float, is_adversarial: bool,
                         attack_type: Optional[str], rng: random.Random) -> bool:
        """Stochastic packet delivery: adversarial nodes may drop packets."""
        base_loss = min(0.05 + distance_m / 20000.0, 0.3)
        if is_adversarial and attack_type == "selective_packet_drop":
            base_loss = 0.4
        elif is_adversarial and attack_type == "transmission_burst":
            base_loss += 0.15
        return rng.random() > base_loss


class IoUTEnvironment:
    """
    Main IoUT multi-agent simulation environment.

    Simulates:
      - Agent deployment and mobility (restricted random waypoint)
      - Acoustic communication with Thorp channel model
      - Adversarial behavior injection (4 attack types)
      - Behavioral metadata generation per monitoring interval
      - Static trust, Bayesian trust, and proposed interrogator baselines

        Note:
            This is a high-fidelity Python proxy of the NS-3 Aqua-Sim workflow,
            not a direct packet-level NS-3 execution.
    """

    def __init__(self, config_path: str = "simulation/configs/simulation_params.json",
                 seed: int = 42):
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.net_cfg = self.config["network"]
        self.mon_cfg = self.config["monitoring"]
        self.mob_cfg = self.config["agent_mobility"]
        self.adv_cfg = self.config["adversarial_attacks"]
        self.energy_cfg = self.config["energy"]
        self.routing_cfg = self.config.get("routing", {})
        self.blockchain_cfg = self.config.get("blockchain", {})

        self.channel = AcousticChannel(self.config["acoustic_channel"])
        self.area_side = math.sqrt(self.net_cfg["deployment_area_km2"]) * 1000  # metres

        self.agents: List[Agent] = []
        self._deploy_agents()
        self._init_behavior_dynamics()

    def _init_behavior_dynamics(self):
        """Initialise per-agent temporal feature dynamics to match training data."""
        self._ar_phi = float(AR_PHI)
        self._noise_sigma = float(NOISE_SIGMA)
        self._feature_mean: Dict[int, np.ndarray] = {}
        self._feature_state: Dict[int, np.ndarray] = {}

        for agent in self.agents:
            if agent.is_adversarial:
                base_mean = _ADV_MEANS.get(
                    str(agent.attack_type),
                    _ADV_MEANS["coordinated_insider"],
                )
                jitter_sigma = 0.05
            else:
                base_mean = _LEGIT_MEAN
                jitter_sigma = 0.04

            jitter = self.np_rng.normal(0.0, jitter_sigma, size=base_mean.shape).astype(np.float32)
            mean = np.clip(base_mean + jitter, 0.05, 0.95).astype(np.float32)
            self._feature_mean[agent.agent_id] = mean
            self._feature_state[agent.agent_id] = mean.copy()

    def _next_behavioral_vector(self, agent: Agent) -> np.ndarray:
        """One-step mean-reverting temporal update (aligned with data generator)."""
        mean = self._feature_mean[agent.agent_id]
        prev = self._feature_state[agent.agent_id]

        if agent.is_adversarial:
            phi = self._ar_phi * 0.7
            noise_sigma = self._noise_sigma * 1.3
        else:
            phi = self._ar_phi
            noise_sigma = self._noise_sigma

        eps = self.np_rng.normal(0.0, noise_sigma, size=mean.shape).astype(np.float32)
        curr = np.clip(phi * prev + (1.0 - phi) * mean + eps, 0.0, 1.0).astype(np.float32)
        self._feature_state[agent.agent_id] = curr
        return curr

    # ------------------------------------------------------------------
    # Agent deployment
    # ------------------------------------------------------------------

    def _deploy_agents(self):
        """Deploy agents uniformly in the deployment area."""
        n = self.net_cfg["num_agents"]
        n_adv = self.net_cfg["num_adversarial"]
        attack_types = self.adv_cfg["types"]

        adv_ids = set(self.rng.sample(range(n), n_adv))

        for i in range(n):
            atype = "auv" if self.rng.random() < self.mob_cfg["auv_fraction"] else "sensor"
            x = self.rng.uniform(0, self.area_side)
            y = self.rng.uniform(0, self.area_side)
            is_adv = i in adv_ids
            if is_adv:
                # Emphasize subtle mimicry attacks to avoid unrealistically easy detection.
                if "low_and_slow" in attack_types and self.rng.random() < 0.45:
                    atk = "low_and_slow"
                else:
                    non_mimicry = [t for t in attack_types if t != "low_and_slow"]
                    atk = self.rng.choice(non_mimicry or attack_types)
            else:
                atk = None
            self.agents.append(Agent(
                agent_id=i, agent_type=atype, x=x, y=y,
                is_adversarial=is_adv, attack_type=atk
            ))

    # ------------------------------------------------------------------
    # Mobility update
    # ------------------------------------------------------------------

    def _move_agents(self):
        """Apply restricted random waypoint mobility to AUV agents."""
        max_speed = self.mob_cfg["max_speed_m_per_s"]
        interval_s = self.mon_cfg["interval_s"]
        for agent in self.agents:
            if agent.agent_type == "auv":
                dx = self.rng.uniform(-1, 1) * max_speed * interval_s
                dy = self.rng.uniform(-1, 1) * max_speed * interval_s
                agent.x = max(0, min(self.area_side, agent.x + dx))
                agent.y = max(0, min(self.area_side, agent.y + dy))

    # ------------------------------------------------------------------
    # Distance utility
    # ------------------------------------------------------------------

    def _distance(self, a: Agent, b: Agent) -> float:
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    # ------------------------------------------------------------------
    # Behavioral metadata generation
    # ------------------------------------------------------------------

    def _generate_behavioral_features(self, agent: Agent,
                                       interval: int) -> np.ndarray:
        """
        Generate 5-dimensional behavioral feature vector for one monitoring window.

        Features (matching paper Section IV.C):
          [0] inter_packet_timing_variance  (normalized 0-1)
          [1] retransmission_frequency      (normalized 0-1)
          [2] routing_stability_index       (0=unstable, 1=stable)
          [3] neighbor_churn_rate           (normalized 0-1)
          [4] protocol_compliance_score     (0=non-compliant, 1=compliant)
        """
        if agent.is_adversarial:
            return self._adversarial_features(agent, interval)
        else:
            return self._legitimate_features(agent, interval)

    def _legitimate_features(self, agent: Agent, interval: int) -> np.ndarray:
        """Legitimate agent features with temporal dynamics matched to training data."""
        return self._next_behavioral_vector(agent)

    def _adversarial_features(self, agent: Agent, interval: int) -> np.ndarray:
        """Adversarial agent features with explicit attack-type mapping."""
        atk = agent.attack_type
        if atk not in {
            "selective_packet_drop",
            "route_manipulation",
            "transmission_burst",
            "coordinated_insider",
            "low_and_slow",
        }:
            agent.attack_type = "coordinated_insider"
        return self._next_behavioral_vector(agent)

    # ------------------------------------------------------------------
    # Packet delivery simulation
    # ------------------------------------------------------------------

    def _simulate_interval_pdr(self, trust_scores: Dict[int, float],
                                 method: str) -> float:
        """
        Compute packet delivery ratio for one monitoring interval.

        Agents below trust threshold are isolated from forwarding for dynamic
        methods (proposed and bayesian) and packets are rerouted with a
        high-but-not-perfect success probability.
        """
        tau_min = self.mon_cfg["trust_threshold_tau_min"]
        packets_attempted = 0
        packets_delivered = 0
        load_estimate = self._estimate_network_load(method, trust_scores)

        for i, src in enumerate(self.agents):
            # Select a random destination agent
            dst_idx = self.rng.randint(0, len(self.agents) - 1)
            dst = self.agents[dst_idx]
            if dst_idx == i:
                continue

            packets_attempted += 1
            dist = self._distance(src, dst)

            # Under dynamic methods: exclude low-trust agents from forwarding.
            if method in ("proposed", "bayesian"):
                score = trust_scores.get(src.agent_id, 1.0)
                if score < tau_min:
                    # Isolation can help or hurt depending on false positives.
                    reroute_p = self._routing_success_probability(
                        load=load_estimate,
                        distance_m=dist,
                        channel_noise=self._sample_channel_noise(),
                    )
                    if self.rng.random() < reroute_p:
                        packets_delivered += 1
                    continue

            delivered = self.channel.delivery_success(
                dist, src.is_adversarial, src.attack_type, self.rng
            )

            if delivered:
                packets_delivered += 1

        if packets_attempted == 0:
            return 0.0
        return packets_delivered / packets_attempted

    def _sample_channel_noise(self) -> float:
        """Sample channel-noise intensity used by routing-success model."""
        mean = float(self.routing_cfg.get("noise_mean", 0.0))
        std = float(self.routing_cfg.get("noise_std", 0.1))
        return float(self.np_rng.normal(mean, std))

    def _estimate_network_load(self, method: str, trust_scores: Dict[int, float]) -> float:
        """Estimate normalized network load in [0,1] for current interval and method."""
        base_load = float(self.routing_cfg.get("base_load", 0.55))
        tau_min = float(self.mon_cfg["trust_threshold_tau_min"])

        if method in ("proposed", "bayesian") and trust_scores:
            isolated = sum(1 for s in trust_scores.values() if s < tau_min)
            isolated_ratio = isolated / max(len(trust_scores), 1)
            isolation_effect = float(self.routing_cfg.get("isolation_load_reduction", 0.25))
            load = base_load - isolation_effect * isolated_ratio
        else:
            load = base_load

        return float(np.clip(load, 0.0, 1.0))

    def _routing_success_probability(self, load: float, distance_m: float, channel_noise: float) -> float:
        """
        Compute bounded reroute success probability p_success = f(load, distance, noise).

        Higher load, longer distance, and higher noise reduce success.
        """
        ref_distance = float(self.routing_cfg.get("distance_ref_m", 1000.0))
        w_load = float(self.routing_cfg.get("w_load", 0.35))
        w_dist = float(self.routing_cfg.get("w_distance", 0.35))
        w_noise = float(self.routing_cfg.get("w_noise", 0.30))
        base_success = float(self.routing_cfg.get("base_success", 0.95))

        # Normalize factors to [0,1] and combine as penalties.
        load_n = float(np.clip(load, 0.0, 1.0))
        dist_n = float(np.clip(distance_m / max(ref_distance, 1e-6), 0.0, 1.0))
        noise_n = float(np.clip(abs(channel_noise), 0.0, 1.0))

        penalty = (w_load * load_n) + (w_dist * dist_n) + (w_noise * noise_n)
        p_success = base_success - penalty
        return float(np.clip(p_success, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Energy simulation (parametric model)
    # ------------------------------------------------------------------

    def _sample_packet_count(self, agent: Agent, method: str) -> float:
        """Sample per-interval packet volume for one agent."""
        base_packets = float(self.energy_cfg.get("base_packet_count", 1.2))
        type_mult = (
            float(self.energy_cfg.get("packet_multiplier_auv", 1.1))
            if agent.agent_type == "auv"
            else float(self.energy_cfg.get("packet_multiplier_sensor", 1.0))
        )
        method_mult = float(self.energy_cfg.get("packet_multiplier", {}).get(method, 1.0))
        noise_std = float(self.energy_cfg.get("packet_count_noise_std", 0.08))
        noisy = self.np_rng.normal(0.0, noise_std)
        return max(0.0, base_packets * type_mult * method_mult + noisy)

    def _sample_compute_cost(self, method: str) -> float:
        """Return per-interval compute cost term for selected trust method."""
        return float(self.energy_cfg.get("compute_cost", {}).get(method, 1.0))

    def _sample_communication_distance(self, agent: Agent) -> float:
        """Estimate interval communication distance (meters) using random peers."""
        peers_per_interval = int(self.energy_cfg.get("distance_sample_peers", 3))
        if len(self.agents) <= 1 or peers_per_interval <= 0:
            return 0.0

        peer_indices = self.np_rng.integers(0, len(self.agents), size=peers_per_interval)
        dists = []
        for idx in peer_indices:
            peer = self.agents[int(idx)]
            if peer.agent_id == agent.agent_id:
                continue
            dists.append(self._distance(agent, peer))

        if not dists:
            return 0.0
        return float(np.mean(dists))

    def _compute_interval_energy_drain_pct(
        self,
        packet_count: float,
        compute_cost: float,
        communication_distance: float,
    ) -> float:
        """
        Parametric drain model:
            drain_pct = alpha * packet_count + beta * compute_cost + gamma * communication_distance
        """
        alpha = float(self.energy_cfg.get("alpha", 1.35))
        beta = float(self.energy_cfg.get("beta", 0.75))
        gamma = float(self.energy_cfg.get("gamma", 0.00035))
        noise_std = float(self.energy_cfg.get("drain_noise_std_pct", 0.12))

        deterministic = (
            alpha * packet_count +
            beta * compute_cost +
            gamma * communication_distance
        )
        noisy = deterministic + self.np_rng.normal(0.0, noise_std)
        return max(0.0, float(noisy))

    def _simulate_energy_consumption(self, interval: int,
                                      method: str) -> Dict[int, float]:
        """Compute residual energy per agent after one monitoring interval."""

        energy_map = {}
        for agent in self.agents:
            packet_count = self._sample_packet_count(agent, method)
            compute_cost = self._sample_compute_cost(method)
            comm_distance = self._sample_communication_distance(agent)
            drain = self._compute_interval_energy_drain_pct(
                packet_count=packet_count,
                compute_cost=compute_cost,
                communication_distance=comm_distance,
            )
            new_bat = max(0.0, agent.battery - drain)
            energy_map[agent.agent_id] = new_bat
            agent.battery = new_bat

        return energy_map

    # ------------------------------------------------------------------
    # Bayesian trust baseline
    # ------------------------------------------------------------------

    def _bayesian_update(self, agent: Agent,
                         features: np.ndarray,
                         alpha_dict: Dict[int, float],
                         beta_dict: Dict[int, float]) -> float:
        """
        Stateful Beta-Bernoulli trust update.

        This accumulates evidence across intervals (instead of resampling
        independent pseudo-counts each call).
        """
        # No oracle label leakage: infer evidence only from observed behavior.
        anomaly_risk = float(np.clip(
            0.25 * features[0] +
            0.25 * features[1] +
            0.20 * (1.0 - features[2]) +
            0.15 * features[3] +
            0.15 * (1.0 - features[4]),
            0.0,
            1.0,
        ))
        evidence_noise = self.rng.uniform(-0.10, 0.10)
        posterior_failure_prob = float(np.clip(anomaly_risk + evidence_noise, 0.05, 0.95))
        observed_success = self.rng.random() > posterior_failure_prob

        if observed_success:
            alpha_dict[agent.agent_id] += 1.0
        else:
            beta_dict[agent.agent_id] += 1.0

        a = alpha_dict[agent.agent_id]
        b = beta_dict[agent.agent_id]
        return a / (a + b)

    def _build_padded_window(self, history: List[np.ndarray], seq_len: int = 64) -> np.ndarray:
        """Pad behavioral history to fixed sequence length for model inference."""
        if len(history) >= seq_len:
            return np.array(history[-seq_len:], dtype=np.float32)

        feat_dim = history[0].shape[0] if history else 5
        if history:
            pad_value = np.array(history[-1], dtype=np.float32)
            pad = np.repeat(pad_value[None, :], seq_len - len(history), axis=0)
        else:
            pad = np.zeros((seq_len, feat_dim), dtype=np.float32)
        if history:
            return np.vstack([np.array(history, dtype=np.float32), pad])
        return pad

    def _compute_detection_metrics(self, trust_scores: Dict[int, float],
                                   tau_threshold: float) -> Dict[str, float]:
        """Compute full confusion matrix and derived detection metrics."""
        tp = tn = fp = fn = 0
        for agent in self.agents:
            score = trust_scores.get(agent.agent_id, 1.0)
            predicted_anomalous = score < tau_threshold
            if predicted_anomalous and agent.is_adversarial:
                tp += 1
            elif predicted_anomalous and (not agent.is_adversarial):
                fp += 1
            elif (not predicted_anomalous) and agent.is_adversarial:
                fn += 1
            else:
                tn += 1

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": float(tp),
            "tn": float(tn),
            "fp": float(fp),
            "fn": float(fn),
        }

    def _is_degenerate_trust(self, trust_scores: Dict[int, float]) -> bool:
        """Detect collapsed trust distributions that indicate invalid intervals."""
        vals = np.array(list(trust_scores.values()), dtype=np.float32)
        if vals.size == 0:
            return True
        spread = float(vals.max() - vals.min())
        std = float(vals.std())
        return (spread < 1e-3) or (std < 1e-4)

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------

    def run(self, num_intervals: int = 20,
            transformer_trust_fn=None,
            sequence_len: int = 64,
            log_trust_stats: bool = False,
            enforce_non_degenerate: bool = True) -> Dict:
        """
        Run the full simulation for a given number of monitoring intervals.

        Args:
            num_intervals: Number of 30-second monitoring intervals to simulate.
            transformer_trust_fn: Callable(sequence_window) -> trust_score.
                Receives one (K,5) window per agent. If None, uses heuristic.
            sequence_len: Temporal window length K for transformer scoring.

        Returns:
            Dictionary with per-interval metrics for all three methods.
        """
        results = {
            "interval": [],
            "accuracy_proposed": [],
            "accuracy_bayesian": [],
            "accuracy_static": [],
            "precision_proposed": [],
            "precision_bayesian": [],
            "precision_static": [],
            "recall_proposed": [],
            "recall_bayesian": [],
            "recall_static": [],
            "f1_proposed": [],
            "f1_bayesian": [],
            "f1_static": [],
            "tp_proposed": [],
            "tp_bayesian": [],
            "tp_static": [],
            "tn_proposed": [],
            "tn_bayesian": [],
            "tn_static": [],
            "fp_proposed": [],
            "fp_bayesian": [],
            "fp_static": [],
            "fn_proposed": [],
            "fn_bayesian": [],
            "fn_static": [],
            "pdr_proposed": [],
            "pdr_bayesian": [],
            "pdr_static": [],
            "energy_proposed": [],
            "energy_bayesian": [],
            "energy_static": [],
            "trust_mean_proposed": [],
            "trust_std_proposed": [],
            "trust_min_proposed": [],
            "trust_max_proposed": [],
            "blockchain_commit_latency_ms": [],
            "blockchain_failed_commits": [],
            "blockchain_commit_success_rate": [],
        }

        initial_battery = float(self.energy_cfg.get("initial_battery_pct", 100.0))

        # Reset agent batteries at start of run
        for agent in self.agents:
            agent.battery = initial_battery

        # Three independent battery trajectories — one per method
        # Each accumulates drain across intervals independently.
        self._batt_proposed = {a.agent_id: initial_battery for a in self.agents}
        self._batt_bayesian = {a.agent_id: initial_battery for a in self.agents}
        self._batt_static   = {a.agent_id: initial_battery for a in self.agents}

        # Running smoothed trust scores for proposed method
        smoothed_trust = {a.agent_id: 1.0 for a in self.agents}
        alpha_smooth = self.mon_cfg["smoothing_factor_alpha"]
        tau_min = self.mon_cfg["trust_threshold_tau_min"]
        prop_inference_noise_std = 0.07

        # Bayesian prior parameters
        bay_alpha = self.config["baselines"]["bayesian_prior_alpha"]
        bay_beta = self.config["baselines"]["bayesian_prior_beta"]
        bay_alpha_dict = {a.agent_id: float(bay_alpha) for a in self.agents}
        bay_beta_dict = {a.agent_id: float(bay_beta) for a in self.agents}

        # Per-agent temporal feature history for sequence model inference
        feature_history: Dict[int, List[np.ndarray]] = {a.agent_id: [] for a in self.agents}

        blockchain_enabled = bool(self.blockchain_cfg.get("enabled", True))
        consortium = None
        if blockchain_enabled:
            consortium = PBFTConsortium(
                num_validators=int(self.blockchain_cfg.get("num_validators", 9)),
                num_byzantine=int(self.blockchain_cfg.get("num_byzantine", 0)),
                seed=int(self.blockchain_cfg.get("seed", self.seed)),
                network_latency_ms=float(self.blockchain_cfg.get("network_latency_ms", 500.0)),
            )

        # Reset temporal behavioral states at run start.
        for agent in self.agents:
            self._feature_state[agent.agent_id] = self._feature_mean[agent.agent_id].copy()

        for interval in range(1, num_intervals + 1):
            self._move_agents()

            # ---- Proposed: transformer trust inference ----
            prop_trust = {}
            for agent in self.agents:
                feats = self._generate_behavioral_features(agent, interval)
                feature_history[agent.agent_id].append(feats)

                if transformer_trust_fn is not None:
                    seq_window = self._build_padded_window(
                        feature_history[agent.agent_id],
                        seq_len=sequence_len,
                    )
                    raw_score = float(transformer_trust_fn(seq_window))
                else:
                    # Heuristic: weighted combination of feature semantics
                    # Higher timing_var + retx_freq → lower trust
                    # Higher routing_stab + protocol_comp → higher trust
                    raw_score = (
                        0.2 * (1 - feats[0]) +   # timing variance (inverse)
                        0.2 * (1 - feats[1]) +   # retx frequency (inverse)
                        0.2 * feats[2] +          # routing stability
                        0.2 * (1 - feats[3]) +   # neighbor churn (inverse)
                        0.2 * feats[4]            # protocol compliance
                    )
                    raw_score = float(np.clip(raw_score, 0.0, 1.0))

                # Exponential smoothing (Equation 6 in paper)
                smoothed = (alpha_smooth * smoothed_trust[agent.agent_id] +
                            (1 - alpha_smooth) * raw_score)
                smoothed_trust[agent.agent_id] = smoothed
                prop_trust[agent.agent_id] = smoothed

            if transformer_trust_fn is not None and log_trust_stats:
                trust_vals = np.array(list(prop_trust.values()), dtype=np.float32)
                print(
                    f"[interval {interval:02d}] proposed trust: "
                    f"mean={trust_vals.mean():.3f} std={trust_vals.std():.3f} "
                    f"min={trust_vals.min():.3f} max={trust_vals.max():.3f}"
                )

            if enforce_non_degenerate and self._is_degenerate_trust(prop_trust):
                raise RuntimeError(
                    f"Degenerate trust distribution at interval {interval}. "
                    "Rejecting run to preserve scientific validity."
                )

            interval_tau = float(np.clip(
                tau_min + self.np_rng.normal(0.0, 0.035),
                0.55,
                0.78,
            ))

            # Proposed trust inference uncertainty from channel/feature noise.
            prop_eval_trust = {
                agent_id: float(np.clip(score + self.np_rng.normal(0.0, prop_inference_noise_std), 0.0, 1.0))
                for agent_id, score in prop_trust.items()
            }

            # ---- Detection metrics: proposed ----
            prop_metrics = self._compute_detection_metrics(prop_eval_trust, interval_tau)

            # ---- Blockchain trust-delta commits (proposed path) ----
            interval_commit_latency_ms = 0.0
            interval_failed_commits = 0
            interval_commit_success_rate = 0.0

            if consortium is not None:
                lat_start = len(consortium.commit_latencies_ms)
                rej_start = consortium.total_rejections
                commits_attempted = 0
                commits_success = 0

                for a in self.agents:
                    trust_score = float(prop_eval_trust.get(a.agent_id, 1.0))
                    rec = TrustRecord(
                        agent_id=f"agent_{a.agent_id:03d}",
                        trust_score=round(trust_score, 6),
                        interval=interval,
                        is_anomalous=bool(trust_score < interval_tau),
                        attack_type=str(a.attack_type) if a.attack_type else "none",
                        timestamp=float(self.np_rng.uniform(0.0, 1.0) + interval),
                        committed_by="interrogator_node_0",
                    )
                    commits_attempted += 1
                    if consortium.submit_trust_delta(rec):
                        commits_success += 1

                latencies = consortium.commit_latencies_ms[lat_start:]
                interval_commit_latency_ms = float(np.mean(latencies)) if latencies else 0.0
                interval_failed_commits = int(consortium.total_rejections - rej_start)
                interval_commit_success_rate = (
                    commits_success / commits_attempted if commits_attempted > 0 else 0.0
                )

            # ---- Bayesian trust baseline ----
            bay_trust = {}
            for a in self.agents:
                feats_latest = feature_history[a.agent_id][-1]
                bay_trust[a.agent_id] = self._bayesian_update(
                    a,
                    feats_latest,
                    bay_alpha_dict,
                    bay_beta_dict,
                )
            bay_metrics = self._compute_detection_metrics(bay_trust, interval_tau)

            # ---- Static trust: no detection capability ----
            static_trust = {a.agent_id: 1.0 for a in self.agents}
            static_metrics = self._compute_detection_metrics(static_trust, interval_tau)

            # ---- PDR ----
            pdr_prop = self._simulate_interval_pdr(prop_eval_trust, "proposed")
            pdr_bay = self._simulate_interval_pdr(bay_trust, "bayesian")
            pdr_static = self._simulate_interval_pdr(static_trust, "static")

            # ---- Energy tracking ----
            # Three independent battery trajectories — one per method.
            # _batt_proposed / _batt_bayesian / _batt_static are initialised
            # at run() start and updated each interval so all three curves
            # deplete correctly over time without interfering with each other.

            # Proposed trajectory drain
            for a in self.agents:
                a.battery = self._batt_proposed[a.agent_id]
            self._simulate_energy_consumption(interval, "proposed")
            energy_prop = np.mean([a.battery for a in self.agents])
            self._batt_proposed = {a.agent_id: a.battery for a in self.agents}

            # Bayesian trajectory drain
            for a in self.agents:
                a.battery = self._batt_bayesian[a.agent_id]
            self._simulate_energy_consumption(interval, "bayesian")
            energy_bay = np.mean([a.battery for a in self.agents])
            self._batt_bayesian = {a.agent_id: a.battery for a in self.agents}

            # Static trajectory drain
            for a in self.agents:
                a.battery = self._batt_static[a.agent_id]
            self._simulate_energy_consumption(interval, "static")
            energy_static = np.mean([a.battery for a in self.agents])
            self._batt_static = {a.agent_id: a.battery for a in self.agents}

            # Restore agents to proposed battery state for next iteration
            for a in self.agents:
                a.battery = self._batt_proposed[a.agent_id]

            # ---- Record ----
            trust_vals = np.array(list(prop_trust.values()), dtype=np.float32)
            results["interval"].append(interval)
            results["accuracy_proposed"].append(round(prop_metrics["accuracy"] * 100, 2))
            results["accuracy_bayesian"].append(round(bay_metrics["accuracy"] * 100, 2))
            results["accuracy_static"].append(round(static_metrics["accuracy"] * 100, 2))
            results["precision_proposed"].append(round(prop_metrics["precision"] * 100, 2))
            results["precision_bayesian"].append(round(bay_metrics["precision"] * 100, 2))
            results["precision_static"].append(round(static_metrics["precision"] * 100, 2))
            results["recall_proposed"].append(round(prop_metrics["recall"] * 100, 2))
            results["recall_bayesian"].append(round(bay_metrics["recall"] * 100, 2))
            results["recall_static"].append(round(static_metrics["recall"] * 100, 2))
            results["f1_proposed"].append(round(prop_metrics["f1"] * 100, 2))
            results["f1_bayesian"].append(round(bay_metrics["f1"] * 100, 2))
            results["f1_static"].append(round(static_metrics["f1"] * 100, 2))
            results["tp_proposed"].append(int(prop_metrics["tp"]))
            results["tp_bayesian"].append(int(bay_metrics["tp"]))
            results["tp_static"].append(int(static_metrics["tp"]))
            results["tn_proposed"].append(int(prop_metrics["tn"]))
            results["tn_bayesian"].append(int(bay_metrics["tn"]))
            results["tn_static"].append(int(static_metrics["tn"]))
            results["fp_proposed"].append(int(prop_metrics["fp"]))
            results["fp_bayesian"].append(int(bay_metrics["fp"]))
            results["fp_static"].append(int(static_metrics["fp"]))
            results["fn_proposed"].append(int(prop_metrics["fn"]))
            results["fn_bayesian"].append(int(bay_metrics["fn"]))
            results["fn_static"].append(int(static_metrics["fn"]))
            results["pdr_proposed"].append(round(pdr_prop * 100, 2))
            results["pdr_bayesian"].append(round(pdr_bay * 100, 2))
            results["pdr_static"].append(round(pdr_static * 100, 2))
            results["energy_proposed"].append(round(energy_prop, 2))
            results["energy_bayesian"].append(round(energy_bay, 2))
            results["energy_static"].append(round(energy_static, 2))
            results["trust_mean_proposed"].append(round(float(trust_vals.mean()), 4))
            results["trust_std_proposed"].append(round(float(trust_vals.std()), 4))
            results["trust_min_proposed"].append(round(float(trust_vals.min()), 4))
            results["trust_max_proposed"].append(round(float(trust_vals.max()), 4))
            results["blockchain_commit_latency_ms"].append(round(interval_commit_latency_ms, 4))
            results["blockchain_failed_commits"].append(int(interval_failed_commits))
            results["blockchain_commit_success_rate"].append(round(float(interval_commit_success_rate), 4))

        return results

    def _compute_accuracy(self, trust_scores: Dict[int, float],
                           tau_min: float) -> float:
        """Compute anomaly detection accuracy: fraction of correct trust decisions."""
        correct = 0
        total = len(self.agents)
        for agent in self.agents:
            score = trust_scores.get(agent.agent_id, 1.0)
            predicted_anomalous = score < tau_min
            if predicted_anomalous == agent.is_adversarial:
                correct += 1
        return correct / total

    def _static_accuracy(self) -> float:
        """
        Static trust always predicts 'legitimate'; accuracy = fraction of
        non-adversarial agents (since it never detects adversarial ones).
        """
        n_legit = sum(1 for a in self.agents if not a.is_adversarial)
        return n_legit / len(self.agents)
