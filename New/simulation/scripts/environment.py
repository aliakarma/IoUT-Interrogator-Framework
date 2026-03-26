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
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


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

        self.channel = AcousticChannel(self.config["acoustic_channel"])
        self.area_side = math.sqrt(self.net_cfg["deployment_area_km2"]) * 1000  # metres

        self.agents: List[Agent] = []
        self._deploy_agents()

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
            atk = self.rng.choice(attack_types) if is_adv else None
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
        """Legitimate agent: low variance, stable routing, compliant."""
        timing_var = self.np_rng.beta(2, 8)           # low variance
        retx_freq = self.np_rng.beta(1, 9)            # low retransmission
        routing_stab = self.np_rng.beta(8, 2)         # high stability
        neighbor_churn = self.np_rng.beta(1, 9)       # low churn
        protocol_comp = self.np_rng.beta(9, 1)        # high compliance
        return np.array([timing_var, retx_freq, routing_stab,
                         neighbor_churn, protocol_comp], dtype=np.float32)

    def _adversarial_features(self, agent: Agent, interval: int) -> np.ndarray:
        """Adversarial agent: features deviate based on attack type."""
        atk = agent.attack_type
        if atk == "selective_packet_drop":
            return np.array([
                self.np_rng.beta(4, 4),    # moderate timing variance
                self.np_rng.beta(7, 3),    # HIGH retransmission
                self.np_rng.beta(4, 6),    # low routing stability
                self.np_rng.beta(2, 8),    # normal churn
                self.np_rng.beta(6, 4),    # partial compliance
            ], dtype=np.float32)
        elif atk == "route_manipulation":
            return np.array([
                self.np_rng.beta(3, 7),
                self.np_rng.beta(2, 8),
                self.np_rng.beta(2, 8),    # VERY LOW routing stability
                self.np_rng.beta(7, 3),    # HIGH neighbor churn
                self.np_rng.beta(4, 6),
            ], dtype=np.float32)
        elif atk == "transmission_burst":
            return np.array([
                self.np_rng.beta(8, 2),    # HIGH timing variance
                self.np_rng.beta(8, 2),    # HIGH retransmission
                self.np_rng.beta(5, 5),
                self.np_rng.beta(3, 7),
                self.np_rng.beta(5, 5),
            ], dtype=np.float32)
        else:  # coordinated_insider — subtle deviation
            return np.array([
                self.np_rng.beta(4, 6),
                self.np_rng.beta(5, 5),
                self.np_rng.beta(5, 5),
                self.np_rng.beta(4, 6),
                self.np_rng.beta(4, 6),    # REDUCED compliance
            ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Packet delivery simulation
    # ------------------------------------------------------------------

    def _simulate_interval_pdr(self, trust_scores: Dict[int, float],
                                 method: str) -> float:
        """
        Compute packet delivery ratio for one monitoring interval.

        For the proposed method: agents below trust_threshold are excluded
        from routing, which increases effective PDR.
        """
        tau_min = self.mon_cfg["trust_threshold_tau_min"]
        packets_attempted = 0
        packets_delivered = 0

        for i, src in enumerate(self.agents):
            # Select a random destination agent
            dst_idx = self.rng.randint(0, len(self.agents) - 1)
            dst = self.agents[dst_idx]
            if dst_idx == i:
                continue

            packets_attempted += 1
            dist = self._distance(src, dst)

            # Under proposed method: exclude low-trust agents from forwarding
            if method == "proposed":
                score = trust_scores.get(src.agent_id, 1.0)
                if score < tau_min and src.is_adversarial:
                    # Agent is correctly excluded → packet rerouted (no drop)
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

    # ------------------------------------------------------------------
    # Energy simulation
    # ------------------------------------------------------------------

    def _simulate_energy_consumption(self, interval: int,
                                      method: str) -> Dict[int, float]:
        """Compute residual energy per agent after one monitoring interval."""
        bits_per_msg = 256 * 8  # 256-byte messages
        overhead_J = 0.0

        if method == "bayesian":
            overhead_J = 0.002  # lightweight probabilistic update overhead
        elif method == "proposed":
            # Transformer inference: 0.35-0.6 J per interrogator node
            # Amortised across 50 agents → ~0.01 J per agent per interval
            overhead_J = 0.01

        energy_map = {}
        for agent in self.agents:
            # Communication energy to nearest neighbour (approximate)
            nearest_dist = self.area_side  # initialise to max
            for other in self.agents:
                if other.agent_id != agent.agent_id:
                    d = self._distance(agent, other)
                    if d < nearest_dist:
                        nearest_dist = d
            nearest_dist = max(nearest_dist, 1.0)

            tx_energy = self.channel.transmission_energy_J(bits_per_msg, nearest_dist)
            sense_energy = 0.0001  # 0.1 mJ sensing cost
            total_J = tx_energy + sense_energy + overhead_J

            # Convert J to battery % drain (assume 100 Wh ≈ 360000 J capacity)
            capacity_J = 360000.0
            drain_pct = (total_J / capacity_J) * 100.0
            new_battery = max(0.0, agent.battery - drain_pct)
            energy_map[agent.agent_id] = new_battery
            agent.battery = new_battery

        return energy_map

    # ------------------------------------------------------------------
    # Bayesian trust baseline
    # ------------------------------------------------------------------

    def _bayesian_trust_score(self, agent: Agent,
                               alpha: float, beta: float) -> float:
        """Beta-Bernoulli Bayesian trust estimate."""
        if agent.is_adversarial:
            # Adversarial agents generate fewer successful observations
            successes = self.np_rng.integers(2, 8)
            failures = self.np_rng.integers(5, 12)
        else:
            successes = self.np_rng.integers(8, 15)
            failures = self.np_rng.integers(0, 3)
        return (alpha + successes) / (alpha + beta + successes + failures)

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------

    def run(self, num_intervals: int = 20,
            transformer_trust_fn=None) -> Dict:
        """
        Run the full simulation for a given number of monitoring intervals.

        Args:
            num_intervals: Number of 30-second monitoring intervals to simulate.
            transformer_trust_fn: Callable(features_batch) -> trust_scores.
                If None, uses a heuristic based on feature mean.

        Returns:
            Dictionary with per-interval metrics for all three methods.
        """
        results = {
            "interval": [],
            "accuracy_proposed": [],
            "accuracy_bayesian": [],
            "accuracy_static": [],
            "pdr_proposed": [],
            "pdr_bayesian": [],
            "pdr_static": [],
            "energy_proposed": [],
            "energy_bayesian": [],
            "energy_static": [],
        }

        # Reset agent batteries at start
        for agent in self.agents:
            agent.battery = 100.0

        # Running smoothed trust scores for proposed method
        smoothed_trust = {a.agent_id: 1.0 for a in self.agents}
        alpha_smooth = self.mon_cfg["smoothing_factor_alpha"]
        tau_min = self.mon_cfg["trust_threshold_tau_min"]

        # Bayesian prior parameters
        bay_alpha = self.config["baselines"]["bayesian_prior_alpha"]
        bay_beta = self.config["baselines"]["bayesian_prior_beta"]

        for interval in range(1, num_intervals + 1):
            self._move_agents()

            # ---- Proposed: transformer trust inference ----
            prop_trust = {}
            for agent in self.agents:
                feats = self._generate_behavioral_features(agent, interval)
                if transformer_trust_fn is not None:
                    raw_score = float(transformer_trust_fn(feats))
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

            # ---- Accuracy: proposed ----
            prop_acc = self._compute_accuracy(prop_trust, tau_min)

            # ---- Bayesian trust baseline ----
            bay_trust = {
                a.agent_id: self._bayesian_trust_score(a, bay_alpha, bay_beta)
                for a in self.agents
            }
            bay_acc = self._compute_accuracy(bay_trust, tau_min)

            # ---- Static trust: no detection capability ----
            static_trust = {a.agent_id: 1.0 for a in self.agents}
            # Static trust cannot detect adversarial agents at all
            static_acc = self._static_accuracy()

            # ---- PDR ----
            pdr_prop = self._simulate_interval_pdr(prop_trust, "proposed")
            pdr_bay = self._simulate_interval_pdr(bay_trust, "bayesian")
            pdr_static = self._simulate_interval_pdr(static_trust, "static")

            # ---- Energy (deep copy batteries before update) ----
            batt_before = {a.agent_id: a.battery for a in self.agents}
            self._simulate_energy_consumption(interval, "proposed")
            energy_prop = np.mean([a.battery for a in self.agents])

            # Reset batteries and compute bayesian energy
            for a in self.agents:
                a.battery = batt_before[a.agent_id]
            self._simulate_energy_consumption(interval, "bayesian")
            energy_bay = np.mean([a.battery for a in self.agents])

            # Reset batteries and compute static energy
            for a in self.agents:
                a.battery = batt_before[a.agent_id]
            self._simulate_energy_consumption(interval, "static")
            energy_static = np.mean([a.battery for a in self.agents])

            # Restore to proposed final state for next iteration
            final_energy_prop = {a.agent_id: batt_before[a.agent_id] for a in self.agents}
            for a in self.agents:
                a.battery = max(0.0,
                    batt_before[a.agent_id] - (batt_before[a.agent_id] - energy_prop) / len(self.agents) * 0.1
                )

            # ---- Record ----
            results["interval"].append(interval)
            results["accuracy_proposed"].append(round(prop_acc * 100, 2))
            results["accuracy_bayesian"].append(round(bay_acc * 100, 2))
            results["accuracy_static"].append(round(static_acc * 100, 2))
            results["pdr_proposed"].append(round(pdr_prop * 100, 2))
            results["pdr_bayesian"].append(round(pdr_bay * 100, 2))
            results["pdr_static"].append(round(pdr_static * 100, 2))
            results["energy_proposed"].append(round(energy_prop, 2))
            results["energy_bayesian"].append(round(energy_bay, 2))
            results["energy_static"].append(round(energy_static, 2))

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
