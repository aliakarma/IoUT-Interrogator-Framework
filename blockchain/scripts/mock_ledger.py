"""
Mock PBFT Blockchain Trust Ledger
===================================
A lightweight Python simulation of the permissioned blockchain consortium
described in the paper (Section IV.D). Implements PBFT-style consensus
among N validator nodes without requiring Hyperledger Fabric.

This mock is used for:
  - Notebook demos (notebooks/03_blockchain_demo.ipynb)
  - Integration testing
  - Demonstrating trust-delta commit and enforcement workflows

Real deployment uses the Go chaincode in blockchain/chaincode/trust_ledger.go.
"""

import hashlib
import json
import time
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrustRecord:
    agent_id:     str
    trust_score:  float
    interval:     int
    is_anomalous: bool
    attack_type:  str
    timestamp:    float
    committed_by: str
    tx_hash:      str = ""

    def __post_init__(self):
        if not self.tx_hash:
            payload = f"{self.agent_id}{self.trust_score}{self.interval}{self.timestamp}"
            self.tx_hash = hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class EnforcementEvent:
    agent_id:          str
    tier:              int      # 1, 2, or 3
    action:            str      # "routing_exclusion", "throttle", "isolate"
    validated_by_pbft: bool
    timestamp:         float
    tx_hash:           str = ""

    def __post_init__(self):
        if not self.tx_hash:
            payload = f"{self.agent_id}{self.tier}{self.action}{self.timestamp}"
            self.tx_hash = hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Validator node
# ---------------------------------------------------------------------------

class ValidatorNode:
    """Simulates a single PBFT validator node."""

    def __init__(self, node_id: str, is_byzantine: bool = False):
        self.node_id = node_id
        self.is_byzantine = is_byzantine
        self.ledger: List[dict] = []

    def prepare_vote(self, proposal: dict) -> Tuple[bool, str]:
        """
        Cast a PREPARE vote on a proposed transaction.
        Byzantine nodes vote randomly; honest nodes validate the record.
        """
        if self.is_byzantine:
            vote = random.random() > 0.5
            return vote, self.node_id

        # Honest validation for trust-delta proposals
        trust_required = {"agent_id", "trust_score", "interval", "timestamp"}
        if trust_required.issubset(proposal.keys()):
            return True, self.node_id

        # Honest validation for enforcement proposals
        enforcement_required = {
            "agent_id", "tier", "action", "validated_by_pbft", "timestamp"
        }
        if enforcement_required.issubset(proposal.keys()):
            tier = int(proposal["tier"])
            if tier == 1:
                valid = True
            elif tier in (2, 3):
                valid = bool(proposal["validated_by_pbft"])
            else:
                valid = False
            return valid, self.node_id

        valid = False
        return valid, self.node_id

    def commit(self, record: dict):
        """Commit a validated record to local ledger replica."""
        self.ledger.append(record)


# ---------------------------------------------------------------------------
# PBFT Consortium
# ---------------------------------------------------------------------------

class PBFTConsortium:
    """
    Simulates a PBFT-based permissioned blockchain consortium.

    With n=9 validators, tolerates f=floor((9-1)/3)=2 Byzantine faults.
    Consensus requires 2f+1=5 matching PREPARE votes.

    Paper reference: Section IV.D and Theoretical Justification Section V.C.
    """

    def __init__(self, num_validators: int = 9,
                 num_byzantine: int = 0,
                 seed: int = 42):
        random.seed(seed)
        self.num_validators = num_validators
        self.f = (num_validators - 1) // 3      # max tolerated byzantine
        self.quorum = 2 * self.f + 1             # votes needed for consensus

        # Initialise validator nodes
        byzantine_ids = set(random.sample(range(num_validators), num_byzantine))
        self.validators = [
            ValidatorNode(
                node_id=f"validator_{i}",
                is_byzantine=(i in byzantine_ids)
            )
            for i in range(num_validators)
        ]

        # Global ledger (primary replica)
        self.ledger: List[dict] = []
        self.enforcement_log: List[dict] = []

        # Statistics
        self.total_commits = 0
        self.total_rejections = 0
        self.commit_latencies_ms: List[float] = []

    # -----------------------------------------------------------------------
    # Core PBFT consensus
    # -----------------------------------------------------------------------

    def _pbft_consensus(self, proposal: dict) -> bool:
        """
        Simulate PBFT 3-phase protocol (pre-prepare → prepare → commit).

        Returns True if quorum is reached (2f+1 agreeing votes).
        O(n^2) messages in real PBFT; simulated here for lightweight demo.
        """
        t0 = time.time()

        # Phase 1: Pre-prepare (primary broadcasts proposal)
        # Phase 2: Prepare (validators vote)
        votes = [node.prepare_vote(proposal) for node in self.validators]
        yes_votes = sum(1 for v, _ in votes if v)

        # Phase 3: Commit (if quorum reached)
        consensus_reached = yes_votes >= self.quorum

        latency_ms = (time.time() - t0) * 1000
        self.commit_latencies_ms.append(latency_ms)

        return consensus_reached

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def submit_trust_delta(self, record: TrustRecord) -> bool:
        """
        Submit a trust delta record for PBFT consensus.

        Called by interrogator nodes when an agent's trust score crosses
        tau_min = 0.65 (Equation 7 in paper).

        Returns True if the record was committed.
        """
        proposal = asdict(record)
        if self._pbft_consensus(proposal):
            committed = asdict(record)
            committed["committed_at"] = time.time()
            self.ledger.append(committed)
            for v in self.validators:
                v.commit(committed)
            self.total_commits += 1
            return True
        else:
            self.total_rejections += 1
            return False

    def record_enforcement(self, event: EnforcementEvent) -> bool:
        """
        Record a tiered enforcement decision via consortium consensus.
        Only Tier-2 and Tier-3 decisions require PBFT validation.
        Tier-1 (local constraint) is logged without consensus.
        """
        proposal = asdict(event)
        if event.tier == 1:
            # Tier-1: immediate local enforcement, no PBFT needed
            self.enforcement_log.append(proposal)
            return True
        if event.tier not in (2, 3):
            return False
        if not event.validated_by_pbft:
            return False
        if self._pbft_consensus(proposal):
            self.enforcement_log.append(proposal)
            return True
        return False

    def get_agent_trust_history(self, agent_id: str) -> List[dict]:
        """Retrieve all trust records for a given agent."""
        return [r for r in self.ledger if r["agent_id"] == agent_id]

    def get_latest_trust_score(self, agent_id: str) -> Optional[float]:
        """Return the most recently committed trust score for an agent."""
        history = self.get_agent_trust_history(agent_id)
        if not history:
            return None
        latest = max(history, key=lambda r: r["interval"])
        return latest["trust_score"]

    def get_enforcement_history(self, agent_id: str) -> List[dict]:
        """Retrieve enforcement events for a given agent."""
        return [e for e in self.enforcement_log if e["agent_id"] == agent_id]

    def ledger_stats(self) -> dict:
        """Return ledger statistics."""
        mean_latency = (sum(self.commit_latencies_ms) / len(self.commit_latencies_ms)
                        if self.commit_latencies_ms else 0.0)
        return {
            "total_records":      len(self.ledger),
            "total_commits":      self.total_commits,
            "total_rejections":   self.total_rejections,
            "enforcement_events": len(self.enforcement_log),
            "mean_latency_ms":    round(mean_latency, 4),
            "num_validators":     self.num_validators,
            "fault_tolerance_f":  self.f,
            "quorum_required":    self.quorum,
        }

    def export_ledger(self, path: str):
        """Export ledger to JSON file."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "ledger":          self.ledger,
                "enforcement_log": self.enforcement_log,
                "stats":           self.ledger_stats(),
            }, f, indent=2)
        print(f"Ledger exported to: {path}")


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------

def run_demo(num_intervals: int = 5, num_agents: int = 10,
             adv_fraction: float = 0.3, seed: int = 42):
    """
    Quick demo of the mock PBFT ledger with synthetic trust scores.
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    consortium = PBFTConsortium(num_validators=9, num_byzantine=0, seed=seed)
    tau_min = 0.65

    n_adv = int(num_agents * adv_fraction)
    adv_ids = set(range(n_adv))

    print(f"=== Mock PBFT Ledger Demo ===")
    print(f"  Validators: {consortium.num_validators}  "
          f"(f={consortium.f}, quorum={consortium.quorum})")
    print(f"  Agents: {num_agents}  ({n_adv} adversarial)")
    print(f"  Intervals: {num_intervals}\n")

    for interval in range(1, num_intervals + 1):
        for agent_id in range(num_agents):
            is_adv = agent_id in adv_ids
            # Simulate trust score (adversarial agents tend below tau_min)
            if is_adv:
                score = float(rng.beta(3, 7))   # skewed low
            else:
                score = float(rng.beta(8, 2))   # skewed high

            if score < tau_min:
                record = TrustRecord(
                    agent_id=f"agent_{agent_id:03d}",
                    trust_score=round(score, 4),
                    interval=interval,
                    is_anomalous=True,
                    attack_type="selective_packet_drop" if is_adv else "false_positive",
                    timestamp=time.time(),
                    committed_by="interrogator_node_0",
                )
                committed = consortium.submit_trust_delta(record)

                # Tier-1 immediate enforcement
                evt = EnforcementEvent(
                    agent_id=f"agent_{agent_id:03d}",
                    tier=1,
                    action="routing_exclusion",
                    validated_by_pbft=False,
                    timestamp=time.time(),
                )
                consortium.record_enforcement(evt)

                # Tier-2 PBFT enforcement if persistent
                if interval > 2:
                    evt2 = EnforcementEvent(
                        agent_id=f"agent_{agent_id:03d}",
                        tier=2,
                        action="throttle",
                        validated_by_pbft=True,
                        timestamp=time.time(),
                    )
                    consortium.record_enforcement(evt2)

    stats = consortium.ledger_stats()
    print("Ledger Statistics:")
    for k, v in stats.items():
        print(f"  {k:<30} {v}")

    return consortium


if __name__ == "__main__":
    consortium = run_demo()
    consortium.export_ledger("data/processed/ledger_demo.json")
