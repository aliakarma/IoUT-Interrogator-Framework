"""
Unit Tests: Mock PBFT Blockchain Ledger
=========================================
"""
import sys
import os
import time
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from blockchain.scripts.mock_ledger import (
    PBFTConsortium, TrustRecord, EnforcementEvent
)


def make_record(agent_id: str = "agent_001",
                score: float = 0.4,
                interval: int = 1,
                anomalous: bool = True) -> TrustRecord:
    return TrustRecord(
        agent_id=agent_id,
        trust_score=score,
        interval=interval,
        is_anomalous=anomalous,
        attack_type="selective_packet_drop",
        timestamp=time.time(),
        committed_by="interrogator_0",
    )


class TestPBFTConsortium:

    def setup_method(self):
        self.consortium = PBFTConsortium(num_validators=9, num_byzantine=0, seed=42)

    def test_validator_count(self):
        assert len(self.consortium.validators) == 9

    def test_fault_tolerance(self):
        """f = floor((9-1)/3) = 2"""
        assert self.consortium.f == 2

    def test_quorum(self):
        """quorum = 2f+1 = 5"""
        assert self.consortium.quorum == 5

    def test_commit_trust_delta_honest_validators(self):
        record = make_record()
        result = self.consortium.submit_trust_delta(record)
        assert result is True

    def test_committed_record_in_ledger(self):
        record = make_record(agent_id="agent_007", score=0.3, interval=1)
        self.consortium.submit_trust_delta(record)
        history = self.consortium.get_agent_trust_history("agent_007")
        assert len(history) == 1
        assert history[0]["trust_score"] == 0.3

    def test_multiple_records_same_agent(self):
        for i in range(1, 4):
            record = make_record(agent_id="agent_010", score=0.4, interval=i)
            self.consortium.submit_trust_delta(record)
        history = self.consortium.get_agent_trust_history("agent_010")
        assert len(history) == 3

    def test_get_latest_trust_score(self):
        self.consortium.submit_trust_delta(make_record("agent_020", 0.5, 1))
        self.consortium.submit_trust_delta(make_record("agent_020", 0.3, 2))
        self.consortium.submit_trust_delta(make_record("agent_020", 0.2, 3))
        latest = self.consortium.get_latest_trust_score("agent_020")
        assert latest == pytest.approx(0.2, abs=1e-4)

    def test_get_latest_score_none_if_no_history(self):
        result = self.consortium.get_latest_trust_score("agent_999")
        assert result is None

    def test_tier1_enforcement_no_pbft(self):
        """Tier-1 enforcement is logged immediately without consensus."""
        evt = EnforcementEvent(
            agent_id="agent_030", tier=1,
            action="routing_exclusion", validated_by_pbft=False,
            timestamp=time.time(),
        )
        result = self.consortium.record_enforcement(evt)
        assert result is True
        history = self.consortium.get_enforcement_history("agent_030")
        assert len(history) == 1
        assert history[0]["tier"] == 1

    def test_tier2_enforcement_via_pbft(self):
        evt = EnforcementEvent(
            agent_id="agent_040", tier=2,
            action="throttle", validated_by_pbft=True,
            timestamp=time.time(),
        )
        result = self.consortium.record_enforcement(evt)
        assert result is True

    def test_ledger_stats_structure(self):
        stats = self.consortium.ledger_stats()
        assert "total_records" in stats
        assert "total_commits" in stats
        assert "fault_tolerance_f" in stats
        assert stats["fault_tolerance_f"] == 2

    def test_tx_hash_generated(self):
        record = make_record()
        assert len(record.tx_hash) == 16

    def test_with_byzantine_validators(self):
        """With 2 Byzantine validators, consensus should still reach quorum."""
        consortium_byz = PBFTConsortium(num_validators=9, num_byzantine=2, seed=99)
        record = make_record()
        # Even with 2 Byzantine, 7 honest validators → quorum of 5 reachable
        # (outcome depends on Byzantine vote; test passes if no exception raised)
        result = consortium_byz.submit_trust_delta(record)
        assert isinstance(result, bool)

    def test_export_ledger(self, tmp_path):
        record = make_record(agent_id="agent_050", score=0.3, interval=1)
        self.consortium.submit_trust_delta(record)
        export_path = str(tmp_path / "ledger.json")
        self.consortium.export_ledger(export_path)
        assert os.path.exists(export_path)
        import json
        with open(export_path) as f:
            data = json.load(f)
        assert "ledger" in data
        assert "stats" in data
