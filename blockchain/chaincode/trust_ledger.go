/*
Trust Ledger Smart Contract
============================
Hyperledger Fabric chaincode skeleton for the IoUT trust evidence ledger.
Stores per-agent trust delta summaries, anomaly events, and enforcement
policy records as described in the paper (Section IV.D).

This is a structural reference implementation. Deploy with:
  peer lifecycle chaincode install trust-ledger.tar.gz
  peer lifecycle chaincode approveformyorg ...
  peer lifecycle chaincode commit ...

NOTE: Requires Hyperledger Fabric 2.4+ and Go 1.18+.
*/

package main

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

// ----------------------------------------------------------------
// Data Structures
// ----------------------------------------------------------------

// TrustRecord stores a single trust delta commit for one agent.
type TrustRecord struct {
	AgentID      string    `json:"agent_id"`
	TrustScore   float64   `json:"trust_score"`
	Interval     int       `json:"interval"`
	IsAnomalous  bool      `json:"is_anomalous"`
	AttackType   string    `json:"attack_type,omitempty"`
	Timestamp    time.Time `json:"timestamp"`
	CommittedBy  string    `json:"committed_by"` // Interrogator node ID
}

// EnforcementEvent records a tiered enforcement action.
type EnforcementEvent struct {
	AgentID          string    `json:"agent_id"`
	EnforcementTier  int       `json:"tier"`  // 1=local, 2=consortium, 3=isolation
	Action           string    `json:"action"` // "routing_exclusion", "throttle", "isolate"
	ValidatedByPBFT  bool      `json:"validated_by_pbft"`
	Timestamp        time.Time `json:"timestamp"`
}

// AgentIdentity stores authenticated agent identity (set at deployment).
type AgentIdentity struct {
	AgentID     string `json:"agent_id"`
	AgentType   string `json:"agent_type"` // "sensor" or "auv"
	PublicKey   string `json:"public_key"`
	RegisteredAt string `json:"registered_at"`
}

// ----------------------------------------------------------------
// Smart Contract
// ----------------------------------------------------------------

// TrustLedgerContract is the Fabric chaincode contract.
type TrustLedgerContract struct {
	contractapi.Contract
}

// RegisterAgent adds a new agent identity to the ledger.
// Called once at deployment time per agent.
func (c *TrustLedgerContract) RegisterAgent(
	ctx contractapi.TransactionContextInterface,
	agentID string,
	agentType string,
	publicKey string,
) error {
	key := fmt.Sprintf("identity_%s", agentID)
	existing, err := ctx.GetStub().GetState(key)
	if err != nil {
		return fmt.Errorf("failed to read state: %v", err)
	}
	if existing != nil {
		return fmt.Errorf("agent %s already registered", agentID)
	}

	identity := AgentIdentity{
		AgentID:      agentID,
		AgentType:    agentType,
		PublicKey:    publicKey,
		RegisteredAt: time.Now().UTC().Format(time.RFC3339),
	}
	data, err := json.Marshal(identity)
	if err != nil {
		return err
	}
	return ctx.GetStub().PutState(key, data)
}

// SubmitTrustDelta commits a trust delta summary for one agent.
// Called by interrogator nodes when a trust score crosses tau_min.
func (c *TrustLedgerContract) SubmitTrustDelta(
	ctx contractapi.TransactionContextInterface,
	agentID string,
	trustScore float64,
	interval int,
	isAnomalous bool,
	attackType string,
	committedBy string,
) error {
	// Verify agent is registered
	identityKey := fmt.Sprintf("identity_%s", agentID)
	identityData, err := ctx.GetStub().GetState(identityKey)
	if err != nil || identityData == nil {
		return fmt.Errorf("agent %s not registered", agentID)
	}

	record := TrustRecord{
		AgentID:     agentID,
		TrustScore:  trustScore,
		Interval:    interval,
		IsAnomalous: isAnomalous,
		AttackType:  attackType,
		Timestamp:   time.Now().UTC(),
		CommittedBy: committedBy,
	}
	data, err := json.Marshal(record)
	if err != nil {
		return err
	}

	// Composite key: trust_{agentID}_{interval}
	key := fmt.Sprintf("trust_%s_%05d", agentID, interval)
	return ctx.GetStub().PutState(key, data)
}

// RecordEnforcementAction records a tiered enforcement decision.
func (c *TrustLedgerContract) RecordEnforcementAction(
	ctx contractapi.TransactionContextInterface,
	agentID string,
	tier int,
	action string,
	validatedByPBFT bool,
) error {
	event := EnforcementEvent{
		AgentID:         agentID,
		EnforcementTier: tier,
		Action:          action,
		ValidatedByPBFT: validatedByPBFT,
		Timestamp:       time.Now().UTC(),
	}
	data, err := json.Marshal(event)
	if err != nil {
		return err
	}
	key := fmt.Sprintf("enforcement_%s_%d", agentID, time.Now().UnixNano())
	return ctx.GetStub().PutState(key, data)
}

// GetAgentTrustHistory retrieves the full trust record history for an agent.
func (c *TrustLedgerContract) GetAgentTrustHistory(
	ctx contractapi.TransactionContextInterface,
	agentID string,
) ([]*TrustRecord, error) {
	prefix := fmt.Sprintf("trust_%s_", agentID)
	resultsIterator, err := ctx.GetStub().GetStateByRange(prefix, prefix+"~")
	if err != nil {
		return nil, err
	}
	defer resultsIterator.Close()

	var records []*TrustRecord
	for resultsIterator.HasNext() {
		queryResult, err := resultsIterator.Next()
		if err != nil {
			return nil, err
		}
		var record TrustRecord
		if err := json.Unmarshal(queryResult.Value, &record); err != nil {
			return nil, err
		}
		records = append(records, &record)
	}
	return records, nil
}

// GetLatestTrustScore returns the most recent trust score for an agent.
func (c *TrustLedgerContract) GetLatestTrustScore(
	ctx contractapi.TransactionContextInterface,
	agentID string,
) (*TrustRecord, error) {
	history, err := c.GetAgentTrustHistory(ctx, agentID)
	if err != nil || len(history) == 0 {
		return nil, fmt.Errorf("no trust records found for agent %s", agentID)
	}
	// Return last record (highest interval index)
	latest := history[0]
	for _, r := range history {
		if r.Interval > latest.Interval {
			latest = r
		}
	}
	return latest, nil
}

// ----------------------------------------------------------------
// Main entry point
// ----------------------------------------------------------------

func main() {
	chaincode, err := contractapi.NewChaincode(&TrustLedgerContract{})
	if err != nil {
		fmt.Printf("Error creating trust ledger chaincode: %v\n", err)
		return
	}
	if err := chaincode.Start(); err != nil {
		fmt.Printf("Error starting trust ledger chaincode: %v\n", err)
	}
}
