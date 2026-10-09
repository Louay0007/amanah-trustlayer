# Shared Learning Across the Noosphere — Design

> Challenge Goal 4. A **working single-node prototype is required**; multi-node sharing "may remain a
> documented design." This document is that design: **what is shared, what remains local, and why.**

## 1. Nodes
Factories • neighborhoods • microgrids. Each runs the same single-node trust layer (`src/detect.py`)
locally, producing local verdicts, confidence, reasons and an audit log. Nothing in this design
requires a node to export raw data.

## 2. What is shared vs. what stays local

| Knowledge type | Shared? | What exactly | Privacy rationale |
|---|---|---|---|
| Anomaly patterns | **Shared** | Feature histograms / prototype vectors and class signatures (e.g. the *shape* of a voltage-sag or a flatline run), or model-weight deltas. | No raw series, no node identity, no timestamps — cannot reconstruct a household's load. |
| Calibration insights | **Shared** | Aggregate drift coefficients (e.g. "these meter batches read +4% over 6 months") and per-class false-positive rates. | Population-level statistics, not per-household; resists re-identification. |
| Attack signatures | **Shared** | Fingerprints of forged/biased-reading attacks so other nodes can pre-empt them. | Signatures describe the *attack*, not the victim. |
| Raw smart-meter series | **Local** | Never leaves the node. | Directly reveals occupancy/appliances. |
| Per-household profiles / node IDs | **Local** | Never shared. | Identity + behavior. |
| Timestamps of events | **Local** | Only relative/coarse timing if ever shared. | Fine timestamps + shape can re-identify. |

## 3. Why federated / lightweight multi-agent
A fault signature learned at one microgrid can warn the others *before* it arrives — collective
defense — while every household's raw data stays on its own node. Aggregation (e.g. FedAvg over the
Isolation Forest + rule parameters) is done on a coordinator that sees only model updates.

## 4. Privacy mechanism (Phase 2)
- **Differential-privacy noise** on shared aggregates/histograms so a single node's contribution is
  bounded and not inferable.
- **No raw-data egress**: the protocol transports model deltas and signatures only.
- **Secure aggregation**: the coordinator sees the sum of updates, not any individual update.
- **Audit**: every shared contribution is logged (what, from which node class, under what noise).

## 5. Phase-1 status
The single-node trust layer is implemented and evaluated (`src/detect.py`, `results/metrics_full.json`).
The multi-node protocol above is the documented design; a two-node simulation of federated signature
sharing is a Phase-2 deliverable.
