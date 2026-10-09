# Project Proposal — AMANAH Trust Layer
### EnergyMind Tunisia: Architects of the Energy Noosphere
### IEEE TSYP14 Technical Challenge · Phase 1 (Ideation & Data Exploration)

> **One-line pitch.** AMANAH is a *trust layer* that stands between raw grid data and automated
> action: it detects when a meter is faulty, drifting, biased, or **lying**, tells a human *what it
> found and why*, and escalates only when a person truly needs to decide — instead of silently
> corrupting the data that downstream automation relies on.

**Team:** _[Team name]_ &nbsp;·&nbsp; **Branch:** _[IEEE Student Branch]_ &nbsp;·&nbsp; **Scope slice:** LV smart-metered neighborhood feeder &nbsp;·&nbsp; **Phase-1 deadline:** 15/10/2026

---

## 1. Scope

Tunisia's grid is absorbing seasonal peaks, fast-growing rooftop PV, a rapidly expanding smart-meter
estate, and an ever-larger fleet of IoT devices. Each new meter or sensor is another point where a
fault, a miscalibration, or a deliberately falsified reading can **silently corrupt** the data that
increasingly-automated systems consume. The cost of "predicting well" is worthless if the input
cannot be trusted.

**We build the trust layer for one concrete slice of the system:** a **low-voltage (LV)
smart-metered neighborhood feeder** — the exact kind of asset proliferating fastest in Tunisia. It
comprises 24 household smart meters, 8 rooftop-PV inverters, 3 sub-metered commercial loads
(clinic, cold store, water pump), and an independently metered feeder head, all resolved at
**15-minute** granularity over a **30-day** Tunisian summer peak (July). Monitored signals are the
ones the challenge names: **voltage (V), current (I), power quality (Q, PF, f), and smart-meter
active power (P).**

The trust layer sits *between* this raw telemetry and any downstream automated control. Its output
is not a control action — it is a **verdict, a confidence, a plain-language reason, and an auditable
record**, which a human or a downstream system can act on with justified trust.

**Why this slice:** it is the highest-density, fastest-growing, most privacy-sensitive corner of the
grid, it exercises all four challenge goals simultaneously (faults, falsification, escalation,
sharing), and it is exactly where a wrong automated decision hurts real people.

---

## 2. Objectives

Our objectives map one-to-one onto the four Challenge Goals, plus the evaluation criteria the spec
names. Each has a concrete Phase-1 "done" test.

| ID | Objective (Challenge Goal) | Phase-1 done test |
|---|---|---|
| **O1** | **Anomaly & bias detection (Goal 1).** Monitor V, I, PQ and smart-meter P; detect equipment faults, degraded sensors, biased readings, corrupted/falsified readings, and natural failures — and **distinguish a genuine grid event from a data-quality problem** because they demand different responses. | 5 anomaly classes + genuine events generated and documented; detector separates genuine vs data-quality with measured recall/precision. |
| **O2** | **Confidence & explainability (Goal 2).** Every flagged event yields a human-verifiable verdict: a confidence score, a plain-language explanation, and an auditable decision log (what / concluded / why). | Decision log emitted per flagged event with reason strings; SHAP + rule-backed attribution; no black-box-only scoring. |
| **O3** | **Human-AI escalation & governance (Goal 3).** Define what the AI decides alone vs. what routes to an operator; provide an operator review/override interface; address both over- and under-trust. | Routing table (confidence × trust × criticality → AUTO / REVIEW / ESCALATE); operator override demo writes back to the log. |
| **O4** | **Shared learning across the Noosphere (Goal 4).** Design + document how multiple nodes share anomaly patterns and calibration insights while protecting privacy (what is shared, what stays local, why). | `docs/SHARING_DESIGN.md` with a shared/local table + federated protocol; working single-node prototype. |
| **O5** | **Evaluation.** Report detection precision/recall, false-positive rate on clean data, explanation quality, and response latency. | `results/metrics_full.json`: per-class P/R/F1, FPR, node/event recall, latency (median & p95). |

---

## 3. Data & anomaly-simulation plan

The spec states that *"real fault/anomaly-labeled grid data is rare, so simulation is expected,"* and that teams must **document how the anomalies were generated**. Our plan therefore combines the three allowed sources — **public sources • STEG open data • simulated equivalents** — into one reproducible, fully-labeled dataset. Full provenance is in `docs/DATASET.md`; generation is deterministic (seed `2026`) via `python src/simulate.py && python src/inject.py`.

### 3.1 Baseline
A physics-informed **LV feeder simulator** (`src/simulate.py`) emits 30 days of July (Tunisian summer-peak) data at **15-minute** resolution for 36 nodes: 24 household smart meters, 8 rooftop-PV inverters, 3 sub-metered loads (clinic / cold-store / water-pump) and a feeder head. Each node reports six channels — **V, I, P, Q, PF, f** — the exact signals Goal 1 names. Load shapes (morning/evening peaks, midday AC hump, weekend dampening) and the PV daylight bell curve are calibrated to STEG seasonal load statistics; voltage droops with feeder loading and current obeys `I = P/(V·PF)`. The clean baseline is kept as a **control** to prove a low false-positive rate.

### 3.2 Monitored signals & provenance
Every reading carries a **provenance source** (household meter / PV inverter / sub-meter / feeder head). This is deliberate: it lets the trust layer cross-check one node against independent references (the physical energy balance and the feeder-head balance) — the basis of our genuine-vs-data-quality discriminator.

### 3.3 Injected anomaly classes (1:1 with Goal 1)
Each class is injected with a documented mechanism, a subtype, and an exact `start_ts/end_ts` recorded in `data/anomaly_manifest.csv`.

| Class (goal-1 term) | Injection mechanism | `is_genuine_grid_event` |
|---|---|---:|
| Equipment fault | voltage sag/swell (per-node step), inrush spike ramp | true |
| Natural equipment failure | open-circuit zeros, total-loss, gradual degradation ramp | true |
| Degraded sensor (drift) | multiplicative `y·(1+α·t)` or additive PF ramp, stuck-at, rising-quantile noise | false |
| Biased reading | constant gain/offset, sign inversion on generation | false |
| Corrupted / falsified reading | flatline, random noise, **provenance-forged plausible value** | false |

### 3.4 Genuine grid events (must NOT be flagged as data problems)
Summer heat-surge (coordinated block-load on all households), PV cloud dip, EV-charging block, scheduled maintenance outage. These are real excursions but remain **physically consistent** — the property that separates them from data-quality problems.

### 3.5 The provenance-forged attack (our differentiator)
`corrupted_reading → forged_plausible` replaces a meter's P/I with values that look entirely reasonable in isolation (within the node's own normal band) yet are wrong. A magnitude- or single-sensor detector misses it; only the **cross-source / physical-consistency check** catches it. This is the concrete demonstration of our provenance-aware trust-scoring innovation.

### 3.6 Label schema & base rate
`data/anomaly_manifest.csv` records, per flagged `(node_id, channel)`: `anomaly_id, class, subtype, start_ts, end_ts, params, provenance_source, is_genuine_grid_event, seed`. The full labeled stream `data/processed/amanah_labeled.csv` adds `is_anomaly, anomaly_class, anomaly_subtype, is_genuine_grid_event, is_data_quality_problem`. Base rate is kept low (**≈0.10%** of cells) so precision/recall are meaningful, and the clean control series bounds the false-positive rate.

---

## 4. Preliminary design for each Challenge Goal

### 4.1 Goal 1 — Anomaly & bias detection (with the genuine-vs-data-quality distinction)
The detector (`src/detect.py`) combines a spec-named **Isolation Forest** with a **rule-backed statistical layer** so nothing is black-box. Per cell `(timestamp, node, channel)` we engineer interpretable features — rolling mean/std, a z-score vs. the node's own history, a step-score, a flatline run-length, a peer-z vs. same-kind neighbors, the **energy-balance residual** `‖P‖ − V·I·PF`, and the **feeder-head residual** (node vs. the independent head measurement). The verdict is a clean, auditable cascade:

- **DATA_QUALITY_PROBLEM** — a *physical invariant is broken*: the energy balance `‖P‖=V·I·PF` no longer holds, or the node disagrees with the feeder-head balance. This is what drift, bias, flatline and falsified readings all do.
- **GENUINE_GRID_EVENT** — a real statistical excursion (|z| or step beyond threshold) that *stays physically consistent* (heat surge, sag/swell, outage, failure).
- **NORMAL** — neither.

Because genuine events remain physically coherent while data-quality problems violate an invariant, the separation is principled, not learned-by-correlation. Phase-1 result: **anomaly-level recall 1.00**, **genuine↔data-quality verdict accuracy 100%** (0 cross-misclassifications in the confusion matrix).

### 4.2 Goal 2 — Confidence scores & explainability
Every flagged cell gets a **confidence** (0.90 for a broken physical invariant, scaled 0.5–0.95 for a statistical excursion), a **plain-language reason** ("V, I, P, PF physically inconsistent — energy balance violated"; "meter disagrees with feeder-head power balance"; "real excursion vs own history, but physically consistent"), and a row in the **auditable decision log** (`results/decision_log.csv`) recording what was detected, concluded, and why. Feature attribution is **SHAP** on the Isolation Forest (top drivers: `z_score`, `z_step`, `head_res`, `peer_z`) plus the rule that fired — never black-box-only. See `results/figures/shap_importance.png`.

### 4.3 Goal 3 — Human-AI escalation & governance (preliminary design)
Actions are routed by **confidence × trust × criticality** into three tiers:

| Tier | Condition | Behavior |
|---|---|---|
| **AUTO** | very high confidence **and** low criticality (e.g. flagged meter, non-essential circuit) | AI acts alone; logged with reason |
| **REVIEW** | medium confidence **or** moderate criticality | queued for operator approval within SLA |
| **ESCALATE** | low confidence **or** safety-critical (clinic essential circuits) | requires a human operator decision before any action |

**Trust calibration** addresses both failure modes: a documented **acceptance rate / override rate** drives a dynamic confidence threshold (frequent overrides → raise the bar, countering *over-trust*; near-zero overrides on a channel → cautiously lower it, countering *under-trust*). The operator interface (Phase 2) exposes the verdict, reason, SHAP attribution and a one-click approve/override that writes back to the audit log.

### 4.4 Goal 4 — Shared learning across the Noosphere (preliminary design)
Multi-node sharing stays a **documented design** (the spec allows this while requiring a working single-node prototype). `docs/SHARING_DESIGN.md` will specify, per knowledge type, **what is shared vs. what stays local**:

- **Shared (privacy-safe):** anonymized *anomaly patterns* (feature histograms / model weight deltas), aggregate *calibration insights* (drift coefficients), and attack signatures — no raw household P/V, no node identity, no timestamps.
- **Local (never shared):** raw smart-meter series, per-household profiles, and any signal that could re-identify a household.
- **Why:** federated / lightweight multi-agent aggregation preserves collective defense (a fault signature learned at one microgrid warns others) while satisfying the privacy requirement via differential-privacy noise on shared aggregates and no raw-data egress.

### 4.5 Evaluation (Goal-1 metrics the spec names)
Measured on a time-based 70/30 split (no leakage; scaler + Isolation Forest fit on train only): per-class **precision/recall/F1**, **false-positive rate on the clean control**, **explanation quality** (every flag carries a reason + attribution), and **response latency** (`results/latency_by_event.csv`, `results/metrics_full.json`). Headline: precision 0.82 · recall 0.87 · F1 0.85 · FPR 0.0003 · node-instance recall 1.00 · median detection delay 0 min, p95 15 min.


