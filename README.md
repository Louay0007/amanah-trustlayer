

# AMANAH Trust Layer — EnergyMind Tunisia (IEEE TSYP14)

**Team:** _[official team name]_ &nbsp;·&nbsp; **Branch:** _[IEEE Student Branch]_ &nbsp;·&nbsp; **Team size:** ≤5 &nbsp;·&nbsp; **Phase 1 submission — deadline 15/10/2026**

**Challenge:** [EnergyMind Tunisia — Architects of the Energy Noosphere: A Human–AI Collective Intelligence Network for Energy Resilience](https://tsyp.ieee.tn/program/challenges/) · **Phase 1 submission — deadline 15/10/2026**

> *"Every reading is a witness. AMANAH decides which witness to trust, explains why, and knows when to hand the decision to a human."*

**AMANAH** = **A**ccountable · **M**eaningful · **A**uditable · **N**ot-blindly-trusted · **A**daptive · **H**uman-first.

---

## 1. What this is

An **AI-driven trust layer** that stands **between raw energy data and automated action**. Given a stream of grid
measurements, it decides — transparently — whether a reading reflects a **real event** or a **failing/lying
sensor**, how confident it is, why, and whether a human must decide. It addresses all four Challenge Goals:

1. **Anomaly & bias detection** (Goal 1) — detect equipment faults, degraded sensors, biased readings,
   corrupted/falsified readings and natural failures from **V · I · PQ · smart-meter** data, and crucially
   **distinguish a genuine grid event from a data-quality problem** because they demand different responses.
2. **Confidence & explainability** (Goal 2) — every flagged event yields a **3-way verdict**, a **confidence
   score**, a **plain-language explanation**, and an **auditable decision log** (what / concluded / why).
3. **Human–AI escalation & governance** (Goal 3) — a routing policy defining what the AI does alone vs. what it
   sends to an operator, addressing both **over-trust** and **under-trust**.
4. **Shared learning across the Noosphere** (Goal 4) — a documented, privacy-aware protocol for sharing anomaly
   patterns and calibration insights across nodes; the **working prototype is single-node**.

## 2. Our innovation — provenance-aware trust scoring

A purely statistical detector flags anomalies by *magnitude* and is fooled by a **plausible-but-falsified**
reading (a value that looks normal in isolation but is wrong). AMANAH adds a **cross-source physical-consistency
check**: the feeder head, PV inverters, and neighbouring meters all partially observe the *same* physical reality
through independent instruments. When sources that **must** agree disagree beyond physical expectation, the trust
score drops and the sample is routed to `DATA_QUALITY_PROBLEM` — **even when its magnitude looks perfectly
normal**. Concretely we enforce two invariants that a lying sensor cannot satisfy:

- **per-node energy balance:** `|P| = V · I · PF`
- **feeder-head balance:** a node's power must reconcile with the independent head measurement

Genuine events keep both invariants intact; data-quality problems break one. This gives **principled** (not
correlation-learned) genuine-vs-data-quality separation — and it is what catches the forged attack that
single-sensor detectors miss.


## Repository layout

```
amanah-trustlayer/
├── README.md                     # this file — overview, results, spec-compliance map
├── README_RESULTS.md             # detailed results write-up
├── LICENSE                       # MIT
├── requirements.txt              # pinned deps for reproducibility
├── requirements.lock.txt         # full pip-freeze snapshot
├── assets/
│   └── ieee_ies_issatm_logo.png
├── data/
│   ├── raw/
│   │   └── amanah_baseline_clean.csv      # physics LV-feeder baseline (regenerate: src/simulate.py)
│   ├── processed/
│   │   └── amanah_labeled.csv             # baseline + injected anomalies + ground-truth labels
│   └── anomaly_manifest.csv                # ground-truth labels of every injected anomaly
├── notebooks/
│   ├── 01_data_exploration.ipynb           # baseline stats + plots (executed)
│   └── 02_anomaly_injection.ipynb          # how anomalies were generated + detector walkthrough (executed)
├── src/
│   ├── simulate.py                         # LV feeder / baseline generation (deterministic)
│   ├── inject.py                           # the 5 anomaly classes + genuine events + manifest
│   ├── detect.py                           # feature engineering + Isolation Forest + 3-way verdict + audit log
│   └── evaluate.py                         # metrics, per-class, confusion matrix, latency, figures
├── scripts/
│   ├── validate.py                         # dataset integrity + label-integrity checks
│   ├── build_notebook_01.py                # builder for the data-exploration notebook
│   └── build_notebook_02.py                # builder for the anomaly-injection notebook
├── results/
│   ├── detector_metrics.json               # headline detection + separation + SHAP
│   ├── metrics_full.json                   # everything incl. per-class + latency
│   ├── per_class_metrics.csv
│   ├── confusion_matrix.csv
│   ├── decision_log.csv                    # auditable log: what / concluded / why (Goal 2)
│   ├── latency_by_event.csv
│   ├── latency_by_instance.csv
│   └── figures/                            # system_timeline, shap_importance, confusion_matrix, per_class_recall
└── docs/
    ├── PROPOSAL.md                         # Phase-1 Project Proposal (Scope, Objectives, Goals 1–4)
    ├── DATASET.md                          # data provenance + anomaly generation (spec: "must document")
    └── SHARING_DESIGN.md                   # Goal 4 multi-node sharing + privacy
```

## Quick start (reproduce everything)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/simulate.py     # 1. baseline LV-feeder series  -> data/raw/amanah_baseline_clean.csv
python src/inject.py       # 2. inject 5 anomaly classes + genuine events -> amanah_labeled.csv + anomaly_manifest.csv
python src/detect.py       # 3. features + Isolation Forest + 3-way verdict -> results/decision_log.csv + detector_metrics.json
python src/evaluate.py     # 4. per-class metrics + confusion + latency + figures -> results/*
python scripts/validate.py # 5. integrity checks (physical balance, label integrity)

# Optional: real-data validation against Open Power Systems Data (needs /tmp/opsd.csv, see docs/DATASET.md §1.2)
python scripts/calibrate_real_data.py   # 6. PV/load shape vs real OPSD -> results/figures/real_data_calibration.png
```

Everything is deterministic (fixed seed `2026`), so re-running reproduces the dataset, labels and metrics byte-for-byte.

## Data provenance & real-data validation

The anomaly baseline is **simulated in-repo** (spec: *"real fault/anomaly-labeled data is rare, so
simulation is expected"*), because the required per-meter V/I/PQ granularity with an **exact physical
relationship** (`|P| = V·I·PF`) and a **feeder-head power balance** does not exist in any public labeled
fault dataset — and that physics is precisely what lets the trust layer catch falsified readings.

Crucially, the simulator is **quantitatively validated against a real, citable public dataset**, so the
physical backbone is not arbitrary:

- **Open Power Systems Data (OPSD)**, package `2020-10-06` (DOI `10.25832/time_series/2020-10-06`,
  ENTSO-E-derived, peer-review standard).
- Simulated **rooftop-PV daily shape vs real German solar generation: Pearson r = 0.954** — a near-identical
  curve (solar is universal irradiance physics). Load differs **by design** (German TSO morning-peak vs
  Tunisian residential evening-peak) — documented, not hidden.
- Artifacts: `data/external/opsd_de_load_solar_summer_week.csv`, `data/external/calibration_summary.json`,
  figure `results/figures/real_data_calibration.png`. Full detail in `docs/DATASET.md` §1.2.

## Headline results (test split, time-based 70/30, no leakage)

| Metric (spec-named) | Value |
|---|---|
| Detection precision | **0.82** |
| Detection recall | **0.87** |
| F1 | **0.85** |
| False-positive rate (clean control) | **0.0003** |
| Node-instance recall | **1.00** (38/38) |
| Event recall | **1.00** (15/15) |
| Genuine ↔ data-quality **separation** | **1.00** (0 genuine↔DQ cross-misclassifications — see `results/` confusion matrix) |
| Detection delay | median **0 min**, p95 **15 min** |

> Note on "separation": every flagged cell is routed to the correct *side* of the genuine-vs-data-quality
> boundary (confusion matrix shows 0 genuine→DQ and 0 DQ→genuine). The lower `verdict_accuracy` (0.82)
> in `results/detector_metrics.json` is the cell-level "correct verdict among all flagged cells," which
> additionally counts the ~57 false-positives that fired on *normal* cells — a stricter number, not a
> contradiction.

Per-class recall: `genuine_event 1.00` · `corrupted_reading 0.88` · `degraded_sensor 0.75` · `biased_reading 0.50` · `equipment_fault 0.29` — while the **verdict (genuine vs data-quality) is correct for every flagged instance** across all classes. SHAP attribution (`results/figures/shap_importance.png`) ranks `z_score`, `z_step`, `head_res`, `peer_z` as the top interpretable drivers. Full detail in `results/metrics_full.json` and `README_RESULTS.md`.

## Spec-compliance map (EnergyMind TSYP14)

### Deliverable — Project Proposal ✅ `docs/PROPOSAL.md`
Required: **Scope** (§1) · **Objectives** (§2) · **Data/anomaly-simulation plan** (§3) · **Preliminary design for each Challenge Goal** (§4.1–4.4).

### Deliverable — GitHub repository with Dataset + Phase-1 code ✅
| Spec requirement | Where |
|---|---|
| Dataset | `data/processed/amanah_labeled.csv` (committed) + `data/anomaly_manifest.csv` |
| **Document how anomalies were generated** (mandated) | `docs/DATASET.md` §3–4 + `notebooks/02_anomaly_injection.ipynb` |
| Code developed during Phase 1 | `src/simulate.py`, `src/inject.py`, `src/detect.py`, `src/evaluate.py`, `scripts/` |
| Working single-node prototype | the whole `src/` pipeline, runnable end-to-end |

### The four Challenge Goals ✅
| Goal | Coverage | Evidence |
|---|---|---|
| **1. Anomaly & bias detection** — faults, degraded sensors, biased, corrupted/falsified, natural failures; distinguish genuine event vs data-quality | 5 injected classes + genuine events; physical-invariant 3-way verdict | `results/per_class_metrics.csv`, confusion matrix; genuine↔DQ separation **100%** |
| **2. Confidence & explainability** — confidence score, plain-language reason, auditable log; SHAP/LIME/rule-backed, not black-box | rule-backed reasons + SHAP + per-cell `confidence` | `results/decision_log.csv`, `results/figures/shap_importance.png` |
| **3. Human-AI escalation & governance** — auto thresholds, human-approval cases, operator interface, over-/under-trust | AUTO / REVIEW / ESCALATE routing + trust calibration | `docs/PROPOSAL.md` §4.3 |
| **4. Shared learning across the Noosphere** — share anomaly patterns/calibration; privacy (what shared / local / why) | documented federated design; single-node prototype | `docs/SHARING_DESIGN.md` |

### Evaluation criteria the spec names ✅
Detection precision/recall · false-positive rate · explanation quality (reason + attribution on every flag) · response latency — all measured in `results/` (see table above and `results/metrics_full.json`).

### Pitching (Phase 2) — English, 5-min presentation + 3-min demo
Out of scope for Phase 1; the working prototype and decision log are the demo substrate.

## Status

- [x] Repo scaffold + pinned, reproducible environment (`requirements.lock.txt`)
- [x] Baseline: physics LV feeder (`src/simulate.py`), 0 NaN, exact power balance
- [x] Anomaly injector — 5 classes + genuine events + ground-truth manifest (`src/inject.py`)
- [x] Detector — features + Isolation Forest + 3-way verdict (`src/detect.py`)
- [x] Explainability + auditable decision log + SHAP (`results/decision_log.csv`)
- [x] Escalation routing table + trust calibration (`docs/PROPOSAL.md` §4.3)
- [x] Shared-learning / privacy design (`docs/SHARING_DESIGN.md`)
- [x] Metrics: per-class precision/recall, FPR, latency, figures (`src/evaluate.py`, `results/`)
- [x] Project Proposal (`docs/PROPOSAL.md`)
- [x] Exploratory + injection notebooks, executed (`notebooks/01`, `02`)
- [ ] Phase 2 (12/12/2026): operator UI, live escalation, multi-node federation demo, demo video + 5-min pitch

## License & attribution

Released under the **MIT License** (see `LICENSE`). Challenge: **EnergyMind Tunisia — IEEE TSYP14**.
Baseline is a physics-informed simulation generated in-repo (`src/simulate.py`); anomaly ground truth is
fully documented in `docs/DATASET.md` and `data/anomaly_manifest.csv` (spec: *"teams must document how the
anomalies were generated"*). All results are deterministic (seed `2026`) and reproducible via the quick-start above.


