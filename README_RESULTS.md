# Results & Evaluation — Phase 1

Every number below is regenerated deterministically by:

```
python src/simulate.py    # clean baseline   -> data/raw/amanah_baseline_clean.csv
python src/inject.py      # + anomalies      -> data/processed/amanah_labeled.csv
                                                   data/anomaly_manifest.csv
python src/detect.py      # train + verdict  -> results/detector_metrics.json
                                                   results/decision_log.csv
python src/evaluate.py    # metrics + figures-> results/*.csv, results/figures/*.png
```

Fixed seed = **2026**. No train/test leakage (time-based 70/30 split; scaler and
Isolation Forest fit on the training window only).

## Detection quality (test split)

| Metric | Value |
|---|---|
| Precision | 0.82 |
| Recall (flagged cells) | 0.87 |
| F1 | 0.85 |
| False-positive rate | 0.0003 |
| **Node-instance recall** | **1.00** (38/38) |
| **Event recall** | **1.00** (15/15) |
| **Genuine vs Data-Quality separation** | **1.00** (0 cross-errors) |

## Per-class recall (test)

| class | genuine? | support | flagged | recall | verdict accuracy |
|---|---|---|---|---|---|
| genuine_event | yes | 192 | 192 | 1.00 | 1.00 |
| corrupted_reading | no | 8 | 7 | 0.88 | 1.00 |
| degraded_sensor | no | 64 | 48 | 0.75 | 1.00 |
| biased_reading | no | 24 | 12 | 0.50 | 1.00 |
| equipment_fault | yes | 14 | 4 | 0.29 | 1.00 |

**Reading the equipment_fault number correctly:** voltage faults inject a strong excursion on the
**V** channel (|z| = 4.3–14.6, always flagged) and a weak one on **I** (|z| < 1.6). Cell-recall counts
the weak I cells as missed, but at the level that matters — *is the node flagged during the fault?* —
recall is 1.00. The fault is caught at the physically correct channel (voltage).

## Response latency (spec metric)

| delay | value |
|---|---|
| median | 0.0 min |
| mean | 1.97 min |
| p95 | 15.0 min |

Detection latency is measured per node-instance against the manifest `[start_ts, end_ts]` window;
p95 = one 15-minute sampling interval.

## How the verdict separates genuine events from data problems

This is the core of the trust layer and it is what the confusion matrix proves:

- **Data-quality problems** (drift, bias, corrupted) break a **physical invariant**: either the
  per-node energy balance `|P| = V·I·PF` (feature `recon_err`) or the feeder-head balance
  (feature `head_res`). They are flagged `DATA_QUALITY_PROBLEM`.
- **Genuine grid events** (heat surge, cloud dip, EV block, faults) keep every physical invariant
  intact (`recon_err ≈ 0`, `head_res ≈ 0`) but show a **statistical excursion** vs the node's own
  history (`z_score`, `z_step`). They are flagged `GENUINE_GRID_EVENT`.

Confusion matrix (test): `genuine → DATA_QUALITY = 0`, `data_quality → GENUINE = 0`.

## Files

- `metrics_full.json` — all headline numbers above
- `per_class_metrics.csv`, `confusion_matrix.csv` — per-class + 3×3 confusion
- `latency_by_instance.csv`, `latency_by_event.csv` — per-instance / per-event detection delay
- `decision_log.csv` — auditable per-cell log: verdict, confidence, plain-language reason, features,
  and the ground-truth label (what was detected, concluded, and why — Goal 2 requirement)
- `figures/system_timeline.png` — feeder head vs Σ children with genuine vs data-quality flags
- `figures/shap_importance.png` — SHAP feature importance over the Isolation Forest
- `figures/confusion_matrix.png`, `figures/per_class_recall.png`

## Anomaly generation documentation

See `docs/DATASET.md` (baseline provenance, the five injected classes mapped 1:1 to Goal 1, and the
ground-truth schema) and `data/anomaly_manifest.csv` (every injected anomaly with exact
`start_ts`/`end_ts`, `class`, `subtype`, `params`, `seed`). This satisfies the spec requirement to
"document how the anomalies were generated."
