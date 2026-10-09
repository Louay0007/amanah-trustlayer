# Dataset — Provenance & Anomaly Generation

> The challenge spec explicitly requires teams to **"document how the anomalies were generated."**
> This file is the authoritative record. Everything below is reproducible from the committed code:
> `python src/simulate.py && python src/inject.py` regenerates the dataset + manifest deterministically
> (fixed seed `2026`).

## 1. Baseline data

Real fault/anomaly-labeled grid data is rare, and the spec states **"simulation is expected."**
Accordingly, the committed baseline is a **physics-informed LV-feeder simulator** (`src/simulate.py`),
not an external download.

| Field | Value |
|---|---|
| Baseline source | **Physics LV-feeder simulator** (`src/simulate.py`), deterministic seed `2026` |
| What it simulates | A Sfax-inspired low-voltage distribution feeder: 24 household smart meters, 8 rooftop-PV inverters, 3 sub-metered loads (clinic / cold-store / water-pump), 1 feeder-head meter |
| Channels | Per node at 15-minute resolution: V, I, P, Q, PF, f |
| Coverage | 30 days (July 2026 = Tunisian summer peak), 622,080 rows |
| Design references (cited, not ingested) | Tunisian summer daily/seasonal load curves consistent with **STEG** public load statistics; ITIC/CENELEC LV voltage bands (EN 50160); cross-checked conceptually against **IEEE DataPort** / **UCI** electricity datasets |
| Generation date | 2026-10-09 |
| License | Simulated equivalents produced in-repo; no third-party data redistributed |

**Rationale (spec compliance):** the spec permits *"public sources • STEG open data • simulated equivalents."*
We use **simulated equivalents** because the required granularity — per-meter V/I/PQ with a physically
consistent relationship (`|P| = V·I·PF`) and a feeder-head power balance — is not available in any public
labeled fault dataset. Building it in-repo gives us an exact ground truth for the anomaly manifest while
keeping the physical structure real. The STEG-inspired daily curve shape (summer morning + evening peaks,
midday AC hump, PV daylight bell) is what makes genuine events physically plausible.

**Provenance tagging:** every channel carries a `provenance` source
(`feeder_head` / `household_meter` / `pv_inverter` / `sub_meter`), which is the basis for the cross-source
trust check (a meter that disagrees with the feeder-head power balance is flagged as untrustworthy).

## 2. Scope of monitored signals

Per node / channel, at **15-minute resolution**: voltage (V), current (I), active power (P),
reactive power (Q), power factor (PF), frequency (f) — exactly the signals Goal 1 names.
Channels are tagged with a **provenance source** enabling the cross-source trust check.

## 3. Anomaly classes injected (1:1 with Goal 1)

| # | Class | Injection mechanism | `is_genuine_grid_event` |
|---|-------|---------------------|:---:|
| 1 | Equipment fault | voltage sag/swell, inrush spikes, phase imbalance | true |
| 2 | Degraded sensor / drift | slow multiplicative `y·(1+α·t)` or additive ramp; stuck-at; rising quant. noise | false |
| 3 | Biased reading | constant gain (scale) or offset (bias) error; sign inversion on generation | false |
| 4 | Corrupted / falsified reading | (a) flatline/constant, (b) random noise, (c) **provenance-forged plausible value** | false |
| 5 | Natural equipment failure | sudden open-circuit (zeros), total loss, gradual degradation | true |

## 4. Genuine grid events (must NOT be flagged as data problems)

Summer heat-surge load, PV midday cloud dip, EV-charging block load, scheduled maintenance outage.
These stress the **3-way verdict** — they are real anomalies in the signal but *not* data-quality problems.

## 5. Ground-truth label schema (`data/anomaly_manifest.csv`)

One row per (anomaly × node × channel) with the exact injection window, so every flagged event can be
verified against the ground truth:

```
anomaly_id, node_id, channel, class, subtype, start_ts, end_ts,
params, provenance_source, is_genuine_grid_event, seed
```

Current manifest: **23 unique anomalies / 99 rows** covering all 5 Goal-1 classes plus genuine events
(see §3). `is_genuine_grid_event` is the label that trains/tests the 3-way verdict.

## 6. Reproducibility

- Fixed RNG seed (`2026`) is stored in the manifest and echoed in every figure and notebook.
- `python src/simulate.py && python src/inject.py` regenerates the full dataset + manifest
  deterministically; `python src/detect.py` re-runs the detector and rewrites `results/`.
- Anomaly base rate is kept low (~0.10% of cells) so precision/recall is meaningful; the clean
  baseline (`data/raw/`) is retained as a low-false-positive control.
- The end-to-end generation and detection workflow is documented in the notebooks
  `01_data_exploration.ipynb` and `02_anomaly_injection.ipynb`.
