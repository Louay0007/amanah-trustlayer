from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

INTERVAL_MIN = 15
STEPS_PER_DAY = 24 * 60 // INTERVAL_MIN
V_NOMINAL = 230.0
F_NOMINAL = 50.0
DEFAULT_SEED = 42


@dataclass
class NodeSpec:
    node_id: str
    kind: str
    base_kw: float = 0.0
    peak_kw: float = 0.0
    provenance: str = "smart_meter"
    params: Dict[str, float] = field(default_factory=dict)


def build_node_specs(n_households=24, n_pv=8):
    specs = []
    hh_peaks = np.linspace(1.2, 4.0, n_households)
    for i in range(n_households):
        specs.append(NodeSpec(
            node_id=f"HH{i + 1:02d}",
            kind="household",
            base_kw=round(float(hh_peaks[i]) * 0.35, 3),
            peak_kw=round(float(hh_peaks[i]), 3),
            provenance="smart_meter",
        ))
    pv_caps = np.linspace(3.0, 6.0, n_pv)
    for i in range(n_pv):
        specs.append(NodeSpec(
            node_id=f"PV{i + 1:02d}",
            kind="pv",
            base_kw=0.0,
            peak_kw=round(float(pv_caps[i]), 3),
            provenance="pv_inverter",
        ))
    specs.append(NodeSpec("CLINIC", "load", base_kw=6.0, peak_kw=9.0,
                          provenance="submeter", params={"critical": 1.0}))
    specs.append(NodeSpec("COLDSTORE", "load", base_kw=3.0, peak_kw=7.0,
                          provenance="submeter", params={"cycling": 1.0}))
    specs.append(NodeSpec("PUMP", "load", base_kw=2.2, peak_kw=5.5,
                          provenance="submeter", params={"deadline": 1.0}))
    specs.append(NodeSpec("FEEDER_HEAD", "feeder_head", base_kw=0.0,
                          peak_kw=0.0, provenance="feeder_head"))
    return specs


def time_index(days, start="2026-07-01"):
    return pd.date_range(start=start, periods=days * STEPS_PER_DAY,
                         freq=f"{INTERVAL_MIN}min")


def daily_load_shape(idx, summer=True):
    hour = np.asarray(idx.hour + idx.minute / 60.0, dtype=float)
    weekend = np.asarray(idx.dayofweek >= 5, dtype=float)

    def gauss(x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    morning = 0.55 * gauss(hour, 8.0, 1.6)
    evening = 1.00 * gauss(hour, 20.5, 2.2)
    ac = (0.75 if summer else 0.25) * gauss(hour, 14.5, 2.6)
    shape = 0.30 + morning + evening + ac
    shape = shape * (1.0 - 0.18 * weekend) + 0.12 * weekend * gauss(hour, 13.0, 3.0)
    shape = shape / shape.max()
    return 0.15 + 0.85 * shape


def pv_shape(idx, capacity_kw, rng):
    hour = np.asarray(idx.hour + idx.minute / 60.0, dtype=float)
    daylight = np.clip(np.sin(np.pi * (hour - 6.0) / 12.0), 0.0, None)
    daylight[(hour < 6.0) | (hour > 18.0)] = 0.0
    cloud = 1.0 - 0.15 * rng.random(len(idx))
    return np.clip(capacity_kw * daylight * cloud, 0.0, None)


def power_factor(load_frac, rng, base=0.92):
    pf = base - 0.10 * load_frac + rng.normal(0, 0.015, len(load_frac))
    return np.clip(pf, 0.80, 0.99)


def voltage(feeder_loading, rng):
    droop = 0.06 * feeder_loading
    noise = rng.normal(0, 0.006, len(feeder_loading))
    return V_NOMINAL * (1.0 - droop) + V_NOMINAL * noise


def generate_baseline(days=30, seed=DEFAULT_SEED, summer=True,
                      n_households=24, n_pv=8):
    rng = np.random.default_rng(seed)
    idx = time_index(days)
    specs = build_node_specs(n_households, n_pv)
    agg_shape = daily_load_shape(idx, summer=summer)

    P_by_node = {}
    for s in specs:
        if s.kind == "household":
            diversity = 0.75 + 0.5 * rng.random(len(idx))
            load_frac = np.clip(agg_shape * diversity, 0, 1)
            P = s.peak_kw * load_frac * (1 + rng.normal(0, 0.03, len(idx)))
            P_by_node[s.node_id] = np.clip(P, 0.0, None)
        elif s.kind == "pv":
            P_by_node[s.node_id] = -pv_shape(idx, s.peak_kw, rng)
        elif s.kind == "load":
            base_pat = agg_shape * s.base_kw
            if s.params.get("cycling"):
                duty = (np.sin(np.arange(len(idx)) * np.pi / 8.0) > 0).astype(float)
                base_pat = base_pat * (0.4 + 0.6 * duty)
            if s.params.get("deadline"):
                hour = idx.hour + idx.minute / 60.0
                run = ((hour > 5) & (hour < 7)) | ((hour > 17) & (hour < 19))
                base_pat = np.where(run, s.peak_kw, 0.1 * s.base_kw)
            P_by_node[s.node_id] = np.clip(
                base_pat * (1 + rng.normal(0, 0.02, len(idx))), 0, None)

    loads = [v for k, v in P_by_node.items() if not k.startswith("PV")]
    net_load = np.sum(loads, axis=0)
    pv_gen = np.sum([v for k, v in P_by_node.items() if k.startswith("PV")], axis=0)
    head_P = net_load + pv_gen
    feeder_cap = sum(s.peak_kw for s in specs if s.kind in ("household", "load"))
    feeder_loading = np.clip(net_load / max(feeder_cap, 1e-6), 0, 1)

    records = []
    for s in specs:
        V = voltage(feeder_loading, rng)
        if s.kind == "feeder_head":
            P = head_P
            frac = np.clip(P / max(feeder_cap, 1e-6), 0, 1)
        else:
            P = P_by_node[s.node_id]
            frac = np.clip(np.abs(P) / max(s.peak_kw, 1e-6), 0, 1)
        pf = power_factor(frac, rng)
        mag = np.abs(P)
        I = np.where(V > 1, (mag * 1000.0) / (V * pf), 0.0)
        f = F_NOMINAL + rng.normal(0, 0.02, len(idx))
        channels = {
            "V": V,
            "I": I,
            "P": P,
            "Q": P * np.tan(np.arccos(pf)),
            "PF": pf,
            "f": f,
        }
        for ch, arr in channels.items():
            records.append(pd.DataFrame({
                "timestamp": idx,
                "node_id": s.node_id,
                "kind": s.kind,
                "provenance": s.provenance,
                "channel": ch,
                "value": np.asarray(arr),
            }))

    df = pd.concat(records, ignore_index=True)
    df["value"] = df["value"].astype(float)
    return df


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "amanah_baseline_clean.csv")
    df = generate_baseline(days=30, seed=DEFAULT_SEED)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path}")
    print(f"  nodes   : {df['node_id'].nunique()}")
    print(f"  channels: {sorted(df['channel'].unique())}")
    print(f"  range   : {df['timestamp'].min()} -> {df['timestamp'].max()}")


if __name__ == "__main__":
    main()
