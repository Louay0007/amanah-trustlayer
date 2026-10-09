from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LABELED = os.path.join(ROOT, "data", "processed", "amanah_labeled.csv")
MANIFEST = os.path.join(ROOT, "data", "anomaly_manifest.csv")

fails = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


df = pd.read_csv(LABELED, parse_dates=["timestamp"])
man = pd.read_csv(MANIFEST, parse_dates=["start_ts", "end_ts"])

check("no NaN values", df["value"].isna().sum() == 0)
check("all 5 Goal-1 classes present",
      set(["equipment_fault", "degraded_sensor", "biased_reading",
           "corrupted_reading", "natural_failure"]).issubset(set(man["class"].unique())))
check("genuine events present", (man["is_genuine_grid_event"]).any())
check("data-quality problems present", (~man["is_genuine_grid_event"]).any())

genuine_classes = set(man.loc[man["is_genuine_grid_event"], "class"].unique())
dq_classes = set(man.loc[~man["is_genuine_grid_event"], "class"].unique())
check("genuine flags = {equipment_fault, natural_failure, genuine_event}",
      genuine_classes == {"equipment_fault", "natural_failure", "genuine_event"},
      f"got {genuine_classes}")
check("dq flags = {degraded_sensor, biased_reading, corrupted_reading}",
      dq_classes == {"degraded_sensor", "biased_reading", "corrupted_reading"},
      f"got {dq_classes}")

clean = df[~df["is_anomaly"]]
wide = clean.pivot_table(index=["timestamp", "node_id"], columns="channel", values="value").reset_index()
w = wide.dropna(subset=["V", "I", "P", "PF"])
err = np.abs(w["P"].abs() * 1000.0 - w["V"] * w["I"] * w["PF"])
check("clean cells obey |P|=V*I*PF/1000 (max err < 0.01)", err.max() < 0.01,
      f"max err {err.max():.6f}")

obs = df.pivot_table(index=["timestamp", "node_id"], columns="channel", values="value").reset_index()
child = obs[obs["node_id"] != "FEEDER_HEAD"]
net = child.groupby("timestamp")["P"].sum()
head = obs[obs["node_id"] == "FEEDER_HEAD"].set_index("timestamp")["P"]
resid = (head - net).abs()
ts_index = resid.index


def mean_resid(windows):
    if windows is None or len(windows) == 0:
        return 0.0
    m = np.zeros(len(resid), dtype=bool)
    for a, b in windows:
        m |= (ts_index >= a) & (ts_index <= b)
    return float(resid[m].mean()) if m.any() else 0.0


genuine_windows = list(man.loc[man["is_genuine_grid_event"], ["start_ts", "end_ts"]].itertuples(index=False))
dq_p_windows = list(man.loc[(~man["is_genuine_grid_event"]) & (man["channel"] == "P"),
                            ["start_ts", "end_ts"]].itertuples(index=False))
all_win = genuine_windows + dq_p_windows

g_r = mean_resid(genuine_windows)
q_r = mean_resid(dq_p_windows)
covered = np.zeros(len(resid), dtype=bool)
for a, b in all_win:
    covered |= (ts_index >= a) & (ts_index <= b)
base_r = float(resid[~covered].mean())

print(f"\n  baseline residual (no P anomaly)      : {base_r:8.4f} kW")
print(f"  mean residual @ genuine events        : {g_r:8.4f} kW")
print(f"  mean residual @ P-channel data-quality: {q_r:8.4f} kW")

check("genuine events leave head coherent (residual stays ~0)", g_r < 0.5,
      f"genuine {g_r:.4f}")
check("P-channel data-quality inflates head residual", q_r > base_r + 0.1,
      f"dq {q_r:.4f} vs base {base_r:.4f}")

print()
if fails:
    print(f"RESULT: {len(fails)} CHECK(S) FAILED -> {fails}")
    raise SystemExit(1)
print("RESULT: ALL CHECKS PASSED")
