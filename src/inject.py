from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

SEED = 2026
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "amanah_baseline_clean.csv")
PROC_DIR = os.path.join(ROOT, "data", "processed")
LABELED = os.path.join(PROC_DIR, "amanah_labeled.csv")
MANIFEST = os.path.join(ROOT, "data", "anomaly_manifest.csv")
CHANNELS = ["V", "I", "P", "Q", "PF", "f"]


def load_wide():
    df = pd.read_csv(RAW, parse_dates=["timestamp"])
    wide = df.pivot_table(index=["timestamp", "node_id"], columns="channel",
                          values="value").reset_index()
    meta = df[["timestamp", "node_id", "kind", "provenance"]].drop_duplicates()
    return meta.merge(wide, on=["timestamp", "node_id"], how="left")


class Injector:
    def __init__(self, wide):
        self.base = wide.reset_index(drop=True)
        self.rng = np.random.default_rng(SEED)
        kinds = self.base[["node_id", "kind"]].drop_duplicates().set_index("node_id")["kind"]
        self.hh = sorted(kinds[kinds == "household"].index)
        self.pv = sorted(kinds[kinds == "pv"].index)
        self.load = sorted(kinds[kinds == "load"].index)
        self.children = self.hh + self.pv + self.load
        self.ts = np.sort(self.base["timestamp"].unique())
        self.N = len(self.ts)
        self.peak = self.base.groupby("node_id")["P"].apply(lambda s: float(s.abs().max())).to_dict()
        self.prov = self.base[["node_id", "provenance"]].drop_duplicates().set_index("node_id")["provenance"].to_dict()
        self.occupied = {}
        self.manifest = []
        self._aid = 0

    def new_id(self):
        self._aid += 1
        return self._aid

    def sample(self, group, k):
        k = min(k, len(group))
        return list(self.rng.choice(group, size=k, replace=False))

    def free(self, nodes, channels, i0, i1):
        for n in nodes:
            for c in channels:
                for (a, b) in self.occupied.get((n, c), []):
                    if not (i1 < a or i0 > b):
                        return False
        return True

    def claim(self, nodes, channels, i0, i1):
        for n in nodes:
            for c in channels:
                self.occupied.setdefault((n, c), []).append((i0, i1))

    def place(self, nodes, channels, dur, tries=80):
        if dur >= self.N:
            dur = self.N
        for _ in range(tries):
            i0 = int(self.rng.integers(0, max(1, self.N - dur)))
            i1 = min(i0 + dur - 1, self.N - 1)
            if self.free(nodes, channels, i0, i1):
                self.claim(nodes, channels, i0, i1)
                return self.ts[i0], self.ts[i1]
        return None, None

    def mask(self, W, nodes, t0, t1):
        return W["node_id"].isin(nodes) & (W["timestamp"] >= t0) & (W["timestamp"] <= t1)

    def ramp(self, W, m, t0, t1):
        pos = ((W.loc[m, "timestamp"] - t0) / (t1 - t0)).to_numpy()
        return np.clip(pos, 0.0, 1.0)

    def recompute_I(self, W, m):
        V = W.loc[m, "V"].to_numpy()
        P = W.loc[m, "P"].to_numpy()
        PF = np.clip(W.loc[m, "PF"].to_numpy(), 0.05, 0.999)
        W.loc[m, "I"] = np.abs(P) * 1000.0 / (V * PF)

    def zero_out(self, W, m):
        for c in ("P", "I", "Q"):
            W.loc[m, c] = 0.0

    def recompute_head(self, W):
        child = W[W["node_id"] != "FEEDER_HEAD"]
        net = child.groupby("timestamp")["P"].sum()
        hm = (W["node_id"] == "FEEDER_HEAD").to_numpy()
        sub = W.loc[hm]
        P = sub["timestamp"].map(net).to_numpy()
        V = sub["V"].to_numpy()
        PF = np.clip(sub["PF"].to_numpy(), 0.05, 0.999)
        W.loc[hm, "P"] = P
        W.loc[hm, "I"] = np.abs(P) * 1000.0 / (V * PF)
        W.loc[hm, "Q"] = P * np.tan(np.arccos(PF))

    def record(self, aid, nodes, channels, cls, subtype, genuine, params, t0, t1):
        for n in nodes:
            for c in channels:
                self.manifest.append({
                    "anomaly_id": aid, "node_id": n, "channel": c, "class": cls,
                    "subtype": subtype, "start_ts": t0, "end_ts": t1,
                    "params": json.dumps(params), "provenance_source": self.prov[n],
                    "is_genuine_grid_event": bool(genuine), "seed": SEED,
                })

    def apply_genuine(self, W):
        self.ev_heat_surge(W)
        self.ev_pv_cloud_dip(W)
        self.ev_ev_block(W)
        self.ev_maintenance(W)
        self.fault_sag(W)
        self.fault_swell(W)
        self.fault_swell_pv(W)
        self.fault_inrush(W)
        self.fail_open_circuit(W)
        self.fail_total_loss_pv(W)
        self.fail_gradual(W)
        self.recompute_head(W)

    def apply_dataquality(self, W):
        self.dq_drift_mult(W)
        self.dq_drift_mult_v(W)
        self.dq_drift_add_pf(W)
        self.dq_stuck(W)
        self.dq_noise_rise(W)
        self.dq_gain(W)
        self.dq_offset(W)
        self.dq_sign_flip(W)
        self.dq_flatline(W)
        self.dq_random_noise(W)
        self.dq_forged_high(W)
        self.dq_forged_low(W)


    def ev_heat_surge(self, W):
        nodes = list(self.hh)
        t0, t1 = self.place(nodes, ["P", "I"], 4)
        if t0 is None:
            return
        m = self.mask(W, nodes, t0, t1)
        block = pd.Series({n: 1.5 * self.peak[n] for n in nodes})
        add = W.loc[m, "node_id"].map(block).to_numpy()
        W.loc[m, "P"] = W.loc[m, "P"].to_numpy() + add
        self.recompute_I(W, m)
        aid = self.new_id()
        self.record(aid, nodes, ["P", "I"], "genuine_event", "heat_surge", True,
                    {"add_peak_mult": 1.5, "n_nodes": len(nodes)}, t0, t1)

    def ev_pv_cloud_dip(self, W):
        nodes = list(self.pv)
        t0, t1 = self.place(nodes, ["P", "I"], 4)
        if t0 is None:
            return
        m = self.mask(W, nodes, t0, t1)
        W.loc[m, "P"] = W.loc[m, "P"] * 0.08
        self.recompute_I(W, m)
        aid = self.new_id()
        self.record(aid, nodes, ["P", "I"], "genuine_event", "pv_cloud_dip", True,
                    {"scale": 0.08, "n_nodes": len(nodes)}, t0, t1)

    def ev_ev_block(self, W):
        nodes = self.sample(self.hh, 1)
        t0, t1 = self.place(nodes, ["P", "I"], 4)
        if t0 is None:
            return
        m = self.mask(W, nodes, t0, t1)
        block = 2.0 * self.peak[nodes[0]]
        W.loc[m, "P"] = W.loc[m, "P"] + block
        self.recompute_I(W, m)
        aid = self.new_id()
        self.record(aid, nodes, ["P", "I"], "genuine_event", "ev_block_load", True,
                    {"add_kw": round(block, 3)}, t0, t1)

    def ev_maintenance(self, W):
        nodes = self.sample(self.load, 1)
        t0, t1 = self.place(nodes, ["P", "I", "Q"], 12)
        if t0 is None:
            return
        m = self.mask(W, nodes, t0, t1)
        self.zero_out(W, m)
        aid = self.new_id()
        self.record(aid, nodes, ["P", "I", "Q"], "genuine_event", "maintenance_outage",
                    True, {}, t0, t1)

    def fault_sag(self, W):
        nodes = self.sample(self.hh, 1)
        t0, t1 = self.place(nodes, ["V", "I"], 4)
        if t0 is None:
            return
        m = self.mask(W, nodes, t0, t1)
        W.loc[m, "V"] = W.loc[m, "V"] * 0.82
        self.recompute_I(W, m)
        aid = self.new_id()
        self.record(aid, nodes, ["V", "I"], "equipment_fault", "voltage_sag", True,
                    {"dip": 0.18}, t0, t1)

    def fault_swell(self, W):
        nodes = self.sample(self.hh, 1)
        t0, t1 = self.place(nodes, ["V", "I"], 4)
        if t0 is None:
            return
        m = self.mask(W, nodes, t0, t1)
        W.loc[m, "V"] = W.loc[m, "V"] * 1.12
        self.recompute_I(W, m)
        aid = self.new_id()
        self.record(aid, nodes, ["V", "I"], "equipment_fault", "voltage_swell", True,
                    {"swell": 0.12}, t0, t1)

    def fault_swell_pv(self, W):
        nodes = self.sample(self.pv, 1)
        t0, t1 = self.place(nodes, ["V", "I"], 4)
        if t0 is None:
            return
        m = self.mask(W, nodes, t0, t1)
        W.loc[m, "V"] = W.loc[m, "V"] * 0.85
        self.recompute_I(W, m)
        aid = self.new_id()
        self.record(aid, nodes, ["V", "I"], "equipment_fault", "voltage_sag", True,
                    {"dip": 0.15}, t0, t1)

    def fault_inrush(self, W):
        nodes = self.sample(self.hh, 1)
        t0, t1 = self.place(nodes, ["P", "I"], 3)
        if t0 is None:
            return
        m = self.mask(W, nodes, t0, t1)
        pos = self.ramp(W, m, t0, t1)
        W.loc[m, "P"] = W.loc[m, "P"] * (1.0 + 1.6 * (1.0 - pos))
        self.recompute_I(W, m)
        aid = self.new_id()
        self.record(aid, nodes, ["P", "I"], "equipment_fault", "inrush_spike", True,
                    {"peak_mult": 2.6}, t0, t1)

    def fail_open_circuit(self, W):
        nodes = self.sample(self.hh, 1)
        t0, t1 = self.place(nodes, ["P", "I", "Q"], 10)
        if t0 is None:
            return
        m = self.mask(W, nodes, t0, t1)
        self.zero_out(W, m)
        aid = self.new_id()
        self.record(aid, nodes, ["P", "I", "Q"], "natural_failure", "open_circuit",
                    True, {}, t0, t1)

    def fail_total_loss_pv(self, W):
        nodes = self.sample(self.pv, 1)
        t0, t1 = self.place(nodes, ["P", "I", "Q"], 10)
        if t0 is None:
            return
        m = self.mask(W, nodes, t0, t1)
        self.zero_out(W, m)
        aid = self.new_id()
        self.record(aid, nodes, ["P", "I", "Q"], "natural_failure", "total_loss",
                    True, {}, t0, t1)

    def fail_gradual(self, W):
        nodes = self.sample(self.load, 1)
        t0, t1 = self.place(nodes, ["P", "I"], 20)
        if t0 is None:
            return
        m = self.mask(W, nodes, t0, t1)
        pos = self.ramp(W, m, t0, t1)
        W.loc[m, "P"] = W.loc[m, "P"] * (1.0 - 0.8 * pos)
        self.recompute_I(W, m)
        aid = self.new_id()
        self.record(aid, nodes, ["P", "I"], "natural_failure", "gradual_degradation",
                    True, {"drop": 0.8}, t0, t1)



    def dq_drift_mult(self, W):
        nodes = self.sample(self.hh, 1)
        t0, t1 = self.place(nodes, ["P"], 24)
        if t0 is None:
            return
        m = self.mask(W, nodes, t0, t1)
        pos = self.ramp(W, m, t0, t1)
        W.loc[m, "P"] = W.loc[m, "P"] * (1.0 + 0.6 * pos)
        aid = self.new_id()
        self.record(aid, nodes, ["P"], "degraded_sensor", "drift_mult", False,
                    {"alpha": 0.6}, t0, t1)

    def dq_drift_mult_v(self, W):
        nodes = self.sample(self.hh, 1)
        t0, t1 = self.place(nodes, ["V"], 24)
        if t0 is None:
            return
        m = self.mask(W, nodes, t0, t1)
        pos = self.ramp(W, m, t0, t1)
        W.loc[m, "V"] = W.loc[m, "V"] * (1.0 + 0.10 * pos)
        aid = self.new_id()
        self.record(aid, nodes, ["V"], "degraded_sensor", "drift_mult", False,
                    {"alpha": 0.10}, t0, t1)

    def dq_drift_add_pf(self, W):
        nodes = self.sample(self.hh, 1)
        t0, t1 = self.place(nodes, ["PF"], 24)
        if t0 is None:
            return
        m = self.mask(W, nodes, t0, t1)
        pos = self.ramp(W, m, t0, t1)
        W.loc[m, "PF"] = np.clip(W.loc[m, "PF"] + 0.20 * pos, 0.0, 1.0)
        aid = self.new_id()
        self.record(aid, nodes, ["PF"], "degraded_sensor", "drift_add", False,
                    {"beta": 0.20}, t0, t1)

    def dq_stuck(self, W):
        nodes = self.sample(self.hh, 1)
        t0, t1 = self.place(nodes, ["I"], 16)
        if t0 is None:
            return
        m = self.mask(W, nodes, t0, t1)
        frozen = W.loc[m, "I"].iloc[0]
        W.loc[m, "I"] = frozen
        aid = self.new_id()
        self.record(aid, nodes, ["I"], "degraded_sensor", "stuck_at", False,
                    {"frozen": round(float(frozen), 3)}, t0, t1)

    def dq_noise_rise(self, W):
        nodes = self.sample(self.hh, 1)
        t0, t1 = self.place(nodes, ["V"], 16)
        if t0 is None:
            return
        m = self.mask(W, nodes, t0, t1)
        pos = self.ramp(W, m, t0, t1)
        sigma = 1.5 + 12.0 * pos
        noise = self.rng.normal(0.0, 1.0, int(m.sum())) * sigma
        W.loc[m, "V"] = W.loc[m, "V"] + noise
        aid = self.new_id()
        self.record(aid, nodes, ["V"], "degraded_sensor", "noise_rise", False,
                    {"sigma0": 1.5, "sigma1": 13.5}, t0, t1)

    def dq_gain(self, W):
        nodes = self.sample(self.hh, 1)
        t0, t1 = self.place(nodes, ["P"], 12)
        if t0 is None:
            return
        m = self.mask(W, nodes, t0, t1)
        W.loc[m, "P"] = W.loc[m, "P"] * 1.15
        aid = self.new_id()
        self.record(aid, nodes, ["P"], "biased_reading", "gain", False,
                    {"gain": 1.15}, t0, t1)

    def dq_offset(self, W):
        nodes = self.sample(self.hh, 1)
        t0, t1 = self.place(nodes, ["I"], 12)
        if t0 is None:
            return
        m = self.mask(W, nodes, t0, t1)
        W.loc[m, "I"] = W.loc[m, "I"] + 4.0
        aid = self.new_id()
        self.record(aid, nodes, ["I"], "biased_reading", "offset", False,
                    {"offset_a": 4.0}, t0, t1)

    def dq_sign_flip(self, W):
        nodes = self.sample(self.pv, 1)
        t0, t1 = self.place(nodes, ["P"], 8)
        if t0 is None:
            return
        m = self.mask(W, nodes, t0, t1)
        W.loc[m, "P"] = W.loc[m, "P"] * -1.0
        aid = self.new_id()
        self.record(aid, nodes, ["P"], "biased_reading", "sign_inversion", False,
                    {}, t0, t1)

    def dq_flatline(self, W):
        nodes = self.sample(self.hh, 1)
        t0, t1 = self.place(nodes, ["P"], 10)
        if t0 is None:
            return
        m = self.mask(W, nodes, t0, t1)
        const = W.loc[m, "P"].iloc[0]
        W.loc[m, "P"] = const
        aid = self.new_id()
        self.record(aid, nodes, ["P"], "corrupted_reading", "flatline", False,
                    {"const": round(float(const), 3)}, t0, t1)

    def dq_random_noise(self, W):
        nodes = self.sample(self.hh, 1)
        t0, t1 = self.place(nodes, ["I"], 8)
        if t0 is None:
            return
        m = self.mask(W, nodes, t0, t1)
        W.loc[m, "I"] = self.rng.uniform(0.0, 40.0, int(m.sum()))
        aid = self.new_id()
        self.record(aid, nodes, ["I"], "corrupted_reading", "random_noise", False,
                    {"lo": 0.0, "hi": 40.0}, t0, t1)

    def dq_forged_high(self, W):
        nodes = self.sample(self.hh, 1)
        t0, t1 = self.place(nodes, ["P", "I"], 10)
        if t0 is None:
            return
        m = self.mask(W, nodes, t0, t1)
        peak = self.peak[nodes[0]]
        W.loc[m, "P"] = self.rng.uniform(0.7 * peak, peak, int(m.sum()))
        self.recompute_I(W, m)
        aid = self.new_id()
        self.record(aid, nodes, ["P", "I"], "corrupted_reading", "forged_plausible",
                    False, {"band": [round(0.7 * peak, 2), round(peak, 2)]}, t0, t1)

    def dq_forged_low(self, W):
        nodes = self.sample(self.hh, 1)
        t0, t1 = self.place(nodes, ["P", "I"], 10)
        if t0 is None:
            return
        m = self.mask(W, nodes, t0, t1)
        peak = self.peak[nodes[0]]
        W.loc[m, "P"] = self.rng.uniform(0.05 * peak, 0.25 * peak, int(m.sum()))
        self.recompute_I(W, m)
        aid = self.new_id()
        self.record(aid, nodes, ["P", "I"], "corrupted_reading", "forged_plausible",
                    False, {"band": [round(0.05 * peak, 2), round(0.25 * peak, 2)]}, t0, t1)


def build_labeled(inj):
    true = inj.base.copy()
    inj.apply_genuine(true)
    obs = true.copy()
    inj.apply_dataquality(obs)

    long = obs.melt(id_vars=["timestamp", "node_id", "kind", "provenance"],
                    value_vars=CHANNELS, var_name="channel", value_name="value")
    long["is_anomaly"] = False
    long["anomaly_class"] = ""
    long["anomaly_subtype"] = ""
    long["anomaly_id"] = -1
    long["is_genuine_grid_event"] = False
    long["is_data_quality_problem"] = False

    for r in inj.manifest:
        m = ((long["node_id"] == r["node_id"]) & (long["channel"] == r["channel"]) &
             (long["timestamp"] >= r["start_ts"]) & (long["timestamp"] <= r["end_ts"]))
        long.loc[m, "is_anomaly"] = True
        long.loc[m, "anomaly_class"] = r["class"]
        long.loc[m, "anomaly_subtype"] = r["subtype"]
        long.loc[m, "anomaly_id"] = r["anomaly_id"]
        long.loc[m, "is_genuine_grid_event"] = r["is_genuine_grid_event"]
        long.loc[m, "is_data_quality_problem"] = not r["is_genuine_grid_event"]

    return long[["timestamp", "node_id", "kind", "provenance", "channel", "value",
                 "is_anomaly", "anomaly_class", "anomaly_subtype", "anomaly_id",
                 "is_genuine_grid_event", "is_data_quality_problem"]]


def main():
    os.makedirs(PROC_DIR, exist_ok=True)
    wide = load_wide()
    inj = Injector(wide)
    labeled = build_labeled(inj)
    labeled.to_csv(LABELED, index=False)

    man = pd.DataFrame(inj.manifest)
    man = man[["anomaly_id", "node_id", "channel", "class", "subtype", "start_ts",
               "end_ts", "params", "provenance_source", "is_genuine_grid_event", "seed"]]
    man = man.sort_values(["anomaly_id", "node_id", "channel"]).reset_index(drop=True)
    man.to_csv(MANIFEST, index=False)

    total = len(labeled)
    anom = int(labeled["is_anomaly"].sum())
    print(f"Wrote {total:,} labeled rows -> {LABELED}")
    print(f"Wrote {len(man):,} manifest rows -> {MANIFEST}")
    print(f"anomalies injected : {inj._aid}")
    print(f"anomalous cells    : {anom:,} ({100.0 * anom / total:.2f}% base rate)")
    print(f"NaN values         : {int(labeled['value'].isna().sum())}")
    print("class breakdown:")
    print(man.groupby(["class", "is_genuine_grid_event"]).size().to_string())


if __name__ == "__main__":
    main()

