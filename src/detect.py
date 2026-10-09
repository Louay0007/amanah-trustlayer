from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROC_DIR = os.path.join(ROOT, "data", "processed")
LABELED = os.path.join(PROC_DIR, "amanah_labeled.csv")
RES_DIR = os.path.join(ROOT, "results")
DECISIONS = os.path.join(RES_DIR, "decision_log.csv")
METRICS = os.path.join(RES_DIR, "detector_metrics.json")

CHANNELS = ["V", "I", "P", "Q", "PF", "f"]
FEATURES = ["z_score", "z_step", "flatline", "recon_err", "head_res", "peer_z"]
Z_THRESH = 3.5
ZSTEP_THRESH = 6.0
FLATLINE_THRESH = 4
RECON_EPS = 0.25
HEAD_EPS = 0.10
CONTAMINATION = 0.02
ROLL = 96
TRAIN_FRAC = 0.70
COORD_ABS = 4.0
COORD_FRAC = 0.30
COORD_MIN = 4


def engineer_features(lab):
    p = lab[lab.channel == "P"].set_index(["timestamp", "node_id"])["value"]
    v = lab[lab.channel == "V"].set_index(["timestamp", "node_id"])["value"]
    i = lab[lab.channel == "I"].set_index(["timestamp", "node_id"])["value"]
    pf = lab[lab.channel == "PF"].set_index(["timestamp", "node_id"])["value"]

    pf_c = pf.clip(0.05, 0.999)
    mag = p.abs()
    active = (mag > 0.05) & (i > 0.05)
    recon = {
        "P": (mag - (v * i * pf_c / 1000.0)).abs().rename("recon_err"),
        "I": (i - (mag * 1000.0 / (v * pf_c))).abs().rename("recon_err"),
        "V": (v - (mag * 1000.0 / (i * pf_c))).abs().rename("recon_err"),
        "PF": (pf_c - (mag * 1000.0 / (v * i)).clip(0.05, 0.999)).abs().rename("recon_err"),
    }
    for k in recon:
        recon[k] = recon[k].where(active, 0.0)

    head = lab[(lab.node_id == "FEEDER_HEAD") & (lab.channel == "P")].set_index("timestamp")["value"]
    child_sum = lab[(lab.node_id != "FEEDER_HEAD") & (lab.channel == "P")].groupby("timestamp")["value"].sum()
    head_res = (head - child_sum).abs().rename("head_res")

    df = lab.sort_values(["node_id", "channel", "timestamp"]).reset_index(drop=True)
    g = df.groupby(["node_id", "channel"])["value"]
    rm = g.transform(lambda s: s.shift(1).rolling(ROLL, min_periods=12).mean())
    rs = g.transform(lambda s: s.shift(1).rolling(ROLL, min_periods=12).std())
    df["roll_mean"] = rm
    df["roll_std"] = rs
    df["z_score"] = (df["value"] - df["roll_mean"]) / (df["roll_std"] + 1e-9)
    prev = g.shift(1)
    df["z_step"] = (df["value"] - prev).abs() / (df["roll_std"] + 1e-9)
    eq = (df["value"] == prev).astype(int).to_numpy()

    grp = (df["node_id"].astype("category").cat.codes.to_numpy().astype(np.int64)
           * len(CHANNELS)
           + pd.Categorical(df["channel"], categories=CHANNELS).codes.astype(np.int64))
    flat = np.zeros(len(df), dtype=int)
    c = 0
    prev_g = -1
    for k in range(len(df)):
        if eq[k] and grp[k] == prev_g:
            c += 1
        elif eq[k]:
            c = 1
        else:
            c = 0
        flat[k] = c
        prev_g = grp[k]
    df["flatline"] = flat

    rec_series = []
    for ch, rsr in recon.items():
        m = df["channel"] == ch
        mapped = rsr.reindex(df.loc[m, ["timestamp", "node_id"]].set_index(["timestamp", "node_id"]).index).to_numpy()
        sub = pd.Series(np.nan, index=df.index)
        sub.loc[m.to_numpy()] = mapped
        rec_series.append(sub)
    df["recon_err"] = pd.concat(rec_series, axis=1).max(axis=1).where(
        df["channel"].isin(list(recon.keys())), 0.0)

    node_ts = df[["timestamp", "node_id"]].drop_duplicates().reset_index(drop=True)
    node_ts["head_res"] = node_ts["timestamp"].map(head_res)
    df = df.merge(node_ts, on=["timestamp", "node_id"], how="left")

    p_src = lab[lab.channel == "P"][["timestamp", "node_id", "kind", "value"]].copy()
    p_src["peer_med"] = p_src.groupby(["timestamp", "kind"])["value"].transform("median")
    p_src["peer_z"] = (p_src["value"] - p_src["peer_med"]).abs() / (p_src["peer_med"].abs() + 1.0)
    peer_map = p_src.set_index(["timestamp", "node_id"])["peer_z"]
    df["peer_z"] = df.set_index(["timestamp", "node_id"]).index.map(peer_map)

    dev = df[(df.channel == "P") & (df.node_id != "FEEDER_HEAD")].copy()
    dev["big"] = (dev["peer_z"] > 0.5) | (dev["z_step"] > ZSTEP_THRESH) | (dev["z_score"].abs() > Z_THRESH)
    n_kids = df[df.node_id != "FEEDER_HEAD"].groupby("timestamp")["node_id"].nunique()
    coord_n = dev[dev["big"]].groupby("timestamp")["node_id"].nunique()
    coord_frac = (coord_n / n_kids).reindex(df["timestamp"].unique()).fillna(0.0)
    df["coordinated"] = (coord_frac.reindex(df["timestamp"]).to_numpy() >= COORD_FRAC) & \
                        (coord_n.reindex(df["timestamp"]).fillna(0).to_numpy() >= COORD_MIN)

    for c in ["recon_err", "head_res", "peer_z", "roll_std", "roll_mean", "z_step"]:
        df[c] = df[c].fillna(0.0)
    df["z_score"] = df["z_score"].fillna(0.0)
    return df


def top_deviant_rows(df):
    p = df[(df.channel == "P") & (df.node_id != "FEEDER_HEAD")].copy()
    p["az"] = p["z_score"].abs()
    bad_ts = df[(df.head_res > HEAD_EPS) & (df.channel == "P")]["timestamp"].unique()
    bad = set(pd.to_datetime(bad_ts))
    p = p[p["timestamp"].isin(bad)]
    if len(p) == 0:
        return []
    idx = p.groupby("timestamp")["az"].idxmax()
    return list(idx.values)


def recon_localized_rows(df):
    phys = df[df.channel.isin(["V", "I", "P", "PF"])].copy()
    phys["az"] = phys["z_score"].abs()
    phys = phys[phys["recon_err"] > RECON_EPS]
    if len(phys) == 0:
        return []
    return list(phys.groupby(["timestamp", "node_id"])["az"].idxmax().values)


def apply_verdicts(df):
    df = df.reset_index(drop=True)
    recon_rows = recon_localized_rows(df)
    head_rows = top_deviant_rows(df)

    verdict = pd.Series("NORMAL", index=df.index, dtype=object)
    reasons = pd.Series("", index=df.index, dtype=object)
    conf = pd.Series(0.0, index=df.index, dtype=float)

    stat_chan = df["channel"].isin(["V", "I", "P"])
    stat = stat_chan & ((df["z_score"].abs() > Z_THRESH) | (df["z_step"] > ZSTEP_THRESH))
    dq = pd.Series(False, index=df.index)
    if recon_rows:
        dq.iloc[recon_rows] = True
    if head_rows:
        dq.iloc[head_rows] = True

    genuine = stat & (~dq)

    verdict[genuine] = "GENUINE_GRID_EVENT"
    verdict[dq] = "DATA_QUALITY_PROBLEM"

    conf[dq] = 0.90
    conf[genuine] = np.clip(df.loc[genuine, "z_score"].abs() / 10.0, 0.5, 0.95)

    in_recon = pd.Series(False, index=df.index)
    if recon_rows:
        in_recon.iloc[recon_rows] = True
    in_head = pd.Series(False, index=df.index)
    if head_rows:
        in_head.iloc[head_rows] = True
    coord_mask = df["coordinated"] == True

    reasons[in_recon] = "V, I, P, PF physically inconsistent (energy balance violated)"
    reasons[in_head & ~in_recon] = "meter disagrees with feeder-head power balance (cross-source check)"
    reasons[coord_mask & genuine] = "many meters shift together = real network-wide event, physically consistent"
    reasons[genuine & (df["z_score"].abs() > Z_THRESH)] = "real excursion vs own history, but physically consistent"
    reasons[genuine & (df["z_step"] > ZSTEP_THRESH)] = "abrupt real step change, physically consistent"

    df["verdict"] = verdict
    df["reason"] = reasons
    df["confidence"] = conf.round(3)
    return df


def train_if(df, train_mask):
    X = df.loc[train_mask, FEATURES].to_numpy()
    scaler = StandardScaler().fit(X)
    model = IsolationForest(n_estimators=200, contamination=CONTAMINATION,
                            random_state=42, n_jobs=-1)
    model.fit(scaler.transform(X))
    return scaler, model


def apply_if(df, scaler, model):
    Xs = scaler.transform(df[FEATURES].to_numpy())
    df = df.copy()
    df["if_score"] = -model.score_samples(Xs)
    df["if_flag"] = model.predict(Xs) == -1
    return df


def shap_report(df, model, scaler):
    try:
        import shap
        flagged = df[df["verdict"] != "NORMAL"]
        if len(flagged) == 0:
            return {}
        sample = flagged.sample(min(2000, len(flagged)), random_state=42)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(scaler.transform(sample[FEATURES].to_numpy()))
        imp = np.abs(sv).mean(axis=0)
        order = np.argsort(imp)[::-1]
        return {FEATURES[k]: round(float(imp[k]), 4) for k in order}
    except Exception as exc:
        return {"error": str(exc)}


def write_decision_log(df):
    flagged = df[df["verdict"] != "NORMAL"].copy()
    cols = ["timestamp", "node_id", "kind", "provenance", "channel", "value",
            "verdict", "confidence", "reason", "z_score", "z_step", "flatline",
            "recon_err", "head_res", "if_flag", "coordinated",
            "is_anomaly", "anomaly_class", "anomaly_subtype", "anomaly_id",
            "is_genuine_grid_event", "is_data_quality_problem"]
    for c in cols:
        if c not in flagged.columns:
            flagged[c] = pd.NA
    flagged = flagged[cols].sort_values(["timestamp", "node_id", "channel"])
    flagged.to_csv(DECISIONS, index=False)
    return len(flagged)


def score(df, mask, label):
    sub = df[mask]
    flagged = sub[sub["verdict"] != "NORMAL"]
    truth = sub[sub["is_anomaly"]]
    tp = len(flagged[flagged["is_anomaly"]])
    fp = len(flagged[~flagged["is_anomaly"]])
    fn = len(truth[truth["verdict"] == "NORMAL"])
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / len(sub[~sub["is_anomaly"]]) if len(sub[~sub["is_anomaly"]]) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"split": label, "flagged": int(len(flagged)), "tp": int(tp), "fp": int(fp),
            "fn": int(fn), "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "false_positive_rate": round(fpr, 6)}


def separation(df, mask):
    sub = df[mask]
    fg = sub[sub["verdict"] == "GENUINE_GRID_EVENT"]
    dq = sub[sub["verdict"] == "DATA_QUALITY_PROBLEM"]
    fg_correct = int((fg["is_genuine_grid_event"]).sum()) if len(fg) else 0
    dq_correct = int((dq["is_data_quality_problem"]).sum()) if len(dq) else 0
    tot = len(fg) + len(dq)
    acc = (fg_correct + dq_correct) / tot if tot else 0.0
    return {"genuine_flagged": int(len(fg)), "dq_flagged": int(len(dq)),
            "genuine_correct": fg_correct, "dq_correct": dq_correct,
            "verdict_accuracy": round(acc, 4)}


def main():
    os.makedirs(RES_DIR, exist_ok=True)
    lab = pd.read_csv(LABELED, parse_dates=["timestamp"])
    df = engineer_features(lab)

    times = np.sort(df["timestamp"].unique())
    cut = times[int(len(times) * TRAIN_FRAC)]
    df["is_train"] = df["timestamp"] < cut

    scaler, model = train_if(df, df["is_train"])
    df = apply_if(df, scaler, model)
    df = apply_verdicts(df)

    n_log = write_decision_log(df)
    test_mask = ~df["is_train"]
    metrics = {
        "detection": score(df, test_mask, "test"),
        "separation": separation(df, test_mask),
        "if_feature_importance_shap": shap_report(df, model, scaler),
        "thresholds": {"z": Z_THRESH, "z_step": ZSTEP_THRESH, "flatline": FLATLINE_THRESH,
                       "recon_eps": RECON_EPS, "head_eps": HEAD_EPS},
        "train_frac": TRAIN_FRAC, "train_rows": int(df["is_train"].sum()),
        "test_rows": int(test_mask.sum()), "decision_log_rows": n_log,
    }
    with open(METRICS, "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

