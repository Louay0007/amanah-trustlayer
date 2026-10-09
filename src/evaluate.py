from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import detect as D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(RES_DIR, "figures")
MANIFEST = os.path.join(ROOT, "data", "anomaly_manifest.csv")

os.makedirs(FIG_DIR, exist_ok=True)


def truth_category(row):
    if not row["is_anomaly"]:
        return "normal"
    if row["is_genuine_grid_event"]:
        return "genuine"
    return "data_quality"


def per_class_metrics(df):
    rows = []
    classes = sorted(df.loc[df["is_anomaly"], "anomaly_class"].unique())
    for cls in classes:
        sub = df[df["anomaly_class"] == cls]
        support = len(sub)
        flagged = int((sub["verdict"] != "NORMAL").sum())
        recall = flagged / support if support else 0.0
        if bool(sub["is_genuine_grid_event"].iloc[0]):
            target = "GENUINE_GRID_EVENT"
        else:
            target = "DATA_QUALITY_PROBLEM"
        verdict_ok = int((sub.loc[sub["verdict"] != "NORMAL", "verdict"] == target).sum())
        verr_acc = verdict_ok / flagged if flagged else 0.0
        rows.append({
            "anomaly_class": cls,
            "is_genuine_grid_event": bool(sub["is_genuine_grid_event"].iloc[0]),
            "support_cells": support,
            "flagged_cells": flagged,
            "recall": round(recall, 4),
            "verdict_target": target,
            "verdict_accuracy": round(verr_acc, 4),
        })
    return pd.DataFrame(rows)


def confusion(df):
    d = df.copy()
    d["truth"] = d.apply(truth_category, axis=1)
    verdict_order = ["NORMAL", "GENUINE_GRID_EVENT", "DATA_QUALITY_PROBLEM"]
    truth_order = ["normal", "genuine", "data_quality"]
    cm = pd.crosstab(d["truth"], d["verdict"])
    cm = cm.reindex(index=truth_order, columns=verdict_order, fill_value=0)
    return cm


def latency(df):
    dec = df[df["verdict"] != "NORMAL"][["timestamp", "node_id"]]
    man = pd.read_csv(MANIFEST, parse_dates=["start_ts", "end_ts"])
    inst = man.groupby(["anomaly_id", "node_id"], as_index=False).agg(
        start=("start_ts", "min"), end=("end_ts", "max"), cls=("class", "first"))
    inst = inst.merge(dec, on="node_id", how="left")
    inst = inst[(inst["timestamp"] >= inst["start"]) & (inst["timestamp"] <= inst["end"])]
    first_flag = inst.groupby(["anomaly_id", "node_id"], as_index=False)["timestamp"].min()
    inst = inst[["anomaly_id", "node_id", "start", "end", "cls"]].drop_duplicates().merge(
        first_flag, on=["anomaly_id", "node_id"], how="left")
    inst = inst.rename(columns={"timestamp": "first_flag_ts", "cls": "class"})
    inst["detected"] = inst["first_flag_ts"].notna()
    inst["delay_min"] = (inst["first_flag_ts"] - inst["start"]).dt.total_seconds() / 60.0
    node_df = inst[["anomaly_id", "node_id", "class", "start", "end", "first_flag_ts",
                    "delay_min", "detected"]].rename(columns={"start": "start_ts", "end": "end_ts"})

    ev = node_df.groupby("anomaly_id").agg(
        cls=("class", "first"), n_nodes=("node_id", "count"),
        n_nodes_detected=("detected", "sum"),
        min_delay_min=("delay_min", "min")).reset_index()
    ev = ev.rename(columns={"cls": "class"})
    ev["event_detected"] = ev["n_nodes_detected"] > 0
    ev_df = ev[["anomaly_id", "class", "n_nodes", "n_nodes_detected",
                "event_detected", "min_delay_min"]]

    detected = node_df[node_df["detected"]]
    summary = {
        "node_instances": int(len(node_df)),
        "node_detected": int(len(detected)),
        "node_recall": round(len(detected) / len(node_df), 4) if len(node_df) else 0.0,
        "delay_median_min": round(float(detected["delay_min"].median()), 2) if len(detected) else None,
        "delay_mean_min": round(float(detected["delay_min"].mean()), 2) if len(detected) else None,
        "delay_p95_min": round(float(detected["delay_min"].quantile(0.95)), 2) if len(detected) else None,
        "events": int(len(ev_df)),
        "events_detected": int(ev_df["event_detected"].sum()),
        "event_recall": round(float(ev_df["event_detected"].mean()), 4) if len(ev_df) else 0.0,
    }
    return node_df, ev_df, summary



def fig_timeline(lab, df):
    head = lab[(lab.node_id == "FEEDER_HEAD") & (lab.channel == "P")][["timestamp", "value"]]
    child = lab[(lab.node_id != "FEEDER_HEAD") & (lab.channel == "P")].groupby("timestamp")["value"].sum()
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(child.index, child.values, lw=0.8, color="#1f77b4", label="sum of child meters (P)")
    ax.plot(head["timestamp"], head["value"], lw=1.0, color="#2ca02c", label="feeder head (P)")
    gen = df[(df.verdict == "GENUINE_GRID_EVENT") & (df.channel == "P")]
    dq = df[(df.verdict == "DATA_QUALITY_PROBLEM") & (df.channel == "P")]

    def yvals(ts):
        return [child.reindex([t]).iloc[0] for t in ts]

    if len(gen):
        ax.scatter(gen["timestamp"], yvals(gen["timestamp"]), s=8, color="#ff7f0e",
                   zorder=5, label="flagged GENUINE")
    if len(dq):
        ax.scatter(dq["timestamp"], yvals(dq["timestamp"]), s=14, marker="x",
                   color="#d62728", zorder=6, label="flagged DATA-QUALITY")
    ax.set_title("AMANAH trust layer: genuine events stay physically consistent; data problems are flagged")
    ax.set_xlabel("time")
    ax.set_ylabel("active power P (kW)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "system_timeline.png"), dpi=130)
    plt.close(fig)


def fig_shap(shap_imp):
    items = {k: v for k, v in shap_imp.items() if isinstance(v, (int, float))}
    if not items:
        return
    names = list(items.keys())
    vals = [items[n] for n in names]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh(names[::-1], vals[::-1], color="#4c72b0")
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("Isolation-Forest feature importance (SHAP)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "shap_importance.png"), dpi=130)
    plt.close(fig)


def fig_confusion(cm):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(len(cm.columns)))
    ax.set_xticklabels(cm.columns, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(cm.index)))
    ax.set_yticklabels(cm.index, fontsize=8)
    ax.set_xlabel("model verdict")
    ax.set_ylabel("ground truth")
    ax.set_title("Verdict confusion matrix (test)")
    vmax = cm.values.max() if cm.values.size else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm.values[i, j]), ha="center", va="center",
                    color="white" if cm.values[i, j] > vmax * 0.5 else "black", fontsize=10)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "confusion_matrix.png"), dpi=130)
    plt.close(fig)


def fig_per_class(pc):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    colors = ["#2ca02c" if g else "#d62728" for g in pc["is_genuine_grid_event"]]
    ax.bar(pc["anomaly_class"], pc["recall"], color=colors)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("detection recall")
    ax.set_title("Per-class detection recall (green=genuine, red=data-quality)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    for i, v in enumerate(pc["recall"]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "per_class_recall.png"), dpi=130)
    plt.close(fig)


def main():
    lab = pd.read_csv(D.LABELED, parse_dates=["timestamp"])
    df = D.engineer_features(lab)
    times = np.sort(df["timestamp"].unique())
    cut = times[int(len(times) * D.TRAIN_FRAC)]
    df["is_train"] = (df["timestamp"] < cut)
    scaler, model = D.train_if(df, df["is_train"])
    df = D.apply_if(df, scaler, model)
    df = D.apply_verdicts(df)

    test_mask = ~df["is_train"]
    test = df[test_mask].copy()

    pc = per_class_metrics(test)
    cm = confusion(test)
    node_df, ev_df, lat = latency(df)

    pc.to_csv(os.path.join(RES_DIR, "per_class_metrics.csv"), index=False)
    cm.to_csv(os.path.join(RES_DIR, "confusion_matrix.csv"))
    node_df.to_csv(os.path.join(RES_DIR, "latency_by_instance.csv"), index=False)
    ev_df.to_csv(os.path.join(RES_DIR, "latency_by_event.csv"), index=False)

    shap_imp = D.shap_report(df, model, scaler)
    fig_timeline(lab, df)
    fig_shap(shap_imp)
    fig_confusion(cm)
    fig_per_class(pc)

    full = {
        "detection_test": D.score(df, test_mask, "test"),
        "separation_test": D.separation(df, test_mask),
        "per_class": pc.to_dict(orient="records"),
        "confusion_matrix": cm.to_dict(),
        "latency": lat,
        "if_feature_importance_shap": shap_imp,
    }
    with open(os.path.join(RES_DIR, "metrics_full.json"), "w") as fh:
        json.dump(full, fh, indent=2)

    print("=== per-class (test) ===")
    print(pc.to_string(index=False))
    print("\n=== confusion (test) ===")
    print(cm.to_string())
    print("\n=== latency ===")
    print(json.dumps(lat, indent=2))
    print("\nfigures ->", FIG_DIR)


if __name__ == "__main__":
    main()
