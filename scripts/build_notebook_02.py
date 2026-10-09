import nbformat as nbf
import os

NB_DIR = "notebooks"
os.makedirs(NB_DIR, exist_ok=True)


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9"},
}
c = []
c.append(md(
    "# 02 · Anomaly Generation & Trust-Layer Walkthrough\n"
    "**AMANAH Trust Layer — EnergyMind Tunisia (IEEE TSYP14)**\n\n"
    "The spec requires teams to **document how the anomalies were generated**. This notebook is that\n"
    "documentation: it reproduces the labeled dataset, shows what each of the **five Goal-1 anomaly\n"
    "classes** looks like in the signal, then runs the trust layer (`src/detect.py`) and shows its\n"
    "3-way verdict and auditable decision log.\n\n"
    "**To regenerate from scratch:** `python src/simulate.py && python src/inject.py`\n\n"
    "**Generation is deterministic** — fixed seed `2026`, recorded per row in `data/anomaly_manifest.csv`."))
c.append(code(
    "import sys, os\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "sys.path.insert(0, os.path.abspath('..'))\n"
    "sys.path.insert(0, os.path.abspath('../src'))\n"
    "import detect as D\n"
    "pd.set_option('display.width', 140)\n"
    "pd.set_option('display.max_columns', 30)\n"
    "LABELED = D.LABELED\n"
    "MANIFEST = os.path.abspath('../data/anomaly_manifest.csv')\n"
    "if not (os.path.exists(LABELED) and os.path.exists(MANIFEST)):\n"
    "    print('labeled data missing -> running src/simulate.py + src/inject.py ...')\n"
    "    os.system('cd .. && python src/simulate.py && python src/inject.py')\n"
    "print('labeled exists :', os.path.exists(LABELED))\n"
    "print('manifest exists:', os.path.exists(MANIFEST))"))
c.append(md(
    "## 1 · The two-frame design (why it works)\n\n"
    "Anomalies are injected in two regimes, and the difference is the *whole point* of the method:\n\n"
    "- **Genuine grid events** (equipment fault, natural failure, heat surge, cloud dip, EV block,\n"
    "  maintenance) are applied to a *true* frame and the **feeder head is recomputed from children**.\n"
    "  They stay **physically consistent** (`|P|=V·I·PF` and head = Σ children still hold).\n"
    "- **Data-quality problems** (drift, bias, corruption) are applied to the *observation* only.\n"
    "  They **break a physical invariant** — the meter no longer sums to the independent feeder head.\n\n"
    "So the verdict reduces to a clean, interpretable rule: **broken physics ⇒ DATA_QUALITY; a\n"
    "statistical excursion that stays physically consistent ⇒ GENUINE_GRID_EVENT.**"))
c.append(code(
    "lab = pd.read_csv(LABELED, parse_dates=['timestamp'])\n"
    "print('labeled rows  :', len(lab))\n"
    "print('base rate     : {:.2f}%'.format(100*lab.is_anomaly.mean()))\n"
    "print('NaN values    :', int(lab.value.isna().sum()))\n"
    "print('\\nclass x verdict flags:')\n"
    "print(lab[lab.is_anomaly].groupby(['anomaly_class','is_genuine_grid_event']).size().to_string())\n"
    "man = pd.read_csv(MANIFEST)\n"
    "print('\\nmanifest rows :', len(man), '| unique anomalies:', man.anomaly_id.nunique(), '| seed:', man.seed.iloc[0])"))
c.append(md(
    "## 2 · What each anomaly class looks like\n"
    "Below we overlay the injected value against the clean baseline for one concrete example of each\n"
    "class. This is the visual documentation of the generation mechanism."))
c.append(code(
    "clean = pd.read_csv(os.path.abspath('../data/raw/amanah_baseline_clean.csv'), parse_dates=['timestamp'])\n"
    "examples = ['equipment_fault','natural_failure','genuine_event','degraded_sensor','biased_reading','corrupted_reading']\n"
    "fig, axes = plt.subplots(2, 3, figsize=(15, 7))\n"
    "for ax, cls in zip(axes.ravel(), examples):\n"
    "    row = man[man['class']==cls]\n"
    "    if len(row)==0:\n"
    "        ax.set_title(cls+' (none)'); ax.axis('off'); continue\n"
    "    r = row.iloc[0]; nid=r.node_id; ch=r.channel\n"
    "    t0=pd.Timestamp(r.start_ts); t1=pd.Timestamp(r.end_ts)\n"
    "    a = lab[(lab.node_id==nid)&(lab.channel==ch)].set_index('timestamp').value\n"
    "    b = clean[(clean.node_id==nid)&(clean.channel==ch)].set_index('timestamp').value\n"
    "    win = slice(t0-pd.Timedelta(hours=8), t1+pd.Timedelta(hours=8))\n"
    "    ax.plot(b[win].index, b[win].values, color='#999', lw=1, label='clean')\n"
    "    ax.plot(a[win].index, a[win].values, color='#d62728', lw=1.2, label='injected')\n"
    "    ax.axvspan(t0, t1, color='#d62728', alpha=0.08)\n"
    "    ax.set_title(f'{cls} · {ch} · {nid}', fontsize=9); ax.tick_params(labelsize=7)\n"
    "    if ax is axes[0][0]: ax.legend(fontsize=7)\n"
    "fig.suptitle('Clean vs injected signal per anomaly class (shaded = injected window)', fontsize=11)\n"
    "plt.tight_layout(); plt.show()"))
c.append(md(
    "## 3 · Run the trust layer\n"
    "`detect.engineer_features` builds the interpretable features; `detect.apply_verdicts` assigns the\n"
    "3-way verdict (`NORMAL` / `GENUINE_GRID_EVENT` / `DATA_QUALITY_PROBLEM`) with a plain-language\n"
    "reason and a confidence score. Everything below runs on the real pipeline code."))
c.append(code(
    "df = D.engineer_features(lab)\n"
    "df = D.apply_verdicts(df)\n"
    "flagged = df[df.verdict != 'NORMAL']\n"
    "print('cells scored   :', len(df))\n"
    "print('cells flagged  :', len(flagged), '({:.2f}%)'.format(100*len(flagged)/len(df)))\n"
    "print('\\nverdict breakdown:')\n"
    "print(df.verdict.value_counts().to_string())\n"
    "print('\\nfeatures used :', D.FEATURES)"))

c.append(md(
    "## 4 · Auditable decision log (explainability, Goal 2)\n"
    "Every flagged cell carries what was detected, the verdict, **why**, and a confidence — exactly the\n"
    "auditable record the spec asks for. A sample:"))
c.append(code(
    "cols = ['timestamp','node_id','channel','value','verdict','confidence','reason']\n"
    "sample = flagged.sort_values('confidence', ascending=False)[cols].head(12)\n"
    "for _, r in sample.iterrows():\n"
    "    print(f\"{r.timestamp}  {r.node_id:<14} {r.channel:<2} \"\n"
    "          f\"{r.value:9.2f}  ->  {r.verdict:<22} conf={r.confidence:.2f}\")\n"
    "    print(f\"    why: {r.reason}\")"))
c.append(md(
    "## 5 · Per-class detection quality\n"
    "Detection recall per injected class, and the **verdict accuracy** (did the trust layer put each\n"
    "class on the correct side of the genuine / data-quality boundary?). Full metrics live in\n"
    "`results/` (produced by `src/evaluate.py`)."))
c.append(code(
    "truth = flagged[flagged.is_anomaly]\n"
    "print('flagged & truly anomalous:', len(truth))\n"
    "print('\\nrecall per class (flagged cells / injected cells):')\n"
    "g = truth.groupby('anomaly_class').size()\n"
    "injected = lab[lab.is_anomaly].groupby('anomaly_class').size()\n"
    "for cls, n in g.items():\n"
    "    tot = int(injected.get(cls, 0))\n"
    "    print(f'  {cls:<20} flagged {n:>4}  / injected {tot:>4}')\n"
    "verdict_ok = truth[\n"
    "    ((truth.is_genuine_grid_event) & (truth.verdict=='GENUINE_GRID_EVENT')) |\n"
    "    ((~truth.is_genuine_grid_event) & (truth.verdict=='DATA_QUALITY_PROBLEM'))]\n"
    "print('\\nverdict accuracy on flagged anomalies: {:.3f}'.format(len(verdict_ok)/max(1,len(truth))))\n"
    "print('  correct genuine vs data-quality split:', len(verdict_ok), '/', len(truth))"))
c.append(md(
    "## 6 · Headline results\n"
    "From `results/metrics_full.json` (held-out test split, no leakage — scaler and Isolation Forest\n"
    "fit on the training window only):"))
c.append(code(
    "import json\n"
    "mp = os.path.abspath('../results/metrics_full.json')\n"
    "if os.path.exists(mp):\n"
    "    m = json.load(open(mp))\n"
    "    det = m.get('detection', {})\n"
    "    lat = m.get('latency', {})\n"
    "    print('precision :', det.get('precision'))\n"
    "    print('recall    :', det.get('recall'))\n"
    "    print('f1        :', det.get('f1'))\n"
    "    print('FPR       :', det.get('false_positive_rate'))\n"
    "    print('node recall (any injected node caught):', lat.get('node_recall'))\n"
    "    print('event recall (any injected event caught):', lat.get('event_recall'))\n"
    "    print('detection delay median (min):', lat.get('delay_median_min'), '| p95 (min):', lat.get('delay_p95_min'))\n"
    "else:\n"
    "    print('run src/evaluate.py first to produce metrics_full.json')"))
c.append(md(
    "---\n"
    "**Takeaways**\n"
    "- Five Goal-1 anomaly classes + genuine Tunisian events, generated deterministically and fully documented.\n"
    "- The provenance-aware physical-consistency check separates *genuine events* from *data-quality\n"
    "  problems* — the innovation — with clean, explainable evidence.\n"
    "- Next: escalation/governance design + operator interface in Phase 2 (see `docs/PROPOSAL.md`)."))
nb.cells = c
with open(os.path.join(NB_DIR, "02_anomaly_injection.ipynb"), "w") as fh:
    nbf.write(nb, fh)
print("wrote 02_anomaly_injection.ipynb with", len(c), "cells")

