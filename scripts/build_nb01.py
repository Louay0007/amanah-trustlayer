import nbformat as nbf

NB_DIR = "notebooks"
import os
os.makedirs(NB_DIR, exist_ok=True)


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


nb1 = nbf.v4.new_notebook()
nb1.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9"},
}
c1 = []
c1.append(md(
    "# 01 · Baseline Data Exploration\n"
    "**AMANAH Trust Layer — EnergyMind Tunisia (IEEE TSYP14)**\n\n"
    "This notebook explores the **clean baseline** produced by `src/simulate.py`: a physics-informed\n"
    "low-voltage smart-metered neighborhood feeder at 15-minute resolution for July (Tunisian summer peak).\n\n"
    "It documents the physical backbone that the anomaly injector (`02`) and the trust layer build on.\n\n"
    "**To regenerate:** `python src/simulate.py`"))
c1.append(code(
    "import sys, os\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "sys.path.insert(0, os.path.abspath('..'))\n"
    "pd.set_option('display.width', 120)\n"
    "pd.set_option('display.max_columns', 30)\n"
    "RAW = os.path.abspath('../data/raw/amanah_baseline_clean.csv')\n"
    "if not os.path.exists(RAW):\n"
    "    print('baseline missing -> running src/simulate.py to regenerate ...')\n"
    "    os.system('cd .. && python src/simulate.py')\n"
    "print('baseline exists:', os.path.exists(RAW))"))
c1.append(md(
    "## 1 · Load and shape\n"
    "Long format: one row per `(timestamp, node_id, channel)`. Channels are the signals Goal 1 names:\n"
    "voltage (V), current (I), active power (P), reactive power (Q), power factor (PF), frequency (f)."))
c1.append(code(
    "df = pd.read_csv(RAW, parse_dates=['timestamp'])\n"
    "print('rows          :', len(df))\n"
    "print('time range    :', df.timestamp.min(), '->', df.timestamp.max())\n"
    "print('nodes         :', df.node_id.nunique())\n"
    "print('channels      :', sorted(df.channel.unique()))\n"
    "print('NaN values    :', int(df.value.isna().sum()))\n"
    "print('\\nnode kinds:')\n"
    "print(df.groupby('kind').node_id.nunique().to_string())\n"
    "print('\\nprovenance tags:')\n"
    "print(df.groupby('provenance').node_id.nunique().to_string())"))
c1.append(md(
    "## 2 · Physical sanity checks\n"
    "Before trusting the baseline we verify three invariants that the trust layer will later rely on:\n"
    "1. per-node energy balance `|P| = V·I·PF`,\n"
    "2. the feeder head equals the net of all children (loads minus PV generation),\n"
    "3. realistic LV voltage / power-factor / frequency bands."))
c1.append(code(
    "wide = df.pivot_table(index=['timestamp','node_id'], columns='channel', values='value').reset_index()\n"
    "recon = (wide.P.abs() - wide.V*wide.I*wide.PF/1000.0).abs()\n"
    "child = df[(df.node_id!='FEEDER_HEAD')&(df.channel=='P')].groupby('timestamp').value.sum()\n"
    "head = df[(df.node_id=='FEEDER_HEAD')&(df.channel=='P')].set_index('timestamp').value\n"
    "imbalance = (head - child).abs()\n"
    "print('energy-balance |P|-|V·I·PF| max err :', round(recon.max(), 6))\n"
    "print('feeder-head - sum(children) max err :', round(imbalance.max(), 6))\n"
    "v = df[df.channel=='V'].value; pf = df[df.channel=='PF'].value; f = df[df.channel=='f'].value\n"
    "print('voltage  band :', round(v.min(),1), '-', round(v.max(),1), 'V')\n"
    "print('PF       band :', round(pf.min(),2), '-', round(pf.max(),2))\n"
    "print('freq     band :', round(f.min(),2), '-', round(f.max(),2), 'Hz')"))
c1.append(md(
    "## 3 · Daily load shape\n"
    "Aggregate neighborhood load shows the expected morning peak, evening peak, and a midday\n"
    "air-conditioning hump typical of a Tunisian summer day."))
c1.append(code(
    "agg = df[(df.node_id!='FEEDER_HEAD')&(df.channel=='P')].copy()\n"
    "agg['tod'] = agg.timestamp.dt.hour + agg.timestamp.dt.minute/60\n"
    "load = agg[agg.value>0].groupby(agg.tod.round(2)).value.sum()\n"
    "gen = agg[agg.value<0].groupby(agg.tod.round(2)).value.sum().abs()\n"
    "fig, ax = plt.subplots(figsize=(9,4))\n"
    "ax.plot(load.index, load.values, label='load (P>0)', color='#1f77b4')\n"
    "ax.plot(gen.index, gen.values, label='PV generation', color='#ff7f0e')\n"
    "ax.set_xlabel('hour of day'); ax.set_ylabel('kW (sum over meters)'); ax.legend()\n"
    "ax.set_title('Aggregate neighborhood: load and PV by time of day (July)')\n"
    "plt.tight_layout(); plt.show()"))
c1.append(md(
    "## 4 · Feeder head vs sum of children\n"
    "Net import at the feeder head should track the sum of children (load − PV). The tight overlay is\n"
    "the **physical reference** that lets the trust layer catch a meter lying: if one node's reading is\n"
    "wrong, it no longer sums to the (independently measured) head."))
c1.append(code(
    "fig, ax = plt.subplots(figsize=(12,4))\n"
    "ax.plot(child.index, child.values, lw=0.9, label='sum of children (net)', color='#1f77b4')\n"
    "ax.plot(head.index, head.values, lw=1.0, label='feeder head (independent)', color='#2ca02c')\n"
    "ax.set_xlabel('time'); ax.set_ylabel('P (kW)'); ax.legend(loc='upper right', fontsize=8)\n"
    "ax.set_title('Feeder head vs sum of children — the cross-source physical check')\n"
    "plt.tight_layout(); plt.show()"))
c1.append(md(
    "---\n"
    "**Baseline takeaways**\n"
    "- 30 days × 15-min × 36 nodes × 6 channels, zero NaNs.\n"
    "- Physical invariants hold to numerical precision, so any deviation introduced later is a *signal*.\n"
    "- The feeder head is an **independent observation** of the same reality the meters measure — the\n"
    "  foundation of the provenance-aware trust score.\n\n"
    "Next: **`02_anomaly_injection.ipynb`** documents how the five anomaly classes are generated."))
nb1.cells = c1
with open(os.path.join(NB_DIR, "01_data_exploration.ipynb"), "w") as fh:
    nbf.write(nb1, fh)
print("wrote 01_data_exploration.ipynb with", len(c1), "cells")
