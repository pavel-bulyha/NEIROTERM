#!/usr/bin/env python3
import re
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

# 1) Path setup
BASE_DIR   = Path(__file__).parent.resolve()
PLOTS_DIR  = BASE_DIR / "analysis_plots"
OUTPUT_DIR = BASE_DIR / "analysis_results"
PLOTS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# 2) Filename patterns for CNN/Perceptron and RNN logs
PATTERNS = [
    re.compile(
        r"(?P<dataset>.+?)_"
        r"(?P<mode>[^_]+)_"
        r"h1(?P<h1>\d+)_"
        r"h2(?P<h2>\d+)_"
        r"drop1(?P<drop1>\d+)_"
        r"drop2(?P<drop2>\d+)_"
        r"lr(?P<lr>[0-9eE\-\+\.]+)_"
        r"bs(?P<bs>\d+)\.tsv$"
    ),
    re.compile(
        r"(?P<dataset>.+?)_"
        r"(?P<mode>[^_]+)_"
        r"u(?P<h1>\d+)_"
        r"dr(?P<drop1>\d+)_"
        r"dd(?P<drop2>\d+)_"
        r"lr(?P<lr>[0-9eE\-\+\.]+)_"
        r"bs(?P<bs>\d+)\.tsv$"
    ),
]

def parse_log_filename(name: str):
    for pat in PATTERNS:
        m = pat.match(name)
        if not m:
            continue
        gd = m.groupdict()
        gd.setdefault("h2", None)
        return {
            "dataset": gd["dataset"],
            "mode":    gd["mode"],
            "h1":      int(gd["h1"]),
            "h2":      int(gd["h2"]) if gd["h2"] is not None else np.nan,
            "drop1":   int(gd["drop1"]) / 100.0,
            "drop2":   int(gd["drop2"]) / 100.0,
            "lr":      float(gd["lr"]),
            "bs":      int(gd["bs"])
        }
    return None

records = []

# 3) Parse logs and collect best metrics
for log_path in BASE_DIR.rglob("**/logs/*.tsv"):
    meta = parse_log_filename(log_path.name)
    if meta is None:
        continue

    # network name (folder above logs/)
    network = log_path.parents[1].name
    meta["network"] = network

    # extract class imbalance ratio from dataset name
    ratio_match = re.search(r"(\d+)-(\d+)", meta["dataset"])
    if not ratio_match:
        continue
    neg, pos = map(int, ratio_match.groups())
    meta["neg"] = neg
    meta["pos"] = pos
    meta["ratio"] = pos / neg

    df = pd.read_csv(log_path, sep="\t")
    required = {"epoch", "val_recall", "val_neg_recall", "val_precision", "val_accuracy"}
    if not required.issubset(df.columns):
        continue

    # compute additional metrics
    df["balanced_acc"] = (df["val_recall"] + df["val_neg_recall"]) / 2
    df["val_f1"] = 2 * df["val_recall"] * df["val_precision"] \
                   / (df["val_recall"] + df["val_precision"] + 1e-8)

    # select best epoch by balanced accuracy
    best = df.loc[df["balanced_acc"].idxmax()]

    records.append({
        **meta,
        "best_epoch":           int(best["epoch"]),
        "best_val_recall":      best["val_recall"],
        "best_val_neg_recall":  best["val_neg_recall"],
        "best_val_precision":   best["val_precision"],
        "best_val_accuracy":    best["val_accuracy"],
        "best_val_f1":          best["val_f1"],
        "best_balanced_acc":    best["balanced_acc"]
    })

# 4) Build DataFrame
results = pd.DataFrame(records)
if results.empty:
    print("No valid logs found. Exiting.")
    exit(1)

# 5) Save summary CSVs
results.to_csv(OUTPUT_DIR / "all_results_extended.csv", index=False)

best_net = results.loc[results.groupby("network")["best_balanced_acc"].idxmax()]
best_net.to_csv(OUTPUT_DIR / "best_per_network.csv", index=False)

best_ds = results.loc[results.groupby("dataset")["best_balanced_acc"].idxmax()]
best_ds.to_csv(OUTPUT_DIR / "best_per_dataset.csv", index=False)

best_all = results.loc[results["best_balanced_acc"].idxmax()]
best_all.to_frame().T.to_csv(OUTPUT_DIR / "overall_best.csv", index=False)

top_k = results.nlargest(10, "best_balanced_acc")
top_k.to_csv(OUTPUT_DIR / "top_10.csv", index=False)

# 6) Plotting
sns.set(style="whitegrid", font_scale=1.1)

# Boxplot of best balanced accuracy per network
plt.figure(figsize=(8, 4))
sns.boxplot(data=results, x="network", y="best_balanced_acc", palette="mako")
plt.title("Best Balanced Accuracy per Network")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "balanced_acc_boxplot.png")
plt.close()

# Boxplot of best F1 score per network
plt.figure(figsize=(8, 4))
sns.boxplot(data=results, x="network", y="best_val_f1", palette="mako")
plt.title("Best F1 Score per Network")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "best_val_f1_boxplot.png")
plt.close()

# Pairplot of all hyperparameters vs balanced accuracy
pair_cols = ["h1", "h2", "drop1", "drop2", "lr", "bs", "best_balanced_acc"]
sns.pairplot(results[pair_cols + ["network"]], hue="network", diag_kind="hist")
plt.savefig(PLOTS_DIR / "hyperparams_pairplot.png")
plt.close()

# Full correlation matrix
num_cols = [
    "h1", "h2", "drop1", "drop2", "lr", "bs",
    "best_val_accuracy", "best_val_precision",
    "best_val_recall", "best_val_neg_recall",
    "best_val_f1", "best_balanced_acc"
]
corr = results[num_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Full Correlation Matrix")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "full_corr_matrix.png")
plt.close()

# Analysis of common hyperparameters across all networks
common = ["lr", "bs", "drop1", "drop2"]
sns.pairplot(results[common + ["best_balanced_acc", "network"]],
             hue="network", diag_kind="hist")
plt.savefig(PLOTS_DIR / "common_hyperparams_pairplot.png")
plt.close()

plt.figure(figsize=(8, 6))
parallel_coordinates(
    results[["network"] + common],
    "network",
    colormap=sns.color_palette("mako", n_colors=results["network"].nunique())
)
plt.title("Parallel Coordinates of Common Hyperparameters")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "common_params_parallel_coords.png")
plt.close()

# Per-network parameter ↔ metric correlation matrices
perf_cols = [
    "best_val_accuracy", "best_val_precision",
    "best_val_recall", "best_val_neg_recall",
    "best_val_f1", "best_balanced_acc"
]
for net in results["network"].unique():
    sub = results[results["network"] == net]
    param_cols = [c for c in ["h1", "h2", "drop1", "drop2", "lr", "bs"] if not sub[c].isna().all()]
    m = sub[param_cols + perf_cols].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(m, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(f"{net} Parameter↔Metric Correlation")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{net.lower()}_corr_matrix.png")
    plt.close()

print("Analysis complete!")
print("Results CSVs in", OUTPUT_DIR)
print("Plots in", PLOTS_DIR)