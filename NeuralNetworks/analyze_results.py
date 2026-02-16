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

# 2) Filename patterns for CNN/Perceptron, RNN and CNN logs
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
    # CNN pattern: train_data_..._BCE_kernel10_stride2_dropconv6_dropdense10_lr6e-03_bs128.tsv
    re.compile(
        r"(?P<dataset>.+?)_"
        r"(?P<mode>[^_]+)_"
        r"kernel(?P<h1>\d+)_"
        r"stride(?P<h2>\d+)_"
        r"dropconv(?P<drop1>\d+)_"
        r"dropdense(?P<drop2>\d+)_"
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
        "best_epoch":           int(best["epoch"]),  # type: ignore
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
best_all.to_frame().T.to_csv(OUTPUT_DIR / "overall_best.csv", index=False)  # type: ignore

top_k = results.nlargest(10, "best_balanced_acc")
top_k.to_csv(OUTPUT_DIR / "top_10.csv", index=False)

# 6) Plotting
sns.set(style="whitegrid", font_scale=1.1)


def savefig_dynamic(plt, filename, extra_adjust=True):
    """Save figure with dynamic adjustments to prevent label overlap."""
    plt.tight_layout()
    if extra_adjust:
        plt.savefig(filename, bbox_inches='tight', dpi=150)
    else:
        plt.savefig(filename, dpi=150)
    plt.close()


def get_boxplot_figsize(data, x_col):
    """Calculate dynamic figure size based on number of categories."""
    n_cats = data[x_col].nunique()
    width = max(8, n_cats * 2)
    height = 6
    return (width, height)


# Boxplot of best balanced accuracy per network
figsize = get_boxplot_figsize(results, "network")
plt.figure(figsize=figsize)
ax = sns.boxplot(data=results, x="network", y="best_balanced_acc", hue="network", palette="mako", legend=False)
plt.title("Best Balanced Accuracy per Network")
plt.xticks(rotation=30, ha='right')
# Add margins to y-axis to prevent "squishing"
y_min, y_max = results["best_balanced_acc"].min(), results["best_balanced_acc"].max()
y_margin = (y_max - y_min) * 0.1
plt.ylim(y_min - y_margin, y_max + y_margin)
savefig_dynamic(plt, PLOTS_DIR / "balanced_acc_boxplot.png")

# Boxplot of best F1 score per network
plt.figure(figsize=figsize)
ax = sns.boxplot(data=results, x="network", y="best_val_f1", hue="network", palette="mako", legend=False)
plt.title("Best F1 Score per Network")
plt.xticks(rotation=30, ha='right')
y_min, y_max = results["best_val_f1"].min(), results["best_val_f1"].max()
y_margin = (y_max - y_min) * 0.1
plt.ylim(y_min - y_margin, y_max + y_margin)
savefig_dynamic(plt, PLOTS_DIR / "best_val_f1_boxplot.png")

# Boxplot of best balanced accuracy per loss function
figsize = get_boxplot_figsize(results, "mode")
plt.figure(figsize=figsize)
ax = sns.boxplot(data=results, x="mode", y="best_balanced_acc", hue="mode", palette="viridis", legend=False)
plt.title("Best Balanced Accuracy per Loss Function")
plt.xticks(rotation=45, ha='right')
y_min, y_max = results["best_balanced_acc"].min(), results["best_balanced_acc"].max()
y_margin = (y_max - y_min) * 0.1
plt.ylim(y_min - y_margin, y_max + y_margin)
savefig_dynamic(plt, PLOTS_DIR / "balanced_acc_per_loss_boxplot.png")

# Boxplot of best F1 score per loss function
plt.figure(figsize=figsize)
ax = sns.boxplot(data=results, x="mode", y="best_val_f1", hue="mode", palette="viridis", legend=False)
plt.title("Best F1 Score per Loss Function")
plt.xticks(rotation=45, ha='right')
y_min, y_max = results["best_val_f1"].min(), results["best_val_f1"].max()
y_margin = (y_max - y_min) * 0.1
plt.ylim(y_min - y_margin, y_max + y_margin)
savefig_dynamic(plt, PLOTS_DIR / "f1_per_loss_boxplot.png")

# Boxplot of best balanced accuracy per network and loss function
n_nets = results["network"].nunique()
n_modes = results["mode"].nunique()
figsize = (max(10, n_nets * 3), max(6, n_modes * 2))
plt.figure(figsize=figsize)
sns.boxplot(data=results, x="network", y="best_balanced_acc", hue="mode", palette="viridis")
plt.title("Best Balanced Accuracy per Network and Loss Function")
plt.legend(title="Loss Function", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=30, ha='right')
savefig_dynamic(plt, PLOTS_DIR / "balanced_acc_network_loss_boxplot.png")

# Violin plot of best F1 per loss function
plt.figure(figsize=figsize)
ax = sns.violinplot(data=results, x="mode", y="best_val_f1", hue="mode", palette="viridis", legend=False)
plt.title("Distribution of Best F1 Score per Loss Function")
plt.xticks(rotation=45, ha='right')
y_min, y_max = results["best_val_f1"].min(), results["best_val_f1"].max()
y_margin = (y_max - y_min) * 0.1
plt.ylim(y_min - y_margin, y_max + y_margin)
savefig_dynamic(plt, PLOTS_DIR / "f1_per_loss_violin.png")

# Pairplot of all hyperparameters vs balanced accuracy
pair_cols = ["h1", "h2", "drop1", "drop2", "lr", "bs", "best_balanced_acc"]
n_vars = len(pair_cols)
figsize_pair = (4 * min(n_vars, 5), 4 * min(n_vars, 5))
g = sns.pairplot(results[pair_cols + ["network"]], hue="network", diag_kind="hist", height=4, aspect=1)
g.figure.set_size_inches(figsize_pair[0], figsize_pair[1])
plt.savefig(PLOTS_DIR / "hyperparams_pairplot.png", bbox_inches='tight')
plt.close()

# Full correlation matrix
num_cols = [
    "h1", "h2", "drop1", "drop2", "lr", "bs",
    "best_val_accuracy", "best_val_precision",
    "best_val_recall", "best_val_neg_recall",
    "best_val_f1", "best_balanced_acc"
]
corr = results[num_cols].corr()
n_corr = len(num_cols)
figsize_corr = (max(8, n_corr * 0.8), max(6, n_corr * 0.6))
# Dynamic font size based on matrix size
annot_size = 8 if n_corr <= 10 else 6 if n_corr <= 14 else 5
plt.figure(figsize=figsize_corr)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": annot_size})
plt.title("Full Correlation Matrix", fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=annot_size)
plt.yticks(fontsize=annot_size)
savefig_dynamic(plt, PLOTS_DIR / "full_corr_matrix.png", extra_adjust=False)

# Analysis of common hyperparameters across all networks
common = ["lr", "bs", "drop1", "drop2"]
g = sns.pairplot(results[common + ["best_balanced_acc", "network"]], hue="network", diag_kind="hist", height=3, aspect=1)
g.figure.set_size_inches(10, 10)
plt.savefig(PLOTS_DIR / "common_hyperparams_pairplot.png", bbox_inches='tight')
plt.close()

figsize_parallel = (max(8, len(common) * 2), 6)
plt.figure(figsize=figsize_parallel)
parallel_coordinates(
    results[["network"] + common],
    "network",
    colormap="mako"
)
plt.title("Parallel Coordinates of Common Hyperparameters")
plt.xticks(rotation=45, ha='right')
savefig_dynamic(plt, PLOTS_DIR / "common_params_parallel_coords.png")

# Per-network parameter ↔ metric correlation matrices
perf_cols = [
    "best_val_accuracy", "best_val_precision",
    "best_val_recall", "best_val_neg_recall",
    "best_val_f1", "best_balanced_acc"
]
for net in results["network"].unique():
    sub = results[results["network"] == net]
    param_cols = [c for c in ["h1", "h2", "drop1", "drop2", "lr", "bs"] if not sub[c].isna().all()]
    n_params = len(param_cols) + len(perf_cols)
    m = sub[param_cols + perf_cols].corr()
    figsize_heatmap = (max(6, n_params * 0.7), max(5, n_params * 0.6))
    annot_size = 8 if n_params <= 8 else 6 if n_params <= 12 else 5
    plt.figure(figsize=figsize_heatmap)
    sns.heatmap(m, annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": annot_size})
    plt.title(f"{net} Parameter↔Metric Correlation", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=annot_size)
    plt.yticks(fontsize=annot_size)
    savefig_dynamic(plt, PLOTS_DIR / f"{net.lower()}_corr_matrix.png", extra_adjust=False)

print("Analysis complete!")
print("Results CSVs in", OUTPUT_DIR)
print("Plots in", PLOTS_DIR)
