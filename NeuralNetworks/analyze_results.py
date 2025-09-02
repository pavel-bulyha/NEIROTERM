import re
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1) Path setup
BASE_DIR   = Path(__file__).parent.resolve()
PLOTS_DIR  = BASE_DIR / "analysis_plots"
OUTPUT_DIR = BASE_DIR / "analysis_results"
PLOTS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# 2) Pattern for parsing log filenames
LOG_PATTERN = re.compile(
    r".*_(?P<neg>\d+)-(?P<pos>\d+)_"
    r"(?P<mode>[^_]+)_"
    r"h1(?P<h1>\d+)_h2(?P<h2>\d+)_"
    r"lr(?P<lr>[0-9eE\-\+\.]+)_"
    r"bs(?P<bs>\d+)\.tsv$"
)

records = []

# 3) Collect records from logs
for log_path in BASE_DIR.rglob("**/logs/*.tsv"):
    fname = log_path.name
    m = LOG_PATTERN.match(fname)
    if not m:
        print(f"Skip unparsed log: {fname}")
        continue

    gd = m.groupdict()
    neg, pos = int(gd["neg"]), int(gd["pos"])
    ratio = pos / neg
    mode = gd["mode"]
    h1, h2 = int(gd["h1"]), int(gd["h2"])
    lr, bs = float(gd["lr"]), int(gd["bs"])
    network = log_path.parents[1].name  # network folder above logs

    df = pd.read_csv(log_path, sep="\t")
    if not {"epoch", "val_recall", "val_neg_recall"}.issubset(df.columns):
        print(f"Missing cols in {fname}")
        continue

    # compute balanced accuracy and pick best epoch
    df["balanced_acc"] = (df["val_recall"] + df["val_neg_recall"]) / 2
    best = df.loc[df["balanced_acc"].idxmax()]

    records.append({
        "network": network,
        "neg": neg,
        "pos": pos,
        "ratio": ratio,
        "mode": mode,
        "h1": h1,
        "h2": h2,
        "lr": lr,
        "bs": bs,
        "best_epoch": int(best["epoch"]), # type: ignore
        "best_val_recall": best["val_recall"],
        "best_val_neg_recall": best["val_neg_recall"],
        "best_balanced_acc": best["balanced_acc"]
    })

# 4) DataFrame with all results
results = pd.DataFrame(records)
if results.empty:
    print("No valid logs found. Exiting.")
    exit(1)
results.to_csv(OUTPUT_DIR / "all_results.csv", index=False)

# 5) Best per-network and overall best
best_per_network = (
    results
    .loc[results.groupby("network")["best_balanced_acc"].idxmax()]
    .reset_index(drop=True)
)
best_per_network.to_csv(OUTPUT_DIR / "best_per_network.csv", index=False)

overall_best = results.loc[results["best_balanced_acc"].idxmax()]
overall_best.to_frame().T.to_csv(OUTPUT_DIR / "overall_best.csv", index=False) # type: ignore

# 6) Prepare for slope recording
slope_records = []

# 7) Plot style
sns.set(style="whitegrid", font_scale=1.1)

# 7.1 Best Balanced Accuracy per Network (no regression slope)
plt.figure(figsize=(8, 4))
sns.barplot(
    data=best_per_network,
    x="network", y="best_balanced_acc",
    hue="network", palette="mako", dodge=False
)
plt.xticks(rotation=45)
plt.title("Best Balanced Accuracy per Network")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "best_balanced_acc_per_network.png")
plt.close()

# 7.2 Class Imbalance vs Balanced Accuracy + slope
plt.figure(figsize=(6, 4))
sns.regplot(
    data=results,
    x="ratio", y="best_balanced_acc",
    scatter_kws={"s": 40, "alpha": 0.7},
    line_kws={"color": "red"}
)
plt.xlabel("Positive / Negative Ratio")
plt.ylabel("Best Balanced Accuracy")
plt.title("Class Imbalance vs Balanced Accuracy")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "ratio_vs_balanced_acc.png")
plt.close()

# calculate slope of linear fit for ratio_vs_balanced_acc
slope_ratio, _ = np.polyfit(results["ratio"], results["best_balanced_acc"], 1)
slope_ratio = round(slope_ratio, 8)
slope_records.append({
    "plot_name": "ratio_vs_balanced_acc",
    "slope": slope_ratio
})
print(f"Slope (dBalancedAcc/dRatio): {slope_ratio:.8f}")

# 7.3 Positive vs Negative Recall (no regression slope)
plt.figure(figsize=(6, 6))
sns.scatterplot(
    data=results,
    x="best_val_recall", y="best_val_neg_recall",
    hue="network", s=50
)
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("Positive Recall")
plt.ylabel("Negative Recall")
plt.title("Positive vs Negative Recall")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "recall_correlation.png")
plt.close()

# 8) Correlation matrix of hyperparameters and metric (no regression slope)
corr_matrix = results[["h1", "h2", "lr", "bs", "best_balanced_acc"]].corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "param_corr_matrix.png")
plt.close()

# 9) Scatter + linear trend for each hyperparameter by loss mode
for param in ["h1", "h2", "lr", "bs"]:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        data=results,
        x=param, y="best_balanced_acc",
        hue="mode", style="mode", s=50
    )

    # compute and record slope for each mode
    for mode in results["mode"].unique():
        subset = results[results["mode"] == mode]
        slope, _ = np.polyfit(subset[param], subset["best_balanced_acc"], 1)
        slope = round(slope, 8)
        slope_records.append({
            "plot_name": f"{param}_vs_balanced_acc_by_mode_{mode}",
            "slope": slope
        })
        print(f"Slope (dBalancedAcc/d{param}) in '{mode}': {slope:.8f}")

        sns.regplot(
            data=subset,
            x=param, y="best_balanced_acc",
            scatter=False, label=f"{mode} trend"
        )

    plt.title(f"{param} vs Balanced Accuracy by Loss Mode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{param}_vs_balanced_acc_by_mode.png")
    plt.close()

# 10) Save slope summary to CSV
slope_df = pd.DataFrame(slope_records)
slope_df.to_csv(OUTPUT_DIR / "slope_summary.csv", index=False)
print(f"Slope summary saved to: {OUTPUT_DIR/'slope_summary.csv'}")

print("Analysis complete!")
print(f"- All results:      {OUTPUT_DIR/'all_results.csv'}")
print(f"- Best per network: {OUTPUT_DIR/'best_per_network.csv'}")
print(f"- Overall best:     {OUTPUT_DIR/'overall_best.csv'}")
print(f"- Plots in:         {PLOTS_DIR}")
