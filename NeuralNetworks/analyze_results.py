import re
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1) Настройка путей
BASE_DIR   = Path(__file__).parent.resolve()
PLOTS_DIR  = BASE_DIR / "analysis_plots"
OUTPUT_DIR = BASE_DIR / "analysis_results"
PLOTS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# 2) Паттерн для разбора имён логов
LOG_PATTERN = re.compile(
    r".*_(?P<neg>\d+)-(?P<pos>\d+)_"
    r"(?P<mode>[^_]+)_"
    r"h1(?P<h1>\d+)_h2(?P<h2>\d+)_"
    r"lr(?P<lr>[0-9eE\-\+\.]+)_"
    r"bs(?P<bs>\d+)\.tsv$"
)

records = []

# 3) Собираем все записи из логов
for log_path in BASE_DIR.rglob("**/logs/*.tsv"):
    fname = log_path.name
    m = LOG_PATTERN.match(fname)
    if not m:
        print("Skip unparsed log:", fname)
        continue

    gd = m.groupdict()
    neg, pos = int(gd["neg"]), int(gd["pos"])
    ratio = pos / neg
    mode = gd["mode"]
    h1, h2 = int(gd["h1"]), int(gd["h2"])
    lr, bs = float(gd["lr"]), int(gd["bs"])
    network = log_path.parents[1].name  # папка сети над logs

    df = pd.read_csv(log_path, sep="\t")
    if not {"epoch", "val_recall", "val_neg_recall"}.issubset(df.columns):
        print(f"Missing cols in {fname}")
        continue

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

# 4) Фрейм со всеми результатами
results = pd.DataFrame(records)
if results.empty:
    print("No valid logs found. Exiting.")
    exit(1)

results.to_csv(OUTPUT_DIR / "all_results.csv", index=False)

# 5) Лучший per-network и общий лучший
best_per_network = (
    results
    .loc[results.groupby("network")["best_balanced_acc"].idxmax()]
    .reset_index(drop=True)
)
best_per_network.to_csv(OUTPUT_DIR / "best_per_network.csv", index=False)

overall_best = results.loc[results["best_balanced_acc"].idxmax()]
overall_best.to_frame().T.to_csv( # type: ignore
    OUTPUT_DIR / "overall_best.csv", index=False
)

# 6) Стиль графиков
sns.set(style="whitegrid", font_scale=1.1)

# 7) Основные графики (как было ранее)
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

# 8) Корреляционная матрица гиперпараметров и метрики
corr_matrix = results[["h1", "h2", "lr", "bs", "best_balanced_acc"]].corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "param_corr_matrix.png")
plt.close()

# 9) Scatter с регрессией для каждого гиперпараметра по режимам BCE
for param in ["h1", "h2", "lr", "bs"]:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        data=results,
        x=param, y="best_balanced_acc",
        hue="mode", style="mode", s=50
    )
    # отдельные линейные тренды по каждому mode
    for m, color in zip(results["mode"].unique(), ["C0", "C1"]):
        subset = results[results["mode"] == m]
        sns.regplot(
            data=subset,
            x=param, y="best_balanced_acc",
            scatter=False, color=color, label=f"{m} trend"
        )
    plt.title(f"{param} vs Balanced Accuracy by Loss Mode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{param}_vs_balanced_acc_by_mode.png")
    plt.close()

print("Analysis complete!")
print(f"- All results:      {OUTPUT_DIR/'all_results.csv'}")
print(f"- Best per network: {OUTPUT_DIR/'best_per_network.csv'}")
print(f"- Overall best:     {OUTPUT_DIR/'overall_best.csv'}")
print(f"- Plots in:         {PLOTS_DIR}")
