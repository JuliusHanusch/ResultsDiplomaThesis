import sqlite3
import json
import numpy as np
import pandas as pd
import os

from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================

DB_PATH = "/mnt/c/Users/juliu/Diplomarbeit/HPO_Analyse/AION_Standard.db"
TABLE_NAME = "Results"

OUTPUT_DIR = "/mnt/c/Users/juliu/Diplomarbeit/HPO_Analyse/Correlation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

optim_map = {
    "adamw_torch_fused": 0,
    "adafactor": 1
}

lr_sched_map = {
    "linear": 0,
    "cosine": 1
}

# =========================================================
# LOAD DATA
# =========================================================

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute(f"""
SELECT id, config, Utility, budget
FROM {TABLE_NAME}
WHERE Utility IS NOT NULL
AND BUDGET = 5120000
""")

rows = cursor.fetchall()

print(f"Loaded {len(rows)} runs")

# =========================================================
# BUILD DATAFRAME
# =========================================================

data = []
utility = []

for run_id, config_json, util, budget in rows:

    cfg = json.loads(config_json)

    try:
        data.append({
            "learning_rate": float(cfg["learning_rate"]),
            "warmup_ratio": float(cfg["warmup_ratio"]),
            "optim": float(optim_map[cfg["optim"]]),
            "batch_size_expo": float(cfg["batch_size_expo"]),
            "max_missing_prop": float(cfg["max_missing_prop"]),
            "drop_prob": float(cfg["drop_prob"]),
            "lr_scheduler_type": float(lr_sched_map[cfg["lr_scheduler_type"]]),
            "min_past_expo": float(cfg["min_past_expo"]),
            "mean_span_length": float(cfg["mean_span_length"]),
            "masking_prob": float(cfg["masking_prob"]),
        })

        # convert utility -> performance (higher = better)
        utility.append(-float(util))

    except KeyError as e:
        print(f"Skipping run {run_id}, missing {e}")

df = pd.DataFrame(data)
y = np.array(utility, dtype=np.float64)

print("Data shape:", df.shape)
print("Target shape:", y.shape)

assert len(df) == len(y)

# =========================================================
# RANK-BASED PERFORMANCE (KEY FIX FOR THESIS)
# =========================================================

performance = pd.Series(y).rank(method="average").values

# =========================================================
# CORRELATION ANALYSIS
# =========================================================

results = []

print("\n=== Correlation Analysis (Rank-based performance) ===\n")

for col in df.columns:

    x = df[col].values

    s_corr, s_p = spearmanr(x, performance)
    p_corr, p_p = pearsonr(x, performance)

    results.append({
        "hyperparameter": col,
        "spearman_corr": s_corr,
        "pearson_corr": p_corr,
        "spearman_p": s_p,
        "pearson_p": p_p,
        "abs_spearman": abs(s_corr)
    })

    print(
        f"{col:20s} | "
        f"Spearman: {s_corr: .4f} | "
        f"Pearson: {p_corr: .4f}"
    )

results_df = pd.DataFrame(results)

# rank hyperparameters
results_df = results_df.sort_values("abs_spearman", ascending=False)

print("\n=== RANKED IMPORTANCE ===\n")
print(results_df[["hyperparameter", "spearman_corr", "abs_spearman"]])

# save table
results_df.to_csv(os.path.join(OUTPUT_DIR, "ranking.csv"), index=False)

# =========================================================
# PLOTS DIRECTORY
# =========================================================

PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# =========================================================
# 1. GLOBAL RANKING PLOT
# =========================================================

plt.figure(figsize=(10, 5))

sorted_df = results_df.sort_values("spearman_corr", ascending=True)

plt.barh(
    sorted_df["hyperparameter"],
    sorted_df["spearman_corr"]
)

plt.axvline(0, color="black", linewidth=1)

plt.xlabel("Spearman correlation")
plt.title("Hyperparameter Importance Ranking (Rank-based)")

plt.tight_layout()

plt.savefig(os.path.join(PLOT_DIR, "01_ranking.png"), dpi=300)
plt.close()

print("Saved ranking plot")

# =========================================================
# 2. MAIN EFFECT: mean_span_length
# =========================================================

hp = "mean_span_length"

plt.figure(figsize=(6,4))
plt.scatter(df[hp], performance)

z = np.polyfit(df[hp], performance, 1)
p = np.poly1d(z)

x_sorted = np.sort(df[hp])
plt.plot(x_sorted, p(x_sorted), linestyle="--")

plt.xlabel(hp)
plt.ylabel("Performance Rank")
plt.title(f"{hp} vs Performance Rank")

plt.tight_layout()

plt.savefig(os.path.join(PLOT_DIR, "02_mean_span_length.png"), dpi=300)
plt.close()

print("Saved mean_span_length plot")

# =========================================================
# 3. TOP 3 HYPERPARAMETERS
# =========================================================

top_hps = ["mean_span_length", "learning_rate", "warmup_ratio"]

for hp in top_hps:

    plt.figure(figsize=(6,4))
    plt.scatter(df[hp], performance)

    z = np.polyfit(df[hp], performance, 1)
    p = np.poly1d(z)

    x_sorted = np.sort(df[hp])
    plt.plot(x_sorted, p(x_sorted), linestyle="--")

    plt.xlabel(hp)
    plt.ylabel("Performance Rank")
    plt.title(f"{hp} vs Performance Rank")

    plt.tight_layout()

    plt.savefig(os.path.join(PLOT_DIR, f"03_{hp}.png"), dpi=300)
    plt.close()

    print(f"Saved {hp} plot")

print("\nAll plots saved to:", PLOT_DIR)