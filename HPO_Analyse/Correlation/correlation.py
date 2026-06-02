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

        # =====================================================
        # IMPORTANT FIX:
        # convert utility -> performance (higher = better)
        # =====================================================
        utility.append(-float(util))

    except KeyError as e:
        print(f"Skipping run {run_id}, missing {e}")

df = pd.DataFrame(data)
y = np.array(utility, dtype=np.float64)

print("Data shape:", df.shape)
print("Target shape:", y.shape)

assert len(df) == len(y)

# =========================================================
# CORRELATION ANALYSIS (CORRECTED)
# =========================================================

results = []

print("\n=== Corrected Correlation Analysis (higher = better) ===\n")

for col in df.columns:

    x = df[col].values

    s_corr, s_p = spearmanr(x, y)
    p_corr, p_p = pearsonr(x, y)

    results.append({
        "hyperparameter": col,
        "spearman_corr": s_corr,
        "spearman_p": s_p,
        "pearson_corr": p_corr,
        "pearson_p": p_p,
        "abs_spearman": abs(s_corr)
    })

    print(
        f"{col:20s} | "
        f"Spearman: {s_corr: .4f} | "
        f"Pearson: {p_corr: .4f}"
    )

results_df = pd.DataFrame(results)

# =========================================================
# RANKING (MOST IMPORTANT FIRST)
# =========================================================

results_df = results_df.sort_values(
    "abs_spearman",
    ascending=False
)

print("\n=== RANKED IMPORTANCE (corrected) ===\n")
print(results_df[["hyperparameter", "spearman_corr", "abs_spearman"]])

# =========================================================
# SAVE
# =========================================================

out_file = os.path.join(OUTPUT_DIR, "hyperparameter_ranking_corrected.csv")
results_df.to_csv(out_file, index=False)

print("\nSaved to:", out_file)

# =========================================================
# PLOT
# =========================================================

plt.figure(figsize=(10, 5))

plt.bar(
    results_df["hyperparameter"],
    results_df["spearman_corr"]
)

plt.axhline(0, color="black", linewidth=1)

plt.xticks(rotation=45, ha="right")
plt.ylabel("Spearman correlation (higher = better performance)")
plt.title("Hyperparameter Importance (Corrected Objective Direction)")

plt.tight_layout()

plot_path = os.path.join(OUTPUT_DIR, "correlation_plot_corrected.png")
plt.savefig(plot_path, dpi=200)

print("Plot saved to:", plot_path)