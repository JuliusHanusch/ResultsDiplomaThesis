import sqlite3
import json
import numpy as np
import pandas as pd

from fanova import fANOVA
import fanova.visualizer
import os


from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter
)

# =========================================================
# CONFIG
# =========================================================

DB_PATH = "/mnt/c/Users/juliu/Diplomarbeit/HPO_Analyse/AION_Standard.db"
TABLE_NAME = "Results"

HPO_KEYS = [
    "learning_rate",
    "warmup_ratio",
    "optim",
    "batch_size_expo",
    "max_missing_prop",
    "drop_prob",
    "lr_scheduler_type",
    "min_past_expo",
    "mean_span_length",
    "masking_prob",
]

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
# BUILD DATAFRAME (KEY FIX)
# =========================================================

data = []
run_ids = []

for run_id, config_json, utility, budget in rows:

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

        run_ids.append(run_id)

    except KeyError as e:
        print(f"Skipping run {run_id}, missing {e}")

df = pd.DataFrame(data)

Y = np.array([float(r[2]) for r in rows if r[2] is not None], dtype=np.float64)

print("DataFrame shape:", df.shape)
print("Y shape:", Y.shape)

# =========================================================
# CONFIGSPACE
# =========================================================

cs = ConfigurationSpace()

cs.add([
    UniformFloatHyperparameter("learning_rate", 0.00005, 0.01, log=True),
    UniformFloatHyperparameter("warmup_ratio", 1e-7, 0.1, log=True),

    CategoricalHyperparameter("optim", ["0", "1"]),

    UniformFloatHyperparameter("batch_size_expo", 1, 11),

    UniformFloatHyperparameter("max_missing_prop", 0.8, 1.0),
    UniformFloatHyperparameter("drop_prob", 0.0, 0.5),

    CategoricalHyperparameter("lr_scheduler_type", ["0", "1"]),

    UniformIntegerHyperparameter("min_past_expo", 4, 10),
    UniformIntegerHyperparameter("mean_span_length", 1, 64),

    UniformFloatHyperparameter("masking_prob", 0.1, 0.3),
])

OUTPUT_DIR = "/mnt/c/Users/juliu/Diplomarbeit/HPO_Analyse/fanova/fanova_plots_30Runs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


N_RUNS = 30

all_importances = []

for run in range(N_RUNS):

    print(f"Running fANOVA {run+1}/{N_RUNS}")

    f = fANOVA(
        X=df,
        Y=Y,
        config_space=cs,
    )

    row = {}

    for hp in df.columns:
        res = f.quantify_importance((hp,))
        row[hp] = res[(hp,)]['individual importance']

    all_importances.append(row)

importance_df = pd.DataFrame(all_importances)

summary = pd.DataFrame({
    "mean_importance": importance_df.mean(),
    "std_importance": importance_df.std(),
    "min_importance": importance_df.min(),
    "max_importance": importance_df.max(),
})

summary = summary.sort_values(
    "mean_importance",
    ascending=False
)

print(summary)

summary.to_csv(
    os.path.join(OUTPUT_DIR, "importance_stability.csv")
)

# =========================================================
# PAIRWISE INTERACTIONS
# =========================================================

from itertools import combinations

pair_results = []

for run in range(N_RUNS):

    f = fANOVA(
        X=df,
        Y=Y,
        config_space=cs,
    )

    row = {}

    for hp1, hp2 in combinations(df.columns, 2):

        res = f.quantify_importance((hp1, hp2))

        interaction = res[(hp1, hp2)]["individual importance"]

        row[f"{hp1}__{hp2}"] = interaction

    pair_results.append(row)

pair_df = pd.DataFrame(pair_results)


summary = pd.DataFrame({
    "mean_importance": pair_df.mean(),
    "std_importance": pair_df.std(),
    "min_importance": pair_df.min(),
    "max_importance": pair_df.max()
})

summary["cv"] = (
    summary["std_importance"] /
    summary["mean_importance"].replace(0, np.nan)
)

summary = summary.sort_values(
    "mean_importance",
    ascending=False
)

summary.to_csv(
    os.path.join(OUTPUT_DIR, "pairwise_interaction_summary.csv")
)

print(summary.head(20))