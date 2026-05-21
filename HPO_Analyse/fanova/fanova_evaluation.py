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
    "batch_size",
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
            "batch_size": float(cfg["batch_size"]),
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

    UniformFloatHyperparameter("batch_size", 2, 2048),

    UniformFloatHyperparameter("max_missing_prop", 0.8, 1.0),
    UniformFloatHyperparameter("drop_prob", 0.0, 0.5),

    CategoricalHyperparameter("lr_scheduler_type", ["0", "1"]),

    UniformIntegerHyperparameter("min_past_expo", 4, 10),
    UniformIntegerHyperparameter("mean_span_length", 1, 64),

    UniformFloatHyperparameter("masking_prob", 0.1, 0.3),
])

# =========================================================
# fANOVA (PANDAS MODE)
# =========================================================

f = fANOVA(
    X=df,
    Y=Y,
    config_space=cs
)

# =========================================================
# IMPORTANCE
# =========================================================


OUTPUT_DIR = "/mnt/c/Users/juliu/Diplomarbeit/HPO_Analyse/fanova/fanova_plots_batchsize_maxbudget"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# SINGLE HP IMPORTANCES
# =========================================================

print("\n=== Single Hyperparameter Importances ===\n")

importance_rows = []

for hp in df.columns:

    res = f.quantify_importance((hp,))
    importance = res[(hp,)]['individual importance']

    print(f"{hp:25s}: {importance:.6f}")

    importance_rows.append({
        "hyperparameter": hp,
        "importance": importance
    })

importance_df = pd.DataFrame(importance_rows)

importance_df.to_csv(
    os.path.join(OUTPUT_DIR, "single_hp_importances.csv"),
    index=False
)

# =========================================================
# PAIRWISE INTERACTIONS
# =========================================================

print("\n=== Pairwise Interactions ===\n")

pairs = f.get_most_important_pairwise_marginals(n=10)

pair_rows = []

for p in pairs:

    print(p)

    pair_rows.append({
        "hp1": p[0],
        "hp2": p[1]
    })

pairs_df = pd.DataFrame(pair_rows)

pairs_df.to_csv(
    os.path.join(OUTPUT_DIR, "pairwise_interactions.csv"),
    index=False
)

print("\nSaved CSV files to:", OUTPUT_DIR)

# =========================================================
# VISUALIZATION
# =========================================================



vis = fanova.visualizer.Visualizer(
    f,
    cs,
    OUTPUT_DIR
)

vis.create_all_plots(OUTPUT_DIR)