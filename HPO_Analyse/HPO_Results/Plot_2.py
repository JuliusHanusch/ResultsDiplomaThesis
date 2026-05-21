import sqlite3
import json
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import os


# ---- CONFIG ----
DB_PATH = "/mnt/c/Users/juliu/Diplomarbeit/HPO_Analyse/AION_Standard.db"
TABLE_NAME = "Results"

output_dir = "/mnt/c/Users/juliu/Diplomarbeit/HPO_Analyse/HPO_Results"
output_file = os.path.join(output_dir, "parallel_coordinates_highFidelity.png")

os.makedirs(output_dir, exist_ok=True)

# ---- HYPERPARAMETERS ----
HPARAMS = [
    "learning_rate",
    "warmup_ratio",
    "optim",
    "batch_size",
    "max_missing_prop",
    "drop_prob",
    "lr_scheduler_type",
    "min_past",
    "mean_span_length",
    "masking_prob",
    "training_steps",
    "gradient_accumulation_steps"
]

# ---- LOAD DATA ----
conn = sqlite3.connect(DB_PATH)
query = f"SELECT id, config, Utility, budget FROM {TABLE_NAME}"
df = pd.read_sql_query(query, conn)
conn.close()

# ---- PARSE CONFIG ----
def extract_config(row):
    try:
        config = json.loads(row["config"])
    except json.JSONDecodeError:
        return None

    extracted = {}
    for key in HPARAMS:
        extracted[key] = config.get(key, None)

    extracted["Utility"] = row["Utility"]
    extracted["budget"] = row["budget"]

    return extracted


parsed_rows = df.apply(extract_config, axis=1)
parsed_rows = parsed_rows.dropna()
parsed_rows = parsed_rows[parsed_rows.apply(lambda x: x["budget"] == 5120000)]

df_parsed = pd.DataFrame(parsed_rows.tolist())

# ---- HANDLE CATEGORICAL VARIABLES ----
categorical_cols = ["optim", "lr_scheduler_type"]

for col in categorical_cols:
    df_parsed[col] = df_parsed[col].astype("category").cat.codes

# ---- NORMALIZE NUMERIC COLUMNS ----
for col in df_parsed.columns:
    if col != "Utility":
        min_val = df_parsed[col].min()
        max_val = df_parsed[col].max()

        if max_val > min_val:
            df_parsed[col] = (df_parsed[col] - min_val) / (max_val - min_val)
        else:
            df_parsed[col] = 0.5

# ---- BIN UTILITY ----
df_parsed["Utility_bin"] = pd.qcut(
    df_parsed["Utility"],
    q=4,
    labels=["low", "mid-low", "mid-high", "high"]
)

# ---- PLOT ----
plt.figure(figsize=(15, 7))

parallel_coordinates(
    df_parsed.drop(columns=["Utility"]),
    class_column="Utility_bin",
    colormap=plt.cm.viridis,
    alpha=0.4
)

plt.xticks(rotation=45)
plt.title("Parallel Coordinates Plot of HPO Results")
plt.tight_layout()

plt.savefig(output_file, dpi=300)
plt.close()

print(f"Saved plot to: {output_file}")