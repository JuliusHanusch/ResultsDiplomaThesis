import sqlite3
import pandas as pd
import plotly.express as px
import plotly.io as pio
import json
import os

DB_PATH = "/mnt/c/Users/juliu/Diplomarbeit/HPO_Analyse/AION_Standard.db"
TABLE_NAME = "Results"

output_dir = "/mnt/c/Users/juliu/Diplomarbeit/HPO_Analyse/HPO_Results"
os.makedirs(output_dir, exist_ok=True)

HPARAMS = [
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
    "training_steps",
    "gradient_accumulation_steps"
]

# ---- LOAD ----
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query(f"SELECT config, Utility, budget FROM {TABLE_NAME}", conn)
conn.close()

# ---- PARSE SAFE ----
def extract_config(row):
    config = json.loads(row["config"])

    extracted = {key: config.get(key, None) for key in HPARAMS}
    extracted["Utility"] = row["Utility"]
    extracted["budget"] = row["budget"]


    return extracted

parsed_rows = df.apply(extract_config, axis=1)
parsed_rows = parsed_rows.dropna()
parsed_rows = parsed_rows[parsed_rows.apply(lambda x: x["budget"] == 5120000)]
df_parsed = pd.DataFrame(parsed_rows.tolist())

print("Rows (before filter):", len(df_parsed))

# ---- FILTER UTILITY ----
df_parsed = df_parsed[df_parsed["Utility"] <= 50].reset_index(drop=True)

# ---- CREATE RANK ----
# highest Utility = Rank 1
df_parsed = df_parsed.sort_values("Utility", ascending=False).reset_index(drop=True)
df_parsed["Rank"] = range(1, len(df_parsed) + 1)

# ensure numeric columns are numeric
for col in HPARAMS:
    if col not in ["optim", "lr_scheduler_type"]:
        df_parsed[col] = pd.to_numeric(df_parsed[col], errors="coerce")

categorical_cols = ["optim", "lr_scheduler_type"]

category_mappings = {}

for col in categorical_cols:
    unique_vals = sorted(df_parsed[col].dropna().unique())
    mapping = {val: i for i, val in enumerate(unique_vals)}
    category_mappings[col] = mapping

    print(f"{col} mapping:", mapping)

    df_parsed[col] = df_parsed[col].map(mapping)

# include rank as final axis
plot_dimensions = HPARAMS + ["Rank"]

# ---- PLOT ----
fig = px.parallel_coordinates(
    df_parsed,
    color="Rank",
    dimensions=plot_dimensions,
    color_continuous_scale=px.colors.sequential.Viridis_r  # Rank 1 appears distinct
)

out_html = os.path.join(output_dir, "parallel_coordinates.html")
pio.write_html(fig, out_html, auto_open=False)

print("Saved:", out_html)