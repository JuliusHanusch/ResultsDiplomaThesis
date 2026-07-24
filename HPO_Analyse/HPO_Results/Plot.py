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
]

# ---- LOAD ----
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query(f"SELECT id, config, Utility, budget FROM {TABLE_NAME}", conn)
conn.close()

# ---- PARSE SAFE ----
def extract_config(row):
    config = json.loads(row["config"])

    extracted = {key: config.get(key, None) for key in HPARAMS}
    extracted["Utility"] = row["Utility"]
    extracted["budget"] = row["budget"]
    extracted["id"] = row["id"]

    return extracted

parsed_rows = df.apply(extract_config, axis=1)
df_parsed = pd.DataFrame(parsed_rows.tolist())
df_parsed["label"] = df_parsed["id"].astype(str)
df_parsed["id_str"] = df_parsed["id"].astype(str)
df_parsed = df_parsed.fillna("MISSING")

print("Rows after parsing (before filtering NaNs):", len(df_parsed))

# only filter budget first (IMPORTANT)
df_parsed = df_parsed[df_parsed["budget"] == 5120000].reset_index(drop=True)

print("Rows after budget filter:", len(df_parsed))

print(df_parsed["id"].tolist())



# ---- FILTER UTILITY ----
df_parsed = df_parsed[df_parsed["Utility"] <= 60].reset_index(drop=True)

# ---- CREATE RANK ----
# highest Utility = Rank 1
df_parsed = df_parsed.sort_values("Utility", ascending=True).reset_index(drop=True)
df_parsed["Rank"] = range(1, len(df_parsed) + 1)

print(df_parsed.isin(["MISSING"]).sum())

# ensure numeric columns are numeric
for col in HPARAMS:
    if col not in ["optim", "lr_scheduler_type"]:
        df_parsed[col] = pd.to_numeric(df_parsed[col], errors="coerce")

categorical_cols = ["optim", "lr_scheduler_type"]

category_mappings = {}
reverse_mappings = {}

for col in categorical_cols:
    unique_vals = sorted(df_parsed[col].dropna().unique())

    mapping = {val: i for i, val in enumerate(unique_vals)}
    reverse_mapping = {i: val for val, i in mapping.items()}

    category_mappings[col] = mapping
    reverse_mappings[col] = reverse_mapping

    print(f"\n{col} mapping:")
    print(mapping)

    df_parsed[col] = df_parsed[col].map(mapping)

print("\n=== CATEGORY INTERPRETATION (IMPORTANT FOR PLOTTING) ===")
for col in categorical_cols:
    print(f"\n{col}:")
    for k, v in category_mappings[col].items():
        print(f"  {k} → {v}")

# include rank as final axis
print(df_parsed[HPARAMS + ["Rank", "id"]])
print(df_parsed.isna().sum())

valid_rows = df_parsed.dropna(subset=HPARAMS + ["Rank"])
print("Valid rows for plotting:", len(valid_rows))
print(valid_rows["id"].tolist())

plot_df = df_parsed.copy()

print("Duplicate rows (exact match):")
print(plot_df.duplicated(subset=col, keep=False))

plot_dimensions = HPARAMS

# ---- PLOT ----
fig = px.parallel_coordinates(
    df_parsed,
    color="Rank",
    dimensions=plot_dimensions,
    color_continuous_scale=px.colors.sequential.Viridis_r
)


out_html = os.path.join(output_dir, "parallel_coordinates.html")
pio.write_html(fig, out_html, auto_open=False)

print("Saved:", out_html)