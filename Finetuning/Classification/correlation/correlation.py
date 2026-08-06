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

DB_PATH = (
    "/mnt/c/Users/juliu/Diplomarbeit/"
    "Finetuning/Classification/classification_small.db"
)

TABLE_NAME = "runs"

OUTPUT_DIR = (
    "/mnt/c/Users/juliu/Diplomarbeit/"
    "Finetuning/Classification/correlation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

conn = sqlite3.connect(DB_PATH)

df_raw = pd.read_sql(
    f"SELECT * FROM {TABLE_NAME}",
    conn
)

def get_train_inner_model(config):
    cfg = json.loads(config)
    return int(cfg["TrainInnerModel"])


df_raw["TrainInnerModel"] = df_raw["config"].apply(
    get_train_inner_model
)


df_raw = df_raw[
    df_raw["TrainInnerModel"] == 1
].copy()


print(
    "Runs after TrainInnerModel=True filter:",
    len(df_raw)
)

print(df_raw.head())
print(df_raw.columns)



# =========================================================
# PREPARE DATA
# =========================================================

def prepare_dataset(df):

    data = []
    accuracy = []


    for _, row in df.iterrows():

        cfg = json.loads(row["config"])


        effective_batch = (
            cfg["per_device_train_batch_size"]
            *
            cfg["gradient_accumulation_steps"]
        )


        data.append({

            "num_train_epochs":
                float(cfg["num_train_epochs"]),


            "effective_batch_size":
                float(effective_batch),


            "learning_rate":
                float(cfg["learning_rate"]),


            "dropout_head":
                float(cfg["dropout_head"]),


            "warmup_ratio":
                float(cfg["warmup_ratio"]),


            # "TrainInnerModel":
            #     float(
            #         1 if cfg["TrainInnerModel"]
            #         else 0
            #     )
        })


        # higher is better
        accuracy.append(
            float(row["accuracy"])
        )


    X = pd.DataFrame(data)

    y = np.array(
        accuracy,
        dtype=np.float64
    )


    return X, y



# =========================================================
# DATASET LOOP
# =========================================================

datasets = df_raw["dataset"].unique()


for dataset in datasets:

    print("\n==============================")
    print(dataset)
    print("==============================")


    df = df_raw[
        df_raw["dataset"] == dataset
    ]


    X, y = prepare_dataset(df)


    print(
        "Runs:",
        len(X)
    )


    # =====================================================
    # Rank transform accuracy
    # =====================================================

    performance_rank = (
        pd.Series(y)
        .rank(method="average")
        .values
    )


    results = []


    print(
        "\n=== Spearman Correlation ==="
    )


    for hp in X.columns:


        corr, p = spearmanr(
            X[hp],
            performance_rank
        )


        pearson, pearson_p = pearsonr(
            X[hp],
            y
        )


        results.append({

            "hyperparameter": hp,

            "spearman_corr":
                corr,

            "spearman_p":
                p,

            "pearson_corr":
                pearson,

            "pearson_p":
                pearson_p,

            "abs_spearman":
                abs(corr)
        })


        print(
            f"{hp:25s} "
            f"Spearman={corr: .4f}"
        )


    results_df = pd.DataFrame(results)


    results_df = results_df.sort_values(
        "abs_spearman",
        ascending=False
    )


    print(
        "\n=== Ranking ==="
    )

    print(
        results_df[
            [
                "hyperparameter",
                "spearman_corr",
                "abs_spearman"
            ]
        ]
    )


    dataset_dir = os.path.join(
        OUTPUT_DIR,
        dataset.replace("-", "_")
    )

    os.makedirs(
        dataset_dir,
        exist_ok=True
    )


    results_df.to_csv(
        os.path.join(
            dataset_dir,
            "spearman_ranking_noinnermodel.csv"
        ),
        index=False
    )


    # =====================================================
    # Ranking plot
    # =====================================================

    plt.figure(
        figsize=(8,5)
    )


    plot_df = results_df.sort_values(
        "spearman_corr"
    )


    plt.barh(
        plot_df["hyperparameter"],
        plot_df["spearman_corr"]
    )


    plt.axvline(
        0,
        color="black"
    )


    plt.xlabel(
        "Spearman correlation with accuracy rank"
    )

    plt.title(
        f"{dataset}: Hyperparameter correlation"
    )


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            dataset_dir,
            "ranking_noinnermodel.png"
        ),
        dpi=300
    )


    plt.close()


print("\nFinished.")