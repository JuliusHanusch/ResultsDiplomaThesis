import sqlite3
import json
import numpy as np
import pandas as pd
import os

from fanova import fANOVA

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter
)

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
    "Finetuning/Classification/fanova/fanova_results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


N_RUNS = 10


# =========================================================
# LOAD DATABASE
# =========================================================

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()


cursor.execute(f"""
SELECT *
FROM {TABLE_NAME}
""")

rows = cursor.fetchall()

columns = [
    x[1] for x in cursor.execute(
        f"PRAGMA table_info({TABLE_NAME})"
    ).fetchall()
]


raw_df = pd.DataFrame(rows, columns=columns)

print(raw_df.head())
print(raw_df.columns)


# =========================================================
# CONFIG SPACE
# =========================================================

cs = ConfigurationSpace()


cs.add([
    UniformIntegerHyperparameter(
        "num_train_epochs",
        2,
        40
    ),

    # actual batch size after gradient accumulation
    CategoricalHyperparameter(
        "effective_batch_size",
        [8,16,32,64,128]
    ),

    UniformFloatHyperparameter(
        "learning_rate",
        5e-6,
        1e-3,
        log=True
    ),

    UniformFloatHyperparameter(
        "dropout_head",
        0.0,
        0.3
    ),

    UniformFloatHyperparameter(
        "warmup_ratio",
        0.0,
        0.1
    ),

    CategoricalHyperparameter(
        "TrainInnerModel",
        [0,1]
    )
])


# =========================================================
# PREPARE DATASET FUNCTION
# =========================================================

def prepare_dataset(df):

    X = []
    Y = []

    for _, row in df.iterrows():

        cfg = json.loads(row["config"])

        effective_batch = (
            cfg["per_device_train_batch_size"]
            *
            cfg["gradient_accumulation_steps"]
        )

        X.append({
            "num_train_epochs": int(cfg["num_train_epochs"]),

            "effective_batch_size": int(effective_batch),

            "learning_rate": float(cfg["learning_rate"]),

            "dropout_head": float(cfg["dropout_head"]),

            "warmup_ratio": float(cfg["warmup_ratio"]),

            "TrainInnerModel": int(cfg["TrainInnerModel"])
        })

        Y.append(float(row["accuracy"]))


    X = pd.DataFrame(X)

    X = X[
        list(cs.keys())
    ]

    X = X.astype(float)

    Y = np.array(
        Y,
        dtype=np.float64
    )

    return X, Y



# =========================================================
# RUN FANOVA PER DATASET
# =========================================================

datasets = raw_df["dataset"].unique()


for dataset in datasets:

    print("\n================================")
    print("Dataset:", dataset)
    print("================================")


    dataset_df = raw_df[
        raw_df["dataset"] == dataset
    ]


    X, Y = prepare_dataset(dataset_df)

    X["TrainInnerModel"] = X["TrainInnerModel"].map({
        0: 0,
        1: 1
    })

    X["effective_batch_size"] = X["effective_batch_size"].map({
        8: 0,
        16: 1,
        32: 2,
        64: 3,
        128: 4
    })

    print(X.columns)
    print(cs.keys())
    print(X.iloc[0])

    print(
        "Runs:",
        len(X),
        "X:",
        X.shape,
        "Y:",
        Y.shape
    )


    # -------------------------------
    # Individual importance
    # -------------------------------

    all_importances = []


    for run in range(N_RUNS):

        print(
            f"fANOVA {run+1}/{N_RUNS}"
        )


        f = fANOVA(
            X=X,
            Y=Y,
            config_space=cs
        )


        result = {}


        for hp in X.columns:

            imp = f.quantify_importance(
                (hp,)
            )

            result[hp] = (
                imp[(hp,)]
                ["individual importance"]
            )


        all_importances.append(result)


    importance_df = pd.DataFrame(
        all_importances
    )


    summary = pd.DataFrame({

        "mean_importance":
            importance_df.mean(),

        "std_importance":
            importance_df.std(),

        "min_importance":
            importance_df.min(),

        "max_importance":
            importance_df.max()
    })


    summary = summary.sort_values(
        "mean_importance",
        ascending=False
    )


    print(summary)


    summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{dataset}_importance.csv"
        )
    )


    # =================================================
    # PAIRWISE INTERACTIONS
    # =================================================

    from itertools import combinations


    pair_results = []


    for run in range(N_RUNS):

        f = fANOVA(
            X=X,
            Y=Y,
            config_space=cs
        )


        row = {}


        for hp1, hp2 in combinations(
            X.columns,
            2
        ):

            res = f.quantify_importance(
                (hp1, hp2)
            )


            row[
                f"{hp1}__{hp2}"
            ] = (
                res[(hp1, hp2)]
                ["individual importance"]
            )


        pair_results.append(row)


    pair_df = pd.DataFrame(
        pair_results
    )


    pair_summary = pd.DataFrame({

        "mean_importance":
            pair_df.mean(),

        "std_importance":
            pair_df.std(),

        "min_importance":
            pair_df.min(),

        "max_importance":
            pair_df.max()
    })


    pair_summary["cv"] = (
        pair_summary["std_importance"]
        /
        pair_summary["mean_importance"]
        .replace(0, np.nan)
    )


    pair_summary = pair_summary.sort_values(
        "mean_importance",
        ascending=False
    )


    pair_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{dataset}_pairwise_interactions.csv"
        )
    )


print("\nFinished.")