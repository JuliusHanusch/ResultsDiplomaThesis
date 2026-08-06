import pandas as pd
import matplotlib.pyplot as plt
import os


BASE_DIR = (
    r"/mnt/c/Users/juliu/Diplomarbeit/Finetuning/Classification/Baselines"
)

BASELINE_PATH = os.path.join(
    BASE_DIR,
    "Baselines.csv"
)

MOMENT_PATH = os.path.join(
    BASE_DIR,
    "moment_base.csv"
)

CHRONOS_PATH = os.path.join(
    BASE_DIR,
    "best_results.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "accuracy_plots"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================================================
# LOAD DATA
# =========================================================

baseline_df = pd.read_csv(BASELINE_PATH)
moment_df = pd.read_csv(MOMENT_PATH)
chronos_df = pd.read_csv(CHRONOS_PATH)



datasets = chronos_df["dataset"].unique()


for dataset in datasets:

    print("Plotting:", dataset)


    values = {}
    

    chronos_acc = chronos_df.loc[
        chronos_df["dataset"] == dataset,
        "accuracy"
    ].values[0]

    values["Chronos-BERT-Small"] = chronos_acc


    # -------------------------
    # MOMENT
    # -------------------------

    moment_acc = moment_df.loc[
        moment_df["dataset"] == dataset,
        "test_accuracy"
    ].values[0]

    values["MOMENT"] = moment_acc


    # -------------------------
    # Classical baselines
    # -------------------------

    base = baseline_df[
        baseline_df["dataset"] == dataset
    ].iloc[0]


    values.update({

        "Naive":
            base["Naive_accuracy"],

        "1-NN":
            base["1NN_accuracy"],

        "5-NN":
            base["5NN_accuracy"],

        "Random Forest":
            base["RF_small_accuracy"],

        # "RF large":
        #     base["RF_large_accuracy"]
    })



        # =====================================================
    # SORT + COLOR
    # =====================================================

    sorted_values = dict(
        sorted(
            values.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )

    methods = list(sorted_values.keys())
    accuracies = list(sorted_values.values())


    # best method for this dataset
    best_method = methods[0]


    colors = []

    for method in methods:
        if method == "Chronos-BERT":
            colors.append("tab:blue")      # your model
        elif method == best_method:
            colors.append("tab:green")     # best result
        else:
            colors.append("lightgray")


    # =====================================================
    # PLOT
    # =====================================================

        # =====================================================
    # SORT + COLOR
    # =====================================================

    sorted_values = dict(
        sorted(
            values.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )

    methods = list(sorted_values.keys())
    accuracies = list(sorted_values.values())


    best_method = methods[0]


    colors = []
    labels = []

    for method in methods:

        if method == "Chronos-BERT":
            colors.append("#5B9BD5")   # subtle blue
            labels.append("Chronos-BERT (ours)")

        elif method == best_method:
            colors.append("#70AD47")   # subtle green
            labels.append("Best accuracy")

        else:
            colors.append("#BFBFBF")   # subtle gray
            labels.append("Baseline")


    # =====================================================
    # PLOT
    # =====================================================

    plt.figure(
        figsize=(10,5)
    )


    bars = plt.bar(
        methods,
        accuracies,
        color=colors
    )


    plt.ylim(
        0,
        1.05
    )


    plt.ylabel(
        "Accuracy"
    )


    plt.title(
        f"{dataset} - Classification Accuracy"
    )


    plt.xticks(
        rotation=45,
        ha="right"
    )


    # values above bars
    for bar, acc in zip(
        bars,
        accuracies
    ):

        plt.text(
            bar.get_x() + bar.get_width()/2,
            acc + 0.01,
            f"{acc:.3f}",
            ha="center",
            fontsize=9
        )


        # =====================================================
    # SORT BY ACCURACY
    # =====================================================

    sorted_values = dict(
        sorted(
            values.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )

    methods = list(sorted_values.keys())
    accuracies = list(sorted_values.values())


    # =====================================================
    # PLOT
    # =====================================================

    plt.figure(
        figsize=(10,5)
    )


    bars = plt.bar(
        methods,
        accuracies
    )


    plt.ylim(
        0,
        1.05
    )


    plt.ylabel(
        "Accuracy"
    )


    plt.title(
        f"{dataset} - Classification Accuracy"
    )


    plt.xticks(
        rotation=45,
        ha="right"
    )


    # add values above bars
    for bar, acc in zip(
        bars,
        accuracies
    ):

        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f"{acc:.3f}",
            ha="center",
            fontsize=9
        )


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{dataset}_accuracy.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


print("Finished.")