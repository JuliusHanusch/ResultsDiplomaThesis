import pandas as pd
import matplotlib.pyplot as plt

# Path to your CSV file
csv_path = "/mnt/c/Users/juliu/Diplomarbeit/HPO_Analyse/Comparison/Classification_20Epochs.csv"

# Read CSV
df = pd.read_csv(csv_path)

# Get the Average row
avg = df[df["dataset"] == "Average"].iloc[0]

# Extract values
labels = ["Default", "Optimized", "Random"]
accuracies = [
    avg["Default_accuracy"],
    avg["Optimized_accuracy"],
    avg["Random_accuracy"],
]

# Create plot
plt.figure(figsize=(6, 4))
bars = plt.bar(labels, accuracies)

# Add value labels
for bar, value in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.01,
        f"{value:.3f}",
        ha="center",
        fontsize=11,
    )

plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.title("Classification Accuracy over 5 Datasets")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("average_accuracy.png", dpi=300)
plt.show()

