import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .config import ID2LABEL


def main():
    # Load predictions
    df = pd.read_csv("results/test_predictions_hybrid.csv")

    y_true = df["true_label_id"].tolist()
    y_pred = df["pred_label_id"].tolist()

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    labels = [ID2LABEL[i] for i in sorted(ID2LABEL.keys())]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix – Hybrid Transformer + Stylometry")

    plt.tight_layout()
    out_path = "results/confusion_matrix_hybrid.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix figure to {out_path}")


if __name__ == "__main__":
    main()
