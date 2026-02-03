import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("results/hybrid_training_history.csv")

    # Loss vs epoch
    plt.figure(figsize=(6, 4))
    plt.plot(df["epoch"], df["train_loss"], marker="o", label="Train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Hybrid Model – Training Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_loss = "results/hybrid_training_loss.png"
    plt.savefig(out_loss, dpi=300)
    plt.close()

    # Macro F1 vs epoch
    plt.figure(figsize=(6, 4))
    plt.plot(df["epoch"], df["val_macro_f1"], marker="o", label="Val macro F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.ylim(0.0, 1.05)
    plt.title("Hybrid Model – Validation Macro F1")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_f1 = "results/hybrid_training_macro_f1.png"
    plt.savefig(out_f1, dpi=300)
    plt.close()

    print(f"Saved training curves to {out_loss} and {out_f1}")


if __name__ == "__main__":
    main()
