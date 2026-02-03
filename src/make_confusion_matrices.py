import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def main():
    label_names = ["benign", "human_phish", "llm_phish"]

    # 1) Baseline transformer – full test set
    base = pd.read_csv("results/test_predictions_transformer_only.csv")
    cm_base = confusion_matrix(base["true_label"], base["pred_label"], labels=label_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_base, display_labels=label_names)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix – Transformer Only")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix_baseline.png")
    plt.close()

    # 2) Hybrid – full test set
    hyb = pd.read_csv("results/test_predictions_hybrid.csv")
    cm_hyb = confusion_matrix(hyb["true_label"], hyb["pred_label"], labels=label_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_hyb, display_labels=label_names)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix – Hybrid Transformer + Stylometry")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix_hybrid_full.png")
    plt.close()

    # 3) Hybrid – LLM‑phish subset only
    subset_llm = hyb[hyb["true_label"] == "llm_phish"]
    cm_hyb_llm = confusion_matrix(subset_llm["true_label"], subset_llm["pred_label"], labels=["llm_phish"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_hyb_llm, display_labels=["llm_phish"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix – Hybrid on LLM‑Phish Subset")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix_hybrid_llm_only.png")
    plt.close()

    print("Saved 3 confusion matrices into results/")

if __name__ == "__main__":
    main()
