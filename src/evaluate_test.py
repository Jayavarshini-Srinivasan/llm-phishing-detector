import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .config import BATCH_SIZE, ID2LABEL
from .dataset import EmailDataset
from .model_hybrid import HybridTransformerStylometric


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Test dataset
    test_dataset = EmailDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    model = HybridTransformerStylometric()
    state_dict = torch.load("models/hybrid_transformer_stylometric.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            stylometry = batch["stylometry"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                stylometric_features=stylometry,
            )
            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # Classification report
    print("Test classification report:")
    print(classification_report(all_labels, all_preds, digits=4))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion matrix (rows = true, cols = pred):")
    print(cm)

    # Save predictions for plotting later
    label_names = [ID2LABEL[i] for i in all_labels]
    pred_names = [ID2LABEL[i] for i in all_preds]

    df_out = pd.DataFrame({
        "true_label_id": all_labels,
        "true_label": label_names,
        "pred_label_id": all_preds,
        "pred_label": pred_names,
    })
    df_out.to_csv("results/test_predictions_hybrid.csv", index=False)
    print("Saved test predictions to results/test_predictions_hybrid.csv")


if __name__ == "__main__":
    main()
