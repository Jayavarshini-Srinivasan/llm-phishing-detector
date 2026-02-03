import sys
import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .config import TEXT_COLUMN, LABEL_COLUMN, LABEL2ID, ID2LABEL, BATCH_SIZE
from .model_hybrid import HybridTransformerStylometric
from .stylometry import extract_features_for_text
from transformers import AutoTokenizer


def build_custom_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df[LABEL_COLUMN].isin(LABEL2ID.keys())].reset_index(drop=True)

    texts = df[TEXT_COLUMN].astype(str).tolist()
    labels = [LABEL2ID[l] for l in df[LABEL_COLUMN].tolist()]

    # Build stylometry features on the fly
    stylo = [extract_features_for_text(t) for t in texts]
    stylo = torch.tensor(stylo, dtype=torch.float)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )

    dataset = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "stylometry": stylo,
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    return df, dataset


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.evaluate_custom_csv <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    print("Evaluating on:", csv_path)

    df, data = build_custom_dataset(csv_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = HybridTransformerStylometric()
    state_dict = torch.load("models/hybrid_transformer_stylometric.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    dataset_size = data["input_ids"].size(0)
    idxs = torch.arange(dataset_size)
    loader = DataLoader(idxs, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idxs in loader:
            input_ids = data["input_ids"][batch_idxs].to(device)
            attention_mask = data["attention_mask"][batch_idxs].to(device)
            stylometry = data["stylometry"][batch_idxs].to(device)
            labels = data["labels"][batch_idxs].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                stylometric_features=stylometry,
            )
            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    print("Custom CSV classification report:")
    print(classification_report(all_labels, all_preds, digits=4))
    print("Confusion matrix (rows = true, cols = pred):")
    print(confusion_matrix(all_labels, all_preds))

    # Save predictions
    label_names = [ID2LABEL[i] for i in all_labels]
    pred_names = [ID2LABEL[i] for i in all_preds]

    out_df = pd.DataFrame({
        "text": df[TEXT_COLUMN],
        "true_label": label_names,
        "pred_label": pred_names,
    })
    out_path = "results/custom_eval_predictions.csv"
    out_df.to_csv(out_path, index=False)
    print("Saved custom predictions to", out_path)


if __name__ == "__main__":
    main()
