import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
import pandas as pd

from .config import (
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE_TRANSFORMER,
    LEARNING_RATE_HEAD,
)
from .dataset import EmailDataset
from .model_hybrid import HybridTransformerStylometric


def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        stylometry = batch["stylometry"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            stylometric_features=stylometry,
            labels=labels,
        )
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
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

    report = classification_report(all_labels, all_preds, output_dict=True, digits=4)
    # Print normal text report for you to see
    print(classification_report(all_labels, all_preds, digits=4))

    metrics = {
        "val_accuracy": report["accuracy"],
        "val_macro_f1": report["macro avg"]["f1-score"],
        "val_weighted_f1": report["weighted avg"]["f1-score"],
    }
    return metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dataset = EmailDataset(split="train")
    val_dataset = EmailDataset(split="val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = HybridTransformerStylometric()
    model.to(device)

    # Separate parameters for transformer vs head
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.transformer.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "lr": LEARNING_RATE_TRANSFORMER,
        },
        {
            "params": [
                p for n, p in model.transformer.named_parameters() if any(nd in n for nd in no_decay)
            ],
            "lr": LEARNING_RATE_TRANSFORMER,
        },
        {
            "params": [p for p in model.stylo_mlp.parameters()] + [p for p in model.classifier.parameters()],
            "lr": LEARNING_RATE_HEAD,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    history = []
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        avg_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        print("Average training loss:", avg_loss)
        print("Validation:")
        val_metrics = evaluate(model, val_loader, device)

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            **val_metrics,
        }
        history.append(epoch_record)
    # Save metrics history
    df_hist = pd.DataFrame(history)
    df_hist.to_csv("results/hybrid_training_history.csv", index=False)
    print("Saved training history to results/hybrid_training_history.csv")

    # Save model
    torch.save(model.state_dict(), "models/hybrid_transformer_stylometric.pt")
    print("Model saved to models/hybrid_transformer_stylometric.pt")


if __name__ == "__main__":
    main()
