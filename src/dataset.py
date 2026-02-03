import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .config import DATA_PATH, TEXT_COLUMN, LABEL_COLUMN, LABEL2ID, MODEL_NAME, MAX_LENGTH
from .config import DATA_PATH, TEXT_COLUMN, LABEL_COLUMN, LABEL2ID, MODEL_NAME, MAX_LENGTH

STYLO_PATH = "data/stylometry_features.npy"


class EmailDataset(Dataset):
    def __init__(self, csv_path: str = DATA_PATH, split: str = "train", train_frac: float = 0.7, val_frac: float = 0.15):
        # Load full dataset once
        full_df = pd.read_csv(csv_path)

        # Keep only rows with valid labels
        full_df = full_df[full_df[LABEL_COLUMN].isin(LABEL2ID.keys())].reset_index(drop=True)

        # Load stylometric features for all rows (same order as full_df)
        stylo_all = np.load(STYLO_PATH)
        if len(stylo_all) != len(full_df):
            raise ValueError(f"Stylometry feature count ({len(stylo_all)}) does not match dataset row count ({len(full_df)})")

        # Shuffle once for reproducible splits
        full_df = full_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        stylo_all = stylo_all[full_df.index.values]

        n_total = len(full_df)
        n_train = int(train_frac * n_total)
        n_val = int(val_frac * n_total)

        if split == "train":
            self.df = full_df.iloc[:n_train].reset_index(drop=True)
            self.stylo = stylo_all[:n_train]
        elif split == "val":
            self.df = full_df.iloc[n_train:n_train + n_val].reset_index(drop=True)
            self.stylo = stylo_all[n_train:n_train + n_val]
        elif split == "test":
            self.df = full_df.iloc[n_train + n_val:].reset_index(drop=True)
            self.stylo = stylo_all[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row[TEXT_COLUMN])
        label_str = row[LABEL_COLUMN]
        label_id = LABEL2ID[label_str]

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        stylo_vec = self.stylo[idx]

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "stylometry": torch.tensor(stylo_vec, dtype=torch.float),
            "labels": torch.tensor(label_id, dtype=torch.long),
        }
        return item
