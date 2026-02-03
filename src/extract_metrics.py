from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parent.parent
results = ROOT / "results"
data = ROOT / "data"

# 1) Baseline transformer metrics
df_base = pd.read_csv(results / "test_predictions_transformer_only.csv")
y_true_base = df_base["true_label"]
y_pred_base = df_base["pred_label"]

print("=== BASELINE TRANSFORMER ===")
print(classification_report(y_true_base, y_pred_base))

# 2) Hybrid model metrics
df_hybrid = pd.read_csv(results / "test_predictions_hybrid.csv")
y_true_h = df_hybrid["true_label"]
y_pred_h = df_hybrid["pred_label"]

print("\n=== HYBRID MODEL ===")
print(classification_report(y_true_h, y_pred_h))

# 3) Exact stylometric feature count
features = np.load(data / "stylometry_features.npy")
print(f"\nExact stylometric feature count: {features.shape[1]}")

# 4) Robustness sample size
df_robust = pd.read_csv(results / "custom_eval_predictions.csv")
print(f"Robustness test samples: {len(df_robust)}")
