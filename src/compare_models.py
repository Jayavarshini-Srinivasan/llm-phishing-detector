import pandas as pd
from sklearn.metrics import classification_report

base = pd.read_csv("results/test_predictions_transformer_only.csv")
hybr = pd.read_csv("results/test_predictions_hybrid.csv")

print("Baseline:")
print(classification_report(base["true_label"], base["pred_label"], digits=4))

print("\nHybrid:")
print(classification_report(hybr["true_label"], hybr["pred_label"], digits=4))
