import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .stylometry import extract_features_for_text  # adjust name if different


def main():
    # 1. Load a manageable sample of the dataset
    df = pd.read_csv("data/final_dataset.csv")
    # Use a subset if the file is huge
    df = df.sample(min(len(df), 1000), random_state=0)

    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()

    # 2. Extract stylometric features for each email
    feats_list = []
    for t in texts:
        feats_list.append(extract_features_for_text(t))
    feat_df = pd.DataFrame(feats_list)

    # 3. Compute, for each feature, how much it varies between classes
    feat_df["label"] = labels
    means_by_label = feat_df.groupby("label").mean()          # shape: n_labels x n_features
    # variance of class means across labels as a simple importance proxy
    var_across_labels = means_by_label.var(axis=0)            # length n_features

    # 4. Select top 10 features
    top10 = var_across_labels.sort_values(ascending=False).head(10)
    top_feats = top10.index.tolist()
    top_vals = top10.values

    # 5. Plot horizontal bar chart
    plt.figure(figsize=(6, 4))
    y_pos = np.arange(len(top_feats))
    plt.barh(y_pos, top_vals, color="steelblue")
    plt.yticks(y_pos, top_feats)
    plt.xlabel("Variance of class means")
    plt.title("Top 10 Stylometric Features (by class separation)")
    plt.gca().invert_yaxis()  # largest on top
    plt.tight_layout()
    plt.savefig("results/top10_stylometric_features.png")
    plt.close()

    print("Saved results/top10_stylometric_features.png")

if __name__ == "__main__":
    main()
