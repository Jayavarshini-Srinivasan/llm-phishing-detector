import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

from .config import DATA_PATH, TEXT_COLUMN, LABEL_COLUMN, LABEL2ID, ID2LABEL


def run_tfidf_xgboost_baseline():
    # Load data
    df = pd.read_csv(DATA_PATH)

    # Keep only valid labels
    df = df[df[LABEL_COLUMN].isin(LABEL2ID.keys())].reset_index(drop=True)

    texts = df[TEXT_COLUMN].astype(str).tolist()
    labels = df[LABEL_COLUMN].map(LABEL2ID).tolist()

    # Simple stratified split: 70/15/15 like before
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.30, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        lowercase=True,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    # XGBoost classifier
    clf = XGBClassifier(
        objective="multi:softmax",
        num_class=len(LABEL2ID),
        eval_metric="mlogloss",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
    )

    clf.fit(X_train_vec, y_train)

    # Evaluate on validation
    print("TF-IDF + XGBoost baseline – validation:")
    y_val_pred = clf.predict(X_val_vec)
    print(classification_report(y_val, y_val_pred, digits=4))
    print("Confusion matrix (val):")
    print(confusion_matrix(y_val, y_val_pred))

    # Evaluate on test
    print("\nTF-IDF + XGBoost baseline – test:")
    y_test_pred = clf.predict(X_test_vec)
    print(classification_report(y_test, y_test_pred, digits=4))
    print("Confusion matrix (test):")
    print(confusion_matrix(y_test, y_test_pred))

import numpy as np

from .stylometry import build_and_save_stylometry_features


def run_stylometry_xgboost_baseline():
    # Ensure stylometry features exist
    build_and_save_stylometry_features(output_path="data/stylometry_features.npy")

    df = pd.read_csv(DATA_PATH)
    df = df[df[LABEL_COLUMN].isin(LABEL2ID.keys())].reset_index(drop=True)

    labels = df[LABEL_COLUMN].map(LABEL2ID).tolist()

    stylo_all = np.load("data/stylometry_features.npy")
    if len(stylo_all) != len(df):
        raise ValueError("Stylometry feature count does not match dataset row count")

    # 70/15/15 split, matching the previous logic
    X_train, X_temp, y_train, y_temp = train_test_split(
        stylo_all, labels, test_size=0.30, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    clf = XGBClassifier(
        objective="multi:softmax",
        num_class=len(LABEL2ID),
        eval_metric="mlogloss",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)

    # Validation
    print("\nStylometry-only + XGBoost baseline – validation:")
    y_val_pred = clf.predict(X_val)
    print(classification_report(y_val, y_val_pred, digits=4))
    print("Confusion matrix (val):")
    print(confusion_matrix(y_val, y_val_pred))

    # Test
    print("\nStylometry-only + XGBoost baseline – test:")
    y_test_pred = clf.predict(X_test)
    print(classification_report(y_test, y_test_pred, digits=4))
    print("Confusion matrix (test):")
    print(confusion_matrix(y_test, y_test_pred))

def main():
    run_tfidf_xgboost_baseline()
    run_stylometry_xgboost_baseline()

if __name__ == "__main__":
    main()

