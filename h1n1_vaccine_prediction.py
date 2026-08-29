"""
H1N1 Vaccine Usage Prediction
Portfolio project: Logistic Regression

Goal
----
Predict whether a person received the H1N1 flu vaccine (h1n1_vaccine = 1 or 0)
using survey answers from the National 2009 H1N1 Flu Survey.

Pipeline
--------
1. Load and inspect the data
2. Exploratory data analysis (EDA)
3. Fix missing values
4. Encode categorical features
5. Train / test split (stratified)
6. Balance the training set with SMOTE
7. Train Logistic Regression
8. Evaluate (accuracy, precision, recall, F1, ROC-AUC)
9. Save plots and a cleaned dataset
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "h1n1_vaccine_prediction.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print("=" * 70)
    print("1. LOAD DATA")
    print("=" * 70)
    print(f"Rows: {df.shape[0]:,}   Columns: {df.shape[1]}")
    print("\nColumn names:")
    print(list(df.columns))
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    return df


def explore_data(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("2. EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    target = "h1n1_vaccine"
    print("\nTarget class balance:")
    print(df[target].value_counts())
    print(df[target].value_counts(normalize=True).round(4))

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    missing_pct = (missing / len(df) * 100).round(2)
    missing_table = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    print("\nMissing values (only columns that have them):")
    print(missing_table)
    missing_table.to_csv(OUTPUT_DIR / "missing_values_before_cleaning.csv")

    # Target bar chart
    fig, ax = plt.subplots()
    counts = df[target].value_counts().sort_index()
    ax.bar(["Did not get vaccine (0)", "Got vaccine (1)"], counts.values, color=["#4C78A8", "#F58518"])
    ax.set_ylabel("Number of people")
    ax.set_title("H1N1 Vaccine Uptake")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 200, f"{v:,}", ha="center")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_target_balance.png", dpi=150)
    plt.close(fig)

    # Missingness bar
    fig, ax = plt.subplots(figsize=(10, 7))
    missing_pct.sort_values().plot(kind="barh", ax=ax, color="#4C78A8")
    ax.set_xlabel("Percent missing")
    ax.set_title("Missing Values by Column")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_missing_values.png", dpi=150)
    plt.close(fig)

    # Vaccine rate by doctor recommendation (strong signal from this survey)
    if "dr_recc_h1n1_vacc" in df.columns:
        fig, ax = plt.subplots()
        rate = df.groupby("dr_recc_h1n1_vacc")[target].mean()
        labels = ["Doctor did not recommend", "Doctor recommended"]
        ax.bar(labels, rate.values, color=["#4C78A8", "#F58518"])
        ax.set_ylabel("Share who got H1N1 vaccine")
        ax.set_title("Vaccine Rate by Doctor Recommendation")
        ax.set_ylim(0, 1)
        for i, v in enumerate(rate.values):
            ax.text(i, v + 0.02, f"{v:.1%}", ha="center")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "03_vaccine_by_doctor_rec.png", dpi=150)
        plt.close(fig)

    # Age
    fig, ax = plt.subplots()
    age_order = [
        "18 - 34 Years",
        "35 - 44 Years",
        "45 - 54 Years",
        "55 - 64 Years",
        "65+ Years",
    ]
    rate = df.groupby("age_bracket")[target].mean().reindex(age_order)
    ax.bar(rate.index, rate.values, color="#4C78A8")
    ax.set_ylabel("Share who got H1N1 vaccine")
    ax.set_title("Vaccine Rate by Age Group")
    ax.set_ylim(0, 0.4)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_vaccine_by_age.png", dpi=150)
    plt.close(fig)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix missing values the same way we practiced in class:

    * unique_id is only an ID — drop it so the model does not treat it as a feature
    * has_health_insur is ~46% missing. Missing itself may mean something
      (people who skip insurance questions), so we keep a missing flag
    * Binary / Likert (0-5) columns: fill with the most common value (mode)
    * Text categories: fill with the word 'Unknown' so we do not invent answers
    """
    print("\n" + "=" * 70)
    print("3. CLEAN / FIX MISSING VALUES")
    print("=" * 70)

    cleaned = df.copy()

    # ID is not a predictor
    if "unique_id" in cleaned.columns:
        cleaned = cleaned.drop(columns=["unique_id"])

    # Informative missing flag for the column with the most blanks
    if "has_health_insur" in cleaned.columns:
        cleaned["has_health_insur_missing"] = cleaned["has_health_insur"].isna().astype(int)

    # Numeric / Likert / binary columns
    numeric_cols = cleaned.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "h1n1_vaccine"]

    for col in numeric_cols:
        if cleaned[col].isna().any():
            mode_val = cleaned[col].mode(dropna=True)
            fill_val = mode_val.iloc[0] if len(mode_val) else 0
            n_missing = int(cleaned[col].isna().sum())
            cleaned[col] = cleaned[col].fillna(fill_val)
            print(f"  {col}: filled {n_missing} missing with mode {fill_val}")

    # Text categories
    text_cols = cleaned.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in text_cols:
        if cleaned[col].isna().any():
            n_missing = int(cleaned[col].isna().sum())
            cleaned[col] = cleaned[col].fillna("Unknown")
            print(f"  {col}: filled {n_missing} missing with 'Unknown'")

    leftover = int(cleaned.isnull().sum().sum())
    print(f"\nMissing values remaining: {leftover}")
    assert leftover == 0, "Cleaning failed — still have NaNs"

    cleaned.to_csv(OUTPUT_DIR / "h1n1_cleaned.csv", index=False)
    print(f"Saved cleaned data -> {OUTPUT_DIR / 'h1n1_cleaned.csv'}")
    return cleaned


def encode_features(df: pd.DataFrame):
    """One-hot encode text columns. Likert scores stay numeric."""
    print("\n" + "=" * 70)
    print("4. ENCODE CATEGORICAL FEATURES")
    print("=" * 70)

    y = df["h1n1_vaccine"].astype(int)
    X = df.drop(columns=["h1n1_vaccine"])

    text_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    print("One-hot encoding:", text_cols)
    X = pd.get_dummies(X, columns=text_cols, drop_first=True)

    print(f"Feature matrix shape after encoding: {X.shape}")
    return X, y


def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> None:
    print("\n" + "=" * 70)
    print("5. TRAIN / TEST SPLIT + SMOTE + LOGISTIC REGRESSION")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("Class counts BEFORE SMOTE (training set only):")
    print(y_train.value_counts())

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print("\nClass counts AFTER SMOTE (training set only):")
    print(pd.Series(y_train_smote).value_counts())

    # Compare two models: plain + class_weight balanced on original train,
    # and SMOTE-trained model (the one we report as the main result).
    model = LogisticRegression(max_iter=2000, solver="lbfgs")
    model.fit(X_train_smote, y_train_smote)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    print("\n--- MODEL PERFORMANCE (test set) ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc:.4f}")

    print("\n--- CONFUSION MATRIX ---")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\n--- CLASSIFICATION REPORT ---")
    report = classification_report(y_test, y_pred, digits=4)
    print(report)

    metrics = pd.DataFrame(
        {
            "metric": ["accuracy", "precision", "recall", "f1", "roc_auc"],
            "value": [acc, prec, rec, f1, roc],
        }
    )
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    with open(OUTPUT_DIR / "classification_report.txt", "w") as f:
        f.write("H1N1 Vaccine Prediction — Logistic Regression\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall   : {rec:.4f}\n")
        f.write(f"F1 Score : {f1:.4f}\n")
        f.write(f"ROC-AUC  : {roc:.4f}\n\n")
        f.write("Confusion matrix [[TN FP][FN TP]]:\n")
        f.write(str(cm) + "\n\n")
        f.write(report)

    # Confusion matrix heatmap
    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No vaccine", "Vaccine"],
        yticklabels=["No vaccine", "Vaccine"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_roc_curve.png", dpi=150)
    plt.close(fig)

    # Top coefficients (log-odds)
    coefs = pd.Series(model.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
    top = coefs.head(15).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#F58518" if v > 0 else "#4C78A8" for v in top.values]
    ax.barh(top.index, top.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Logistic Regression coefficient (log-odds)")
    ax.set_title("Top 15 Features by |Coefficient|")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "07_top_coefficients.png", dpi=150)
    plt.close(fig)

    coefs.rename("coefficient").to_csv(OUTPUT_DIR / "feature_coefficients.csv")

    print(f"\nPlots and tables saved in: {OUTPUT_DIR}")
    print("Done.")


def main() -> None:
    df = load_data(DATA_PATH)
    explore_data(df)
    cleaned = clean_data(df)
    X, y = encode_features(cleaned)
    train_and_evaluate(X, y)


if __name__ == "__main__":
    main()
