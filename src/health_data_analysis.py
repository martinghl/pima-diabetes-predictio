\
"""
health_data_analysis.py
-----------------------

End-to-end pipeline for predicting diabetes using the Pima Indians dataset.

Steps:
1. Load CSV data.
2. Clean and preprocess (handle zero-as-missing, impute, scale).
3. Train four models:
   - Logistic Regression
   - Random Forest
   - XGBoost
   - Neural Network (MLP)
4. Evaluate on a held-out test set and via 10-fold cross-validation.
5. Save metrics to CSV/JSON and generate standard figures.

Usage
-----
python -m src.health_data_analysis \\
    --data data/pima-indians-diabetes.csv \\
    --outdir .

"""

import argparse
from pathlib import Path

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from xgboost import XGBClassifier


COLUMN_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]


def load_pima_dataset(path: Path) -> pd.DataFrame:
    """Load the Pima Indians Diabetes dataset from a local CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path, header=None, names=COLUMN_NAMES)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Replace zero values in certain columns and impute with median."""
    df_clean = df.copy()
    missing_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in missing_cols:
        df_clean.loc[df_clean[col] == 0, col] = np.nan
        median_val = df_clean[col].median(skipna=True)
        df_clean[col] = df_clean[col].fillna(median_val)
    return df_clean


def split_and_scale(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Split into train/test and standardize features."""
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X, y, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def build_models(seed: int = 42):
    """Return dict of model_name -> estimator."""
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=seed),
        "xgboost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=seed,
            use_label_encoder=False,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(50, 50),
            activation="relu",
            max_iter=1000,
            random_state=seed,
        ),
    }
    return models


def evaluate_model(model, X_test, y_test):
    """Compute metrics and confusion matrix for a fitted model."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def cross_validate_model(model, X_scaled_full, y, cv: int = 10):
    scores = cross_val_score(model, X_scaled_full, y, cv=cv, scoring="accuracy")
    return scores.mean()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def plot_outcome_distribution(y: pd.Series, outdir: Path):
    ensure_dir(outdir)
    counts = y.value_counts().sort_index()
    plt.figure()
    plt.bar(["No diabetes (0)", "Diabetes (1)"], counts.values)
    plt.ylabel("Count")
    plt.title("Outcome distribution")
    for i, v in enumerate(counts.values):
        plt.text(i, v + 5, str(v), ha="center")
    plt.tight_layout()
    plt.savefig(outdir / "outcome_distribution.png", dpi=300)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, outdir: Path):
    ensure_dir(outdir)
    corr = df.corr()
    plt.figure(figsize=(6, 5))
    im = plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.savefig(outdir / "correlation_heatmap.png", dpi=300)
    plt.close()


def plot_confusion(cm: np.ndarray, labels, title: str, outpath: Path):
    ensure_dir(outpath.parent)
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_feature_importance(model, feature_names, title: str, outpath: Path):
    if not hasattr(model, "feature_importances_"):
        return
    ensure_dir(outpath.parent)
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in order]
    sorted_imp = importances[order]

    plt.figure(figsize=(6, 4))
    plt.bar(range(len(sorted_imp)), sorted_imp)
    plt.xticks(range(len(sorted_imp)), sorted_names, rotation=90)
    plt.ylabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def run_pipeline(data_path: Path, outdir: Path, seed: int = 42):
    outdir = outdir.resolve()
    figures_dir = outdir / "figures"
    ensure_dir(figures_dir)

    print(f"Loading data from {data_path}...")
    df_raw = load_pima_dataset(data_path)
    df_clean = preprocess_data(df_raw)

    cleaned_path = outdir / "cleaned_pima_diabetes.csv"
    df_clean.to_csv(cleaned_path, index=False)
    print(f"Cleaned data saved to {cleaned_path}")

    plot_outcome_distribution(df_clean["Outcome"], figures_dir)
    plot_correlation_heatmap(df_clean, figures_dir)

    X, y, X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale(
        df_clean, seed=seed
    )
    X_scaled_full = scaler.fit_transform(X)

    models = build_models(seed=seed)
    metrics_list = []

    for name, model in models.items():
        print(f"\nTraining model: {name}")
        model.fit(X_train_scaled, y_train)
        metrics = evaluate_model(model, X_test_scaled, y_test)
        cv_acc = cross_validate_model(model, X_scaled_full, y)

        cm = metrics["confusion_matrix"]
        plot_confusion(
            cm,
            labels=["No diabetes", "Diabetes"],
            title=f"Confusion matrix – {name}",
            outpath=figures_dir / f"confusion_matrix_{name}.png",
        )

        if name in {"random_forest", "xgboost"}:
            plot_feature_importance(
                model,
                feature_names=X.columns,
                title=f"Feature importance – {name}",
                outpath=figures_dir / f"feature_importance_{name}.png",
            )

        row = {
            "model": name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "cv_accuracy": cv_acc,
        }
        metrics_list.append(row)

        print(
            f"{name}: acc={metrics['accuracy']:.3f}, "
            f"prec={metrics['precision']:.3f}, "
            f"recall={metrics['recall']:.3f}, "
            f"f1={metrics['f1']:.3f}, "
            f"cv_acc={cv_acc:.3f}"
        )

    metrics_df = pd.DataFrame(metrics_list)
    metrics_csv = outdir / "metrics.csv"
    metrics_json = outdir / "metrics.json"
    metrics_df.to_csv(metrics_csv, index=False)
    metrics_df.to_json(metrics_json, orient="records", indent=2)
    print(f"\nMetrics saved to {metrics_csv} and {metrics_json}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pima Indians Diabetes prediction pipeline"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/pima-indians-diabetes.csv",
        help="Path to CSV file with Pima dataset.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=".",
        help="Output directory for cleaned data, metrics, and figures.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    outdir = Path(args.outdir)
    run_pipeline(data_path, outdir, seed=args.seed)


if __name__ == "__main__":
    main()
