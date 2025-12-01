"""
plot_model_performance.py
-------------------------

Script to visualize performance of multiple models from a metrics.csv file.

Expected CSV columns:
    model, accuracy, precision, recall, f1, cv_accuracy

It generates two PNG files:
    - model_performance_acc_f1.png  (Accuracy vs F1 per model)
    - model_performance_prf1.png    (Precision / Recall / F1 per model)

Usage (from repo root):
    python src/plot_model_performance.py --metrics metrics.csv --outdir .
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_performance(df: pd.DataFrame, outdir: Path) -> None:
    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    models = df["model"].tolist()
    x = range(len(models))

    acc = df["accuracy"].tolist()
    f1 = df["f1"].tolist()
    prec = df["precision"].tolist()
    rec = df["recall"].tolist()

    # Accuracy vs F1
    width = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar([i - width / 2 for i in x], acc, width=width, label="Accuracy")
    plt.bar([i + width / 2 for i in x], f1, width=width, label="F1")
    plt.xticks(list(x), models, rotation=15)
    plt.ylabel("Score")
    plt.title("Model performance (Accuracy vs F1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "model_performance_acc_f1.png", dpi=300)
    plt.close()

    # Precision / Recall / F1
    width2 = 0.25
    plt.figure(figsize=(7, 4))
    plt.bar([i - width2 for i in x], prec, width=width2, label="Precision")
    plt.bar(x, rec, width=width2, label="Recall")
    plt.bar([i + width2 for i in x], f1, width=width2, label="F1")
    plt.xticks(list(x), models, rotation=15)
    plt.ylabel("Score")
    plt.title("Model performance (Precision / Recall / F1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "model_performance_prf1.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot model performance from metrics.csv"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="metrics.csv",
        help="Path to metrics CSV file.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=".",
        help="Directory to save performance plots.",
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    df = pd.read_csv(metrics_path)
    plot_performance(df, Path(args.outdir))


if __name__ == "__main__":
    main()
