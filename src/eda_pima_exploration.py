
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def load_data(path: Path) -> pd.DataFrame:

    df_try = pd.read_csv(path)
    if set(COLUMN_NAMES).issubset(set(df_try.columns)):
        # Reorder columns just in case
        df = df_try[COLUMN_NAMES]
    else:
        df = pd.read_csv(path, header=None, names=COLUMN_NAMES)
    return df


def basic_summary(df: pd.DataFrame, outdir: Path):
    print("Data shape:", df.shape)
    print("\nHead:")
    print(df.head())

    desc = df.describe()
    print("\nDescriptive statistics:")
    print(desc)

    outdir.mkdir(parents=True, exist_ok=True)
    desc.to_csv(outdir / "eda_descriptive_stats.csv")


def plot_outcome_distribution(df: pd.DataFrame, figdir: Path):
    figdir.mkdir(parents=True, exist_ok=True)
    counts = df["Outcome"].value_counts().sort_index()

    plt.figure()
    plt.bar(["No diabetes (0)", "Diabetes (1)"], counts.values)
    plt.ylabel("Count")
    plt.title("Outcome distribution")
    for i, v in enumerate(counts.values):
        plt.text(i, v + 5, str(v), ha="center")
    plt.tight_layout()
    plt.savefig(figdir / "outcome_distribution.png", dpi=300)
    plt.close()


def plot_histograms_by_outcome(df: pd.DataFrame, figdir: Path):

    features = COLUMN_NAMES[:-1]  # exclude Outcome
    figdir.mkdir(parents=True, exist_ok=True)

    df0 = df[df["Outcome"] == 0]
    df1 = df[df["Outcome"] == 1]

    for col in features:
        plt.figure()
        # default colors from matplotlib's color cycle (no explicit color)
        plt.hist(df0[col].dropna(), bins=30, alpha=0.6, label="Outcome 0")
        plt.hist(df1[col].dropna(), bins=30, alpha=0.6, label="Outcome 1")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.title(f"Histogram of {col} by Outcome")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figdir / f"hist_{col}_by_outcome.png", dpi=300)
        plt.close()


def plot_boxplots_by_outcome(df: pd.DataFrame, figdir: Path):
    features = COLUMN_NAMES[:-1]
    figdir.mkdir(parents=True, exist_ok=True)

    for col in features:
        plt.figure()
        data_0 = df.loc[df["Outcome"] == 0, col].dropna()
        data_1 = df.loc[df["Outcome"] == 1, col].dropna()
        plt.boxplot([data_0.values, data_1.values], labels=["Outcome 0", "Outcome 1"])
        plt.ylabel(col)
        plt.title(f"Boxplot of {col} by Outcome")
        plt.tight_layout()
        plt.savefig(figdir / f"boxplot_{col}_by_outcome.png", dpi=300)
        plt.close()


def plot_scatter_pairs(df: pd.DataFrame, figdir: Path):

    figdir.mkdir(parents=True, exist_ok=True)
    pairs = [
        ("Age", "Glucose"),
        ("Age", "BMI"),
        ("Glucose", "BMI"),
    ]

    df0 = df[df["Outcome"] == 0]
    df1 = df[df["Outcome"] == 1]

    for x_col, y_col in pairs:
        plt.figure()
        plt.scatter(df0[x_col], df0[y_col], alpha=0.6, label="Outcome 0")
        plt.scatter(df1[x_col], df1[y_col], alpha=0.6, label="Outcome 1")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"{y_col} vs {x_col} by Outcome")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figdir / f"scatter_{y_col}_vs_{x_col}_by_outcome.png", dpi=300)
        plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, figdir: Path):
    figdir.mkdir(parents=True, exist_ok=True)
    corr = df.corr()

    plt.figure(figsize=(6, 5))
    im = plt.imshow(corr.values, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.savefig(figdir / "correlation_heatmap.png", dpi=300)
    plt.close()


def run_eda(data_path: Path, outdir: Path):
    outdir = outdir.resolve()
    figdir = outdir / "figures_eda"

    print(f"Loading data from {data_path} ...")
    df = load_data(data_path)

    print("\n=== BASIC SUMMARY ===")
    basic_summary(df, outdir)

    print("\nGenerating EDA plots into:", figdir)
    plot_outcome_distribution(df, figdir)
    plot_histograms_by_outcome(df, figdir)
    plot_boxplots_by_outcome(df, figdir)
    plot_scatter_pairs(df, figdir)
    plot_correlation_heatmap(df, figdir)

    print("EDA complete.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exploratory data analysis for the Pima Indians Diabetes dataset"
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
        help="Output directory for EDA summaries and figures.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eda(Path(args.data), Path(args.outdir))
