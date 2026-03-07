# ============================================================
#  01_eda.py  —  Exploratory Data Analysis
#  Run: python src/01_eda.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_raw
from config import USEFUL_SENSORS, SENSOR_COLS, DROP_SENSORS, OUT_DIR

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams["figure.dpi"] = 120


def plot_rul_distribution(train: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].hist(train["RUL"], bins=50, color="steelblue", edgecolor="white")
    axes[0].set_title("RUL Distribution (Training Set)", fontsize=13)
    axes[0].set_xlabel("Remaining Useful Life (cycles)")
    axes[0].set_ylabel("Count")

    engine_life = (
        train.groupby("engine_id")["cycle"].max().sort_values().values
    )
    axes[1].bar(range(len(engine_life)), engine_life,
                color="coral", edgecolor="white")
    axes[1].set_title("Engine Lifecycle Lengths", fontsize=13)
    axes[1].set_xlabel("Engine (sorted by lifespan)")
    axes[1].set_ylabel("Total Cycles Before Failure")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "01_rul_distribution.png"), bbox_inches="tight")
    print("  Saved: 01_rul_distribution.png")
    plt.show()


def plot_sensor_variance(train: pd.DataFrame):
    std = train[SENSOR_COLS].std().sort_values()
    colors = ["crimson" if s in DROP_SENSORS else "steelblue" for s in std.index]

    plt.figure(figsize=(13, 4))
    plt.bar(std.index, std.values, color=colors)
    plt.axhline(0.01, color="black", linestyle="--", label="Drop threshold (std < 0.01)")
    plt.title("Sensor Standard Deviation  (Red = dropped)", fontsize=13)
    plt.xlabel("Sensor")
    plt.ylabel("Std Dev")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "01_sensor_variance.png"), bbox_inches="tight")
    print("  Saved: 01_sensor_variance.png")
    plt.show()


def plot_degradation_curves(train: pd.DataFrame):
    # Normalise each engine's cycle to [0, 1]
    train = train.copy()
    train["cycle_norm"] = train.groupby("engine_id")["cycle"].transform(
        lambda x: x / x.max()
    )

    # Pick 6 sensors that visually degrade
    viz_sensors = [s for s in ["s2", "s3", "s4", "s7", "s11", "s12",
                                "s15", "s20", "s21"]
                   if s in USEFUL_SENSORS][:6]

    sample_engines = train["engine_id"].unique()[:15]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()

    for idx, sensor in enumerate(viz_sensors):
        for eng in sample_engines:
            d = train[train["engine_id"] == eng]
            axes[idx].plot(d["cycle_norm"], d[sensor],
                           alpha=0.35, linewidth=0.8, color="steelblue")
        axes[idx].set_title(f"{sensor} — Degradation Pattern", fontsize=11)
        axes[idx].set_xlabel("Normalised Cycle (0=new, 1=failure)")
        axes[idx].set_ylabel("Sensor Value")

    plt.suptitle("Sensor Readings Over Engine Lifetime (15 engines)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "01_sensor_degradation.png"), bbox_inches="tight")
    print("  Saved: 01_sensor_degradation.png")
    plt.show()


def plot_correlation(train: pd.DataFrame):
    corr_cols = USEFUL_SENSORS + ["RUL"]
    corr = train[corr_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    plt.title("Sensor Correlation Heatmap (incl. RUL)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "01_correlation_heatmap.png"), bbox_inches="tight")
    print("  Saved: 01_correlation_heatmap.png")
    plt.show()

    print("\nTop sensors correlated with RUL:")
    print(corr["RUL"].drop("RUL").abs().sort_values(ascending=False).head(8))


def main():
    print("=" * 50)
    print("  STEP 1 — Exploratory Data Analysis")
    print("=" * 50)

    print("\nLoading dataset...")
    train, _ = load_raw()
    print(f"  Train shape : {train.shape}")
    print(f"  Engines     : {train['engine_id'].nunique()}")
    print(f"  RUL range   : {train['RUL'].min()} – {train['RUL'].max()}")

    print("\nDropped sensors (near-zero variance):", DROP_SENSORS)
    print("Useful sensors :", USEFUL_SENSORS)

    print("\nGenerating plots...")
    plot_rul_distribution(train)
    plot_sensor_variance(train)
    plot_degradation_curves(train)
    plot_correlation(train)

    print("\nEDA complete. All plots saved to outputs/")


if __name__ == "__main__":
    main()
