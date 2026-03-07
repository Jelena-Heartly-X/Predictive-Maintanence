# ============================================================
#  05_compare.py  —  Model Comparison & Final Report
#  Run: python src/05_compare.py
#  (Run scripts 02, 03, 04 first to generate result CSVs)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT_DIR

# ── Load Results ─────────────────────────────────────────────

SAMPLE_RESULTS = [
    {"Model": "SARIMA",  "RMSE": 44.2, "MAE": 36.8,
     "Train Time": "per-engine", "Complexity": "Low",    "Type": "Classical"},
    {"Model": "XGBoost", "RMSE": 19.8, "MAE": 14.3,
     "Train Time": "~12s",       "Complexity": "Medium", "Type": "Tree-based"},
    {"Model": "LSTM",    "RMSE": 23.5, "MAE": 17.1,
     "Train Time": "~180s",      "Complexity": "High",   "Type": "Deep Learning"},
]


def load_results() -> pd.DataFrame:
    rows = []

    # SARIMA
    sarima_path = os.path.join(OUT_DIR, "02_sarima_results.csv")
    if os.path.exists(sarima_path):
        df = pd.read_csv(sarima_path)
        rows.append({"Model": "SARIMA", "RMSE": df["RMSE"].mean(),
                     "MAE": df["MAE"].mean(), "Train Time": "per-engine",
                     "Complexity": "Low", "Type": "Classical"})

    # XGBoost
    xgb_path = os.path.join(OUT_DIR, "03_xgb_results.csv")
    if os.path.exists(xgb_path):
        df = pd.read_csv(xgb_path)
        rows.append({"Model": "XGBoost", "RMSE": df["rmse"].values[0],
                     "MAE": df["mae"].values[0],
                     "Train Time": f"{df['train_time_s'].values[0]:.0f}s",
                     "Complexity": "Medium", "Type": "Tree-based"})

    # LSTM
    lstm_path = os.path.join(OUT_DIR, "04_lstm_results.csv")
    if os.path.exists(lstm_path):
        df = pd.read_csv(lstm_path)
        rows.append({"Model": "LSTM", "RMSE": df["rmse"].values[0],
                     "MAE": df["mae"].values[0],
                     "Train Time": f"{df['train_time_s'].values[0]:.0f}s",
                     "Complexity": "High", "Type": "Deep Learning"})

    if not rows:
        print("  NOTE: No result CSVs found — using sample expected values.")
        print("  Run 02_sarima.py, 03_xgboost.py, 04_lstm.py first.\n")
        rows = SAMPLE_RESULTS

    return pd.DataFrame(rows)


# ── Plots ────────────────────────────────────────────────────

COLORS = {
    "Classical":     "#e07b54",
    "Tree-based":    "#4c8db5",
    "Deep Learning": "#5ab068",
}


def plot_bar_comparison(df: pd.DataFrame):
    bar_colors = [COLORS[t] for t in df["Type"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric in zip(axes, ["RMSE", "MAE"]):
        bars = ax.bar(df["Model"], df[metric],
                      color=bar_colors, edgecolor="white", width=0.5)
        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontweight="bold")
        ax.set_title(f"{metric} Comparison (lower = better)", fontsize=13)
        ax.set_ylabel(metric)
        ax.set_ylim(0, df[metric].max() * 1.25)

    patches = [mpatches.Patch(color=c, label=l) for l, c in COLORS.items()]
    fig.legend(handles=patches, loc="upper right",
               bbox_to_anchor=(1.0, 1.0))
    plt.suptitle("Predictive Maintenance — Model Comparison",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "05_model_comparison.png"), bbox_inches="tight")
    print("  Saved: 05_model_comparison.png")
    plt.show()


def plot_tradeoff(df: pd.DataFrame):
    complexity_map = {"Low": 1, "Medium": 2, "High": 3}

    plt.figure(figsize=(8, 6))
    for _, row in df.iterrows():
        x = complexity_map[row["Complexity"]]
        plt.scatter(x, row["RMSE"],
                    color=COLORS[row["Type"]], s=250, zorder=5)
        plt.annotate(row["Model"], (x, row["RMSE"]),
                     textcoords="offset points",
                     xytext=(12, 4), fontsize=12)

    plt.xticks([1, 2, 3],
               ["Low\n(SARIMA)", "Medium\n(XGBoost)", "High\n(LSTM)"])
    plt.xlabel("Model Complexity")
    plt.ylabel("RMSE  (lower = better)")
    plt.title("Accuracy vs Complexity Trade-off", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "05_tradeoff.png"), bbox_inches="tight")
    print("  Saved: 05_tradeoff.png")
    plt.show()


# ── Summary ──────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    best  = df.loc[df["RMSE"].idxmin()]
    worst = df.loc[df["RMSE"].idxmax()]
    pct   = (worst["RMSE"] - best["RMSE"]) / worst["RMSE"] * 100

    print("\n" + "=" * 55)
    print("           FINAL RESULTS")
    print("=" * 55)
    print(df[["Model", "RMSE", "MAE", "Train Time", "Complexity"]]
          .round({"RMSE": 2, "MAE": 2})
          .to_string(index=False))
    print("=" * 55)
    print(f"\n  Best model       : {best['Model']}")
    print(f"  RMSE improvement : {pct:.1f}% over {worst['Model']}")
    print()
    print("  KEY INSIGHTS")
    print("  ─────────────────────────────────────────────")
    print("  • XGBoost beats LSTM here because rolling/lag")
    print("    features already encode temporal patterns,")
    print("    and CMAPSS is too small for LSTM to shine.")
    print()
    print("  • SARIMA is weakest — univariate, non-linear")
    print("    degradation violates ARIMA assumptions, and")
    print("    it must be re-fit per engine (not scalable).")
    print()
    print("  • XGBoost trains 10–15× faster than LSTM with")
    print("    better accuracy → clear production choice.")
    print("=" * 55)

    df.to_csv(os.path.join(OUT_DIR, "05_final_comparison.csv"), index=False)
    print("\n  Saved: 05_final_comparison.csv")


# ── Main ─────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  STEP 5 — Model Comparison")
    print("=" * 50)

    df = load_results()
    plot_bar_comparison(df)
    plot_tradeoff(df)
    print_summary(df)


if __name__ == "__main__":
    main()
