# ============================================================
#  02_sarima.py  —  SARIMA Baseline Model
#  Run: python src/02_sarima.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_raw
from config import OUT_DIR, SUBSET

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error


SENSOR   = "s11"    # most informative degradation sensor for FD001
N_ENGINES = 10      # how many engines to evaluate (more = slower)


# ── Helpers ──────────────────────────────────────────────────

def adf_test(series: np.ndarray, name: str = "Series"):
    result = adfuller(series)
    print(f"  ADF Test — {name}")
    print(f"    p-value    : {result[1]:.4f}")
    print(f"    Stationary : {'YES' if result[1] < 0.05 else 'NO (needs differencing)'}")


def fit_sarima(series: np.ndarray):
    model = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 0, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def evaluate(actual: np.ndarray, predicted: np.ndarray):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae  = mean_absolute_error(actual, predicted)
    return rmse, mae


# ── Single Engine Deep-dive ───────────────────────────────────

def single_engine_analysis(train_raw: pd.DataFrame):
    print("\n── Single Engine Analysis ──")

    # Pick median-lifespan engine
    lives = train_raw.groupby("engine_id")["cycle"].max()
    target = (lives - lives.median()).abs().idxmin()
    print(f"  Engine {target}  (life = {lives[target]} cycles)")

    series = train_raw[train_raw["engine_id"] == target][SENSOR].values
    split  = int(len(series) * 0.8)
    tr, te = series[:split], series[split:]

    # Stationarity
    adf_test(tr,               f"{SENSOR} (raw)")
    adf_test(np.diff(tr),      f"{SENSOR} (1st diff)")

    # Fit
    fitted   = fit_sarima(tr)
    forecast = fitted.get_forecast(steps=len(te))
    pred     = forecast.predicted_mean
    ci       = forecast.conf_int()

    rmse, mae = evaluate(te, pred)
    print(f"\n  SARIMA  RMSE: {rmse:.3f}   MAE: {mae:.3f}")

    # Plot
    cycles = train_raw[train_raw["engine_id"] == target]["cycle"].values
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(cycles[:split], tr,   label="Train",           color="steelblue")
    ax.plot(cycles[split:], te,   label="Actual",          color="green")
    ax.plot(cycles[split:], pred, label="SARIMA Forecast", color="red", linestyle="--")
    ax.fill_between(cycles[split:], ci.iloc[:, 0], ci.iloc[:, 1],
                    alpha=0.2, color="red", label="95% CI")
    ax.axvline(cycles[split - 1], color="black", linestyle=":", linewidth=1.5,
               label="Train / Test split")
    ax.set_title(f"SARIMA Forecast — Engine {target}, Sensor {SENSOR}", fontsize=13)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Sensor Value")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "02_sarima_single_engine.png"), bbox_inches="tight")
    print("  Saved: 02_sarima_single_engine.png")
    plt.show()

    return rmse, mae


# ── Multi-Engine Evaluation ───────────────────────────────────

def multi_engine_evaluation(train_raw: pd.DataFrame):
    print(f"\n── Multi-Engine Evaluation (n={N_ENGINES}) ──")
    engines = train_raw["engine_id"].unique()[:N_ENGINES]
    results = []

    for eng in engines:
        series = train_raw[train_raw["engine_id"] == eng][SENSOR].values
        if len(series) < 30:
            continue
        split = int(len(series) * 0.8)
        tr, te = series[:split], series[split:]
        try:
            fitted   = fit_sarima(tr)
            pred     = fitted.get_forecast(steps=len(te)).predicted_mean
            rmse, mae = evaluate(te, pred)
            results.append({"engine": eng, "RMSE": rmse, "MAE": mae})
            print(f"  Engine {eng:3d}  RMSE: {rmse:.3f}   MAE: {mae:.3f}")
        except Exception as e:
            print(f"  Engine {eng}: skipped ({e})")

    df = pd.DataFrame(results)
    print(f"\n  ── Average over {len(df)} engines ──")
    print(f"  Mean RMSE : {df['RMSE'].mean():.4f}")
    print(f"  Mean MAE  : {df['MAE'].mean():.4f}")
    df.to_csv(os.path.join(OUT_DIR, "02_sarima_results.csv"), index=False)
    print("  Saved: 02_sarima_results.csv")
    return df


# ── Main ─────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  STEP 2 — SARIMA Baseline")
    print("=" * 50)

    train, _ = load_raw()

    single_engine_analysis(train)
    multi_engine_evaluation(train)

    print("\n── Why SARIMA Struggles Here ──")
    print("  1. Univariate: ignores cross-sensor interactions")
    print("  2. Non-linear degradation violates ARIMA assumptions")
    print("  3. Must be re-fit per engine — not scalable")
    print("  4. Forecasts sensor values, not RUL directly")
    print("\n  → These limitations motivate XGBoost and LSTM.")


if __name__ == "__main__":
    main()
