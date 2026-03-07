# ============================================================
#  features.py  —  Rolling stats, lag features, scaling
# ============================================================

import pandas as pd
import numpy as np
import pickle
import os

from sklearn.preprocessing import MinMaxScaler
from config import USEFUL_SENSORS, ROLLING_WINDOWS, LAG_STEPS, OUT_DIR


# ── Public API ───────────────────────────────────────────────

def build_features(train: pd.DataFrame, test: pd.DataFrame):
    """
    Full feature engineering pipeline.

    Steps
    -----
    1. Drop low-variance sensors
    2. Add rolling mean & std (windows: 5, 10, 30 cycles)
    3. Add lag features (t-1, t-2, t-3)
    4. Scale with MinMaxScaler fit on train only

    Returns
    -------
    train_fe, test_fe : pd.DataFrame
    feature_cols      : list[str]   (all feature column names)
    scaler            : MinMaxScaler
    """
    keep = ["engine_id", "cycle"] + USEFUL_SENSORS + ["RUL"]
    train = train[keep].copy()
    test  = test[keep].copy()

    print("  → Adding rolling features...")
    train = _add_rolling(train, USEFUL_SENSORS, ROLLING_WINDOWS)
    test  = _add_rolling(test,  USEFUL_SENSORS, ROLLING_WINDOWS)

    print("  → Adding lag features...")
    train = _add_lags(train, USEFUL_SENSORS, LAG_STEPS)
    test  = _add_lags(test,  USEFUL_SENSORS, LAG_STEPS)

    feature_cols = [
        c for c in train.columns
        if c not in ("engine_id", "cycle", "RUL")
    ]

    print("  → Scaling features...")
    scaler = MinMaxScaler()
    train[feature_cols] = scaler.fit_transform(train[feature_cols])
    test[feature_cols]  = scaler.transform(test[feature_cols])

    # Persist scaler for inference
    scaler_path = os.path.join(OUT_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"  → Feature count : {len(feature_cols)}")
    return train, test, feature_cols, scaler


# ── Helpers ──────────────────────────────────────────────────

def _add_rolling(df: pd.DataFrame, sensors: list, windows: list) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        for s in sensors:
            grp = df.groupby("engine_id")[s]
            df[f"{s}_rmean{w}"] = grp.transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
            df[f"{s}_rstd{w}"] = grp.transform(
                lambda x: x.rolling(w, min_periods=1).std().fillna(0)
            )
    return df


def _add_lags(df: pd.DataFrame, sensors: list, lags: list) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        for s in sensors:
            df[f"{s}_lag{lag}"] = (
                df.groupby("engine_id")[s]
                .transform(lambda x: x.shift(lag).bfill())
            )
    return df


if __name__ == "__main__":
    from data_loader import load_raw
    train, test = load_raw()
    print("Building features...")
    train_fe, test_fe, feature_cols, scaler = build_features(train, test)
    print(f"Train: {train_fe.shape}  Test: {test_fe.shape}")
    print("Sample cols:", feature_cols[:6])
