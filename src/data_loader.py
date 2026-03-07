# ============================================================
#  data_loader.py  —  Load & label the CMAPSS dataset
# ============================================================

import pandas as pd
import numpy as np
from config import DATA_DIR, COLS, SUBSET, RUL_CLIP


def load_raw(subset: str = SUBSET):
    """
    Load train, test, and RUL files for a given subset.

    Returns
    -------
    train_df : pd.DataFrame  (with RUL column added)
    test_df  : pd.DataFrame  (with RUL column added)
    """
    base = DATA_DIR

    train = pd.read_csv(
        f"{base}/train_{subset}.txt",
        sep=r"\s+", header=None, names=COLS
    )
    test = pd.read_csv(
        f"{base}/test_{subset}.txt",
        sep=r"\s+", header=None, names=COLS
    )
    rul_file = pd.read_csv(
        f"{base}/RUL_{subset}.txt",
        header=None, names=["RUL"]
    )

    train = _add_train_rul(train)
    test  = _add_test_rul(test, rul_file)
    return train, test


def _add_train_rul(df: pd.DataFrame) -> pd.DataFrame:
    """RUL = cycles until the engine actually failed."""
    max_cycle = (
        df.groupby("engine_id")["cycle"]
        .max()
        .reset_index()
        .rename(columns={"cycle": "max_cycle"})
    )
    df = df.merge(max_cycle, on="engine_id")
    df["RUL"] = (df["max_cycle"] - df["cycle"]).clip(upper=RUL_CLIP)
    df.drop("max_cycle", axis=1, inplace=True)
    return df


def _add_test_rul(df: pd.DataFrame, rul_file: pd.DataFrame) -> pd.DataFrame:
    """
    Test RUL = cycles remaining in the test window
               + the known RUL at the last observed cycle.
    """
    last_cycle = (
        df.groupby("engine_id")["cycle"]
        .max()
        .reset_index()
        .rename(columns={"cycle": "max_cycle"})
    )
    df = df.merge(last_cycle, on="engine_id")

    rul_map = {
        i + 1: v
        for i, v in enumerate(rul_file["RUL"].values)
    }
    df["rul_at_end"] = df["engine_id"].map(rul_map)
    df["RUL"] = (
        (df["max_cycle"] - df["cycle"] + df["rul_at_end"])
        .clip(upper=RUL_CLIP)
    )
    df.drop(["max_cycle", "rul_at_end"], axis=1, inplace=True)
    return df


if __name__ == "__main__":
    train, test = load_raw()
    print(f"Train : {train.shape}")
    print(f"Test  : {test.shape}")
    print(train[["engine_id", "cycle", "RUL"]].head())
