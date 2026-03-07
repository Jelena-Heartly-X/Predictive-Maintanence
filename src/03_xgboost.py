# ============================================================
#  03_xgboost.py  —  XGBoost RUL Prediction
#  Run: python src/03_xgboost.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle, time, os, sys, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_raw
from features    import build_features
from config      import OUT_DIR, ALERT_THRESHOLD, RANDOM_STATE

import xgboost as xgb
from sklearn.metrics      import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV


# ── Data Prep ────────────────────────────────────────────────

def prepare_data(train_fe, test_fe, feature_cols):
    X_train = train_fe[feature_cols]
    y_train = train_fe["RUL"]

    # Test: predict RUL at the last observed cycle of each engine
    test_last = test_fe.groupby("engine_id").last().reset_index()
    X_test = test_last[feature_cols]
    y_test = test_last["RUL"]
    return X_train, y_train, X_test, y_test, test_last


# ── Train ────────────────────────────────────────────────────

def train_xgboost(X_train, y_train, X_test, y_test):
    print("\n── Baseline XGBoost ──")
    t0 = time.time()

    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    elapsed = time.time() - t0
    pred    = model.predict(X_test)
    rmse    = np.sqrt(mean_squared_error(y_test, pred))
    mae     = mean_absolute_error(y_test, pred)
    print(f"  RMSE: {rmse:.4f}   MAE: {mae:.4f}   Time: {elapsed:.1f}s")
    return model, pred, rmse, mae, elapsed


def tune_xgboost(X_train, y_train):
    print("\n── GridSearchCV Tuning (3-fold, this takes a few minutes) ──")

    param_grid = {
        "max_depth"       : [4, 6, 8],
        "learning_rate"   : [0.01, 0.05, 0.1],
        "n_estimators"    : [200, 400],
        "subsample"       : [0.7, 0.9],
        "min_child_weight": [1, 3],
    }
    base = xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    gs   = GridSearchCV(base, param_grid, cv=3,
                        scoring="neg_root_mean_squared_error",
                        n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)
    print(f"  Best params  : {gs.best_params_}")
    print(f"  Best CV RMSE : {-gs.best_score_:.4f}")
    return gs.best_params_


def train_best(X_train, y_train, X_test, y_test, best_params):
    print("\n── Final Model (best params) ──")
    t0 = time.time()

    model = xgb.XGBRegressor(
        **best_params,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    elapsed = time.time() - t0
    pred    = model.predict(X_test)
    rmse    = np.sqrt(mean_squared_error(y_test, pred))
    mae     = mean_absolute_error(y_test, pred)
    print(f"  RMSE: {rmse:.4f}   MAE: {mae:.4f}   Time: {elapsed:.1f}s")
    return model, pred, rmse, mae, elapsed


# ── Plots ────────────────────────────────────────────────────

def plot_predictions(y_test, pred, rmse, mae):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    lim = max(float(y_test.max()), float(pred.max())) + 5
    axes[0].scatter(y_test, pred, alpha=0.5, color="steelblue", s=20)
    axes[0].plot([0, lim], [0, lim], "r--", label="Perfect prediction")
    axes[0].set_xlabel("Actual RUL")
    axes[0].set_ylabel("Predicted RUL")
    axes[0].set_title(f"XGBoost: Predicted vs Actual\nRMSE={rmse:.2f}  MAE={mae:.2f}")
    axes[0].legend()

    residuals = pred - y_test.values
    axes[1].hist(residuals, bins=30, color="coral", edgecolor="white")
    axes[1].axvline(0, color="black", linestyle="--")
    axes[1].set_xlabel("Residual (Predicted − Actual)")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_xgb_predictions.png"), bbox_inches="tight")
    print("  Saved: 03_xgb_predictions.png")
    plt.show()


def plot_feature_importance(model, feature_cols):
    importance = (
        pd.Series(model.feature_importances_, index=feature_cols)
        .sort_values(ascending=False)
        .head(20)
    )
    plt.figure(figsize=(12, 5))
    importance.plot(kind="bar", color="steelblue", edgecolor="white")
    plt.title("Top 20 Feature Importances — XGBoost", fontsize=13)
    plt.xlabel("Feature")
    plt.ylabel("Importance Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_xgb_feature_importance.png"), bbox_inches="tight")
    print("  Saved: 03_xgb_feature_importance.png")
    plt.show()
    print("\nTop 10 features:")
    print(importance.head(10).to_string())


# ── Early Warning System ─────────────────────────────────────

def early_warning(test_last, pred, y_test):
    print(f"\n── Early Warning System (threshold ≤ {ALERT_THRESHOLD} cycles) ──")
    df = test_last.copy()
    df["pred_RUL"]   = pred
    df["actual_RUL"] = y_test.values
    df["ALERT"]      = df["pred_RUL"] <= ALERT_THRESHOLD

    true_alerts  = ((df["ALERT"]) & (df["actual_RUL"] <= ALERT_THRESHOLD)).sum()
    false_alerts = ((df["ALERT"]) & (df["actual_RUL"] >  ALERT_THRESHOLD)).sum()
    missed       = ((~df["ALERT"]) & (df["actual_RUL"] <= ALERT_THRESHOLD)).sum()

    print(f"  True Alerts     (caught)  : {true_alerts}")
    print(f"  False Alerts    (noise)   : {false_alerts}")
    print(f"  Missed Failures (danger!) : {missed}")


# ── Main ─────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  STEP 3 — XGBoost")
    print("=" * 50)

    # 1. Load & engineer features
    print("\nLoading & engineering features...")
    train, test = load_raw()
    train_fe, test_fe, feature_cols, _ = build_features(train, test)
    X_train, y_train, X_test, y_test, test_last = prepare_data(
        train_fe, test_fe, feature_cols
    )

    # 2. Baseline
    _, pred_base, rmse_base, mae_base, t_base = train_xgboost(
        X_train, y_train, X_test, y_test
    )

    # 3. Tune  (comment out if you want a quick run)
    best_params = tune_xgboost(X_train, y_train)

    # 4. Final model
    best_model, pred_best, rmse_best, mae_best, t_best = train_best(
        X_train, y_train, X_test, y_test, best_params
    )

    # 5. Plots
    print("\nGenerating plots...")
    plot_predictions(y_test, pred_best, rmse_best, mae_best)
    plot_feature_importance(best_model, feature_cols)

    # 6. Early warning
    early_warning(test_last, pred_best, y_test)

    # 7. Save
    model_path = os.path.join(OUT_DIR, "xgb_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    pd.DataFrame([{
        "model": "XGBoost", "rmse": rmse_best,
        "mae": mae_best, "train_time_s": t_best,
    }]).to_csv(os.path.join(OUT_DIR, "03_xgb_results.csv"), index=False)

    print(f"\nSaved model → {model_path}")
    print("Saved: 03_xgb_results.csv")


if __name__ == "__main__":
    main()
