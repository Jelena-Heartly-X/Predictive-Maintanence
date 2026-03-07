# ============================================================
#  04_lstm.py  —  LSTM RUL Prediction
#  Run: python src/04_lstm.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, time, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_raw
from config      import (
    OUT_DIR, USEFUL_SENSORS, RUL_CLIP, RANDOM_STATE,
    LSTM_WINDOW, LSTM_EPOCHS, LSTM_BATCH_SIZE, LSTM_PATIENCE
)

from sklearn.preprocessing  import MinMaxScaler
from sklearn.metrics        import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models   import Sequential
from tensorflow.keras.layers   import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ── Sequence Builder ─────────────────────────────────────────

def make_sequences(df: pd.DataFrame, sensors: list, window: int = LSTM_WINDOW):
    """
    Slide a window over each engine's cycles.
    Each sample: (window, n_sensors) → RUL at last step.
    """
    X, y = [], []
    for eng in df["engine_id"].unique():
        data = df[df["engine_id"] == eng][sensors + ["RUL"]].values
        for i in range(window, len(data) + 1):
            X.append(data[i - window : i, :-1])
            y.append(data[i - 1, -1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def make_test_sequences(df: pd.DataFrame, sensors: list, window: int = LSTM_WINDOW):
    """
    For each test engine use its last `window` cycles.
    Pad with zeros if the engine has fewer cycles.
    """
    X, y = [], []
    for eng in df["engine_id"].unique():
        data = df[df["engine_id"] == eng][sensors + ["RUL"]].values
        feat = data[:, :-1]
        if len(feat) >= window:
            X.append(feat[-window:])
        else:
            pad = np.zeros((window - len(feat), feat.shape[1]), dtype=np.float32)
            X.append(np.vstack([pad, feat]))
        y.append(data[-1, -1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ── Model ────────────────────────────────────────────────────

def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),

        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ── Plots ────────────────────────────────────────────────────

def plot_training(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(history.history["loss"],     label="Train")
    axes[0].plot(history.history["val_loss"], label="Val")
    axes[0].set_title("LSTM Training Loss (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history.history["mae"],     label="Train")
    axes[1].plot(history.history["val_mae"], label="Val")
    axes[1].set_title("LSTM Training MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "04_lstm_training.png"), bbox_inches="tight")
    print("  Saved: 04_lstm_training.png")
    plt.show()


def plot_predictions(y_test, pred, rmse, mae):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    lim = max(float(y_test.max()), float(pred.max())) + 5
    axes[0].scatter(y_test, pred, alpha=0.5, color="steelblue", s=20)
    axes[0].plot([0, lim], [0, lim], "r--", label="Perfect prediction")
    axes[0].set_xlabel("Actual RUL")
    axes[0].set_ylabel("Predicted RUL")
    axes[0].set_title(f"LSTM: Predicted vs Actual\nRMSE={rmse:.2f}  MAE={mae:.2f}")
    axes[0].legend()

    residuals = pred - y_test
    axes[1].hist(residuals, bins=30, color="coral", edgecolor="white")
    axes[1].axvline(0, color="black", linestyle="--")
    axes[1].set_xlabel("Residual (Predicted − Actual)")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "04_lstm_predictions.png"), bbox_inches="tight")
    print("  Saved: 04_lstm_predictions.png")
    plt.show()


# ── Main ─────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  STEP 4 — LSTM")
    print("=" * 50)

    # 1. Load raw data
    print("\nLoading data...")
    train, test = load_raw()

    # 2. Scale sensors (fit on train only)
    scaler = MinMaxScaler()
    train[USEFUL_SENSORS] = scaler.fit_transform(train[USEFUL_SENSORS])
    test[USEFUL_SENSORS]  = scaler.transform(test[USEFUL_SENSORS])

    # 3. Build sequences
    print("Building sequences...")
    X_train, y_train = make_sequences(train, USEFUL_SENSORS, LSTM_WINDOW)
    X_test,  y_test  = make_test_sequences(test, USEFUL_SENSORS, LSTM_WINDOW)
    print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")

    # 4. Build & summarise model
    model = build_model((LSTM_WINDOW, len(USEFUL_SENSORS)))
    model.summary()

    # 5. Train
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=LSTM_PATIENCE,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, verbose=1),
    ]
    print("\nTraining LSTM...")
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.1f}s")

    # 6. Evaluate
    pred = model.predict(X_test, verbose=0).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae  = mean_absolute_error(y_test, pred)
    print(f"\n  RMSE: {rmse:.4f}   MAE: {mae:.4f}")

    # 7. Plots
    print("\nGenerating plots...")
    plot_training(history)
    plot_predictions(y_test, pred, rmse, mae)

    # 8. Save
    model_path = os.path.join(OUT_DIR, "lstm_model.keras")
    model.save(model_path)
    pd.DataFrame([{
        "model": "LSTM", "rmse": rmse,
        "mae": mae, "train_time_s": elapsed,
    }]).to_csv(os.path.join(OUT_DIR, "04_lstm_results.csv"), index=False)

    print(f"\nSaved model → {model_path}")
    print("Saved: 04_lstm_results.csv")


if __name__ == "__main__":
    main()
