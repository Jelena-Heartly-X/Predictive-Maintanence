# ============================================================
#  config.py  —  All project-wide constants in one place
# ============================================================

import os

# ── Paths ────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data", "CMaps")
OUT_DIR    = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUT_DIR, exist_ok=True)

# ── Dataset ──────────────────────────────────────────────────
SUBSET = "FD001"          # change to FD002/FD003/FD004 to experiment

COLS = (
    ["engine_id", "cycle", "op1", "op2", "op3"]
    + [f"s{i}" for i in range(1, 22)]
)

# Sensors with near-zero variance — carry no information
DROP_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
SENSOR_COLS  = [f"s{i}" for i in range(1, 22)]
USEFUL_SENSORS = [s for s in SENSOR_COLS if s not in DROP_SENSORS]

# ── Feature Engineering ──────────────────────────────────────
ROLLING_WINDOWS = [5, 10, 30]
LAG_STEPS       = [1, 2, 3]
RUL_CLIP        = 125          # piecewise-linear RUL cap (standard for CMAPSS)

# ── Models ───────────────────────────────────────────────────
LSTM_WINDOW     = 30           # look-back window in cycles
LSTM_EPOCHS     = 100
LSTM_BATCH_SIZE = 256
LSTM_PATIENCE   = 10           # early stopping patience

RANDOM_STATE    = 42
ALERT_THRESHOLD = 30           # flag engines with RUL ≤ this value
