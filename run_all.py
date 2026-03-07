# ============================================================
#  run_all.py  —  Run the entire pipeline end to end
#  Run: python run_all.py
#
#  Or run individual steps:
#    python src/01_eda.py
#    python src/02_sarima.py
#    python src/03_xgboost.py
#    python src/04_lstm.py
#    python src/05_compare.py
# ============================================================

import sys
import os

# Add src/ to path so all scripts can import each other
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

def run_step(name, func):
    print(f"\n{'#'*60}")
    print(f"#  {name}")
    print(f"{'#'*60}")
    try:
        func()
    except Exception as e:
        print(f"\n  ⚠  {name} failed: {e}")
        print("     Continuing to next step...\n")


if __name__ == "__main__":
    import importlib

    steps = [
        ("EDA",        "01_eda",      "main"),
        ("SARIMA",     "02_sarima",   "main"),
        ("XGBoost",    "03_xgboost",  "main"),
        ("LSTM",       "04_lstm",     "main"),
        ("Comparison", "05_compare",  "main"),
    ]

    for label, module_name, fn in steps:
        mod  = importlib.import_module(module_name)
        func = getattr(mod, fn)
        run_step(label, func)

    print("\n✅  All steps complete. Check the outputs/ folder for plots & results.")
