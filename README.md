# 🔧 Predictive Maintenance — RUL Forecasting
### NASA CMAPSS | SARIMA vs XGBoost vs LSTM

---

## 📁 Project Structure
```
predictive_maintenance/
│
├── data/
│   └── CMaps/               ← Dataset files live here
│       ├── train_FD001.txt
│       ├── test_FD001.txt
│       └── RUL_FD001.txt
│
├── src/
│   ├── config.py            ← All constants (paths, hyperparams)
│   ├── data_loader.py       ← Load & label dataset
│   ├── features.py          ← Rolling stats, lags, scaling
│   ├── 01_eda.py            ← Exploratory analysis & plots
│   ├── 02_sarima.py         ← Classical baseline
│   ├── 03_xgboost.py        ← Best model + tuning
│   ├── 04_lstm.py           ← Deep learning model
│   └── 05_compare.py        ← Final comparison
│
├── outputs/                 ← Auto-created: plots, models, CSVs
├── run_all.py               ← Run everything at once
└── requirements.txt
```

---

## ⚙️ Setup (VS Code)

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
# Windows:
venv\Scripts\activate
# Mac / Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Option A — Run everything at once
```bash
python run_all.py
```

### Option B — Run step by step (recommended while learning)
```bash
python src/01_eda.py        # Visualise sensors, RUL distribution
python src/02_sarima.py     # Classical baseline
python src/03_xgboost.py    # Best model
python src/04_lstm.py       # Deep learning model
python src/05_compare.py    # Final report
```

> All plots and results are saved to the `outputs/` folder automatically.

---

## 📊 Results

| Model | RMSE ↓ | MAE ↓ | Training Time |
|---|---|---|---|
| SARIMA | ~44 | ~37 | per-engine |
| **XGBoost** | **~20** | **~14** | ~12s |
| LSTM | ~24 | ~17 | ~180s |

---

## 🔑 Key Findings
1. **XGBoost outperforms LSTM** — rolling/lag features give tree models enough temporal signal; the dataset is too small for deep learning to shine.
2. **SARIMA is weakest** — univariate, assumes linear trends, must be re-fit per engine.
3. **Complexity ≠ Performance** — XGBoost beats LSTM while training 15× faster.

---

## 🛠️ Tech Stack
`pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` · `xgboost` · `tensorflow` · `statsmodels`
