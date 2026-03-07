# ⚙️ Predictive Maintenance — RUL Forecasting
### NASA CMAPSS Turbofan Engine Dataset

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square)

---

## 📌 Problem Statement

Predict the **Remaining Useful Life (RUL)** of aircraft turbofan engines from multivariate sensor time-series data. Early and accurate RUL prediction enables proactive, cost-effective maintenance — reducing unplanned downtime and safety risks in aerospace and industrial systems.

---

## 📦 Dataset

**NASA CMAPSS — Commercial Modular Aero-Propulsion System Simulation**

| Property | Details |
|---|---|
| Source | [Kaggle — behrad3d/nasa-cmaps](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) |
| Engines | 100 training, 100 test |
| Sensors | 21 sensor channels + 3 operational settings |
| Subsets | FD001–FD004 (this project uses FD001) |
| Target | Remaining Useful Life in engine cycles |

---

## 🗂️ Project Structure

```
predictive_maintenance/
│
├── data/
│   └── CMaps/                    ← Dataset files
│       ├── train_FD001.txt
│       ├── test_FD001.txt
│       └── RUL_FD001.txt
│
├── src/
│   ├── config.py                 ← All constants & hyperparameters
│   ├── data_loader.py            ← Load & label dataset
│   ├── features.py               ← Feature engineering pipeline
│   ├── 01_eda.py                 ← Exploratory analysis
│   ├── 02_sarima.py              ← Classical baseline
│   ├── 03_xgboost.py             ← Best performing model
│   ├── 04_lstm.py                ← Deep learning model
│   └── 05_compare.py             ← Final model comparison
│
├── outputs/                      ← Auto-generated plots, models, CSVs
├── app.py                        ← Streamlit dashboard
├── run_all.py                    ← Run full pipeline at once
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/predictive-maintenance.git
cd predictive-maintenance

# 2. Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# Mac / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Option A — Streamlit Dashboard (recommended)
```bash
streamlit run app.py
```
Opens at `http://localhost:8501` with 5 interactive pages.

### Option B — Individual scripts
```bash
python src/01_eda.py          # Sensor analysis & visualisation
python src/02_sarima.py       # Classical time-series baseline
python src/03_xgboost.py      # Best model + hyperparameter tuning
python src/04_lstm.py         # Deep learning model
python src/05_compare.py      # Final comparison report
```

### Option C — Full pipeline
```bash
python run_all.py
```

> All plots and results are saved automatically to the `outputs/` folder.

---

## 🧠 Methodology

### Feature Engineering
- Dropped **7 constant sensors** (near-zero variance: s1, s5, s6, s10, s16, s18, s19)
- Created **rolling mean & std** over windows of 5, 10, and 30 cycles per sensor
- Added **lag features** (t−1, t−2, t−3) to capture short-term temporal trends
- Clipped RUL at **125 cycles** (standard piecewise-linear target for CMAPSS)
- Applied **MinMaxScaler** fitted on training data only (no data leakage)

### Models

| Model | Input Strategy | Approach |
|---|---|---|
| **SARIMA** | Single sensor per engine | Univariate time-series forecasting |
| **XGBoost** | All rolling + lag features | Multivariate regression, GridSearchCV tuning |
| **LSTM** | 30-cycle sliding window of raw sensors | Sequence-to-scalar deep learning |

---

## 📊 Results

| Model | RMSE ↓ | MAE ↓ | Training Time | Complexity |
|---|---|---|---|---|
| SARIMA | ~44.2 | ~36.8 | Per-engine | Low |
| **XGBoost** | **~19.8** | **~14.3** | **~12s** | Medium |
| LSTM | ~23.5 | ~17.1 | ~180s | High |

> ✅ **XGBoost achieved the best accuracy with the fastest training time.**

---

## 🔑 Key Findings

**1. XGBoost outperforms LSTM**
Rolling and lag features explicitly encode temporal degradation patterns, giving tree-based models sufficient signal. The CMAPSS dataset (~20K rows) is also too small for deep learning to demonstrate a clear advantage.

**2. SARIMA has fundamental limitations here**
It is univariate (misses all cross-sensor interactions), assumes linear trends that don't match accelerating degradation near failure, and must be re-fit per engine — making it unscalable.

**3. Complexity ≠ Performance**
XGBoost beat the LSTM by ~15% RMSE while training 15× faster. This reinforces the importance of evaluating production constraints (latency, compute cost) alongside accuracy.

**4. Feature Engineering was the biggest driver**
Rolling mean features were consistently the top-ranked features in XGBoost's importance chart — more impactful than model architecture choices.

---

## ⚠️ Early Warning System

The XGBoost model powers a threshold-based alert system. Engines with a predicted RUL ≤ 30 cycles trigger a **maintenance alert**, enabling proactive scheduling before failure occurs.

---

## 🖥️ Streamlit Dashboard

The interactive dashboard has 5 pages:

| Page | Description |
|---|---|
| 🏠 Overview | Dataset statistics and RUL distribution |
| 📊 EDA | Interactive sensor variance, degradation curves, and correlation heatmap |
| 🤖 Train Models | Train SARIMA / XGBoost / LSTM directly in the browser with live progress |
| 🔮 Predict RUL | Upload engine sensor CSV → instant RUL prediction + health score |
| 🏆 Compare Models | Side-by-side RMSE/MAE charts, trade-off plot, and key findings |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Data Processing | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| Classical ML | `statsmodels` (SARIMA), `scikit-learn` |
| Gradient Boosting | `xgboost` |
| Deep Learning | `tensorflow` / `keras` |
| Dashboard | `streamlit` |

---

## 🚀 Future Work

- Extend to FD002–FD004 subsets (multiple operating conditions and fault modes)
- Experiment with Transformer-based architectures for sequence modelling
- Add anomaly detection layer for unseen fault signatures
- Deploy dashboard to Streamlit Cloud for a public-facing demo

---

## 📄 License

MIT License — feel free to use and adapt this project.
