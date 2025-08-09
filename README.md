# ML_Stock_Price_Prediction
Machine Learning model to predict stock prices using TensorFlow and LSTM.

Multi-task TensorFlow/Keras model for **stock price forecasting** that learns **(1) next‑day direction (up/down)** and **(2) a 90‑day price path** from OHLCV and technical indicators. The project is built end‑to‑end with **no data leakage**, **residual log‑return targets**, **attention pooling**, **horizon‑weighted loss**, and **walk‑forward evaluation**.

> ⚠️ **Disclaimer:** This repository is for education. It is **not** financial advice and is not a production trading system.

---

## 🧠 What this project does

- **Two heads (multi‑task):**
  - **Direction head:** binary classification of next‑day move (up/down).
  - **Forecast head:** multi‑step regression of the next **90 daily prices** (via log‑returns).
- **Features:** OHLCV + MA7/MA21/MA20, EMA12/EMA26, MACD & signal, Bollinger Bands, ATR(14), RSI(14), returns/lagged features, rolling volatility.
- **Targets:** log‑returns; optionally **residual log‑returns** = (log‑return − mean training log‑return) to reduce long‑horizon drift bias.
- **No‑leakage pipeline:** scalers fit on train only; validation/test transformed with the train scaler.
- **Attention pooling:** simple temporal attention over the last LSTM layer to learn which timesteps matter most.
- **Horizon‑weighted loss:** exponential decay so earlier forecast days matter more than far‑out days.
- **Evaluation:** holdout metrics **and** **walk‑forward** (time‑ordered CV) with **baselines** (Hold & Drift), plus optional **price‑space calibration** and **drift ensemble**.

---

## ⚙️ Configuration (top of `stock_price_prediction.py`)

Edit these constants to control behavior:

| Param | Default | What it does |
|---|---:|---|
| `TICKER` | `"PLTR"` | Which stock to model (Yahoo Finance symbol). |
| `LOOKBACK` | `60` | Past days in each input sequence. |
| `FORECAST_HORIZON` | `90` | Future days to forecast. |
| `BATCH_SIZE` | `32` | Mini‑batch size. |
| `EPOCHS` | `100` | Max training epochs (EarlyStopping usually stops earlier). |
| `VALIDATION_SPLIT` | `0.2` | Final 20% of time used as holdout validation. |
| `EARLY_STOP_PATIENCE` | `10` | EarlyStopping patience (epochs). |
| `DROPOUT_RATE` | `0.1` | Regularization between LSTM layers. |
| `LR` | `1e-3` | Initial learning rate (Adam). |
| `LOSS_WEIGHTS` | `{'direction': 1.0, 'forecast': 0.01}` | Balance classification vs. forecast loss. |
| `USE_CNN` | `True` | Prepend a `Conv1D` before LSTMs to capture local patterns. |
| `USE_BIDIR` | `False` | Use Bidirectional LSTM (off for pure forecasting by default). |
| `USE_ATTENTION` | `True` | Temporal attention pooling over final LSTM sequence. |
| `PREDICT_RETURNS` | `True` | Forecast log‑returns (converted to prices at evaluation). |
| `RESIDUAL_RETURNS` | `True` | Train on residuals: (log‑return − mean_train_logret); add back at inference. |
| `FORECAST_WEIGHT_GAMMA` | `0.02` | Exponential decay for horizon‑weighted MSE (↑ to focus more on near term). |
| `CALIBRATE_PREDICTIONS` | `True` | Linear calibration in price space on validation. |
| `ALPHA_ENSEMBLE_DRIFT` | `0.0` | Blend with drift baseline at inference (0–1). |
| `RUN_WALK_FORWARD` | `True` | Run time‑ordered cross‑validation after the holdout evaluation. |

> Tip: For tighter near‑term tracking, try `FORECAST_WEIGHT_GAMMA = 0.05–0.12`.

---

## 🛠️ Setup

**Python:** 3.10–3.13  
**Recommended:** fresh virtual environment.

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install deps
pip install --upgrade pip
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn yfinance
```

> If you have a `requirements.txt`, use `pip install -r requirements.txt`.

---

## 🚀 Run

```bash
python stock_price_prediction.py
```

What happens:
1. Downloads 5y OHLCV via `yfinance` and builds features.
2. Splits train/validation by time (**no leakage**).
3. Builds sequences (X, y_dir, y_fore) with **residual log‑returns** (if enabled).
4. Builds the model (Conv1D → LSTM(128→64→32) → Attention → two heads).
5. Trains with **EarlyStopping** and **ReduceLROnPlateau**.
6. Evaluates against **Hold** and **Drift** baselines; prints metrics:
   - Direction: **Accuracy** (+ confusion matrix)
   - Forecast: **MAE**, **RMSE**, **R²** (in **price space**)
7. (Optional) Runs **walk‑forward** evaluation and prints mean ± std across folds.
8. Saves the model to `stock_price_model.keras` and shows plots.

---

## 📏 Evaluation details

- **Baselines:**
  - **Hold**: last price repeated for all 90 days.
  - **Drift**: compound the mean training log‑return for 90 days.
- **Calibration:** Optional linear recalibration of price paths on validation (`true ≈ slope·pred + intercept`).
- **Walk‑forward:** TimeSeriesSplit; each fold trains on earlier data, predicts the next chunk.

---

## 🧪 Loading the saved model

```python
from tensorflow import keras
# Easiest: load without compile (custom loss handled on recompile)
model = keras.models.load_model("stock_price_model.keras", compile=False)

# Recompile using your helper (sets optimizer, weighted loss, metrics)
from stock_price_prediction import compile_model
model = compile_model(model)
```

## 🧩 Design choices & lessons learned

- **Predict returns, not prices** → reduces smooth bias and compounding issues.  
- **Residual returns** → de‑bias long‑horizon drift.  
- **Horizon‑weighted loss** → prioritize the next few weeks.  
- **Attention pooling** → focus on informative recent days.  
- **No‑leakage scaling** → fit scalers on train only.  
- **Walk‑forward** → honest, regime‑aware validation.  
- **Baselines first** → always beat Hold/Drift before celebrating.