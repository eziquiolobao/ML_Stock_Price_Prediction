# ===============================================================
# 1) Imports & Global Settings
# ===============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
)
from sklearn.model_selection import TimeSeriesSplit

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    LSTM,
    Dropout,
    Dense,
    BatchNormalization,
    Conv1D,
    Bidirectional,
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# ===============================================================
# 2) User Config
# ===============================================================
TICKER = "PLTR"
LOOKBACK = 60
FORECAST_HORIZON = 90
BATCH_SIZE = 32
EPOCHS = 100
VALIDATION_SPLIT = 0.2  # used for early stopping only
EARLY_STOP_PATIENCE = 10

# New options
USE_CNN = True
USE_BIDIR = False
PREDICT_RETURNS = True
FORECAST_WEIGHT_GAMMA = 0.02
DROPOUT_RATE = 0.1
LR = 1e-3
LOSS_WEIGHTS = {"direction": 1.0, "forecast": 0.01}
RUN_WALK_FORWARD = False  # optional walk-forward evaluation


# ===============================================================
# 3) Download & Feature Engineering
# ===============================================================
def download_and_engineer(ticker: str) -> pd.DataFrame:
    """Download historical data and compute technical indicators."""

    df = yf.download(ticker, period="5y")
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    # Moving Averages
    df["MA7"] = df["Close"].rolling(7).mean()
    df["MA21"] = df["Close"].rolling(21).mean()

    # MACD
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands (ensure Series, avoid DataFrame assignment)
    roll = df["Close"].rolling(window=20, min_periods=20)
    ma20 = roll.mean()
    std20 = roll.std()
    # If yfinance returns a MultiIndex frame and .rolling() yields a DataFrame, squeeze to Series
    if isinstance(ma20, pd.DataFrame):
        ma20 = ma20.iloc[:, 0]
    if isinstance(std20, pd.DataFrame):
        std20 = std20.iloc[:, 0]
    df["MA20"] = ma20.to_numpy()
    df["BB_UPPER"] = (ma20 + 2 * std20).to_numpy()
    df["BB_LOWER"] = (ma20 - 2 * std20).to_numpy()

    # ATR
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            (df["High"] - df["Low"]).abs(),
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    # Returns, lags and volatility
    df["RET1"] = df["Close"].pct_change()
    df["LOGRET1"] = np.log1p(df["RET1"])
    df["RET5"] = df["Close"].pct_change(5)
    df["VOLAT_21"] = df["LOGRET1"].rolling(21).std()
    df["CLOSE_LAG1"] = df["Close"].shift(1)
    df["CLOSE_LAG5"] = df["Close"].shift(5)

    # RSI
    delta = df["Close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / down
    df["RSI"] = 100 - (100 / (1 + rs))

    df.dropna(inplace=True)
    df.to_csv(f"{ticker}_features.csv")
    print(f"Saved feature dataset to {ticker}_features.csv")
    return df


feature_cols = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "MA7",
    "MA21",
    "EMA12",
    "EMA26",
    "MACD",
    "MACD_SIGNAL",
    "MA20",
    "BB_UPPER",
    "BB_LOWER",
    "ATR14",
    "RET1",
    "LOGRET1",
    "RET5",
    "VOLAT_21",
    "CLOSE_LAG1",
    "CLOSE_LAG5",
    "RSI",
]


# ===============================================================
# 4) Train/Val Split & Scaling (No Leakage)
# ===============================================================
def prepare_data(df: pd.DataFrame):
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx - LOOKBACK :]

    feature_scaler = MinMaxScaler().fit(train_df[feature_cols])
    price_scaler = MinMaxScaler().fit(train_df[["Close"]])

    train_scaled = feature_scaler.transform(train_df[feature_cols])
    val_scaled = feature_scaler.transform(val_df[feature_cols])

    return (
        train_scaled,
        val_scaled,
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        price_scaler,
    )


# ===============================================================
# 5) Sequence Builder
# ===============================================================
def create_sequences(
    scaled_features: np.ndarray,
    raw_df: pd.DataFrame,
    lookback: int,
    horizon: int,
    predict_returns: bool,
    price_scaler: MinMaxScaler,
):
    """Create sequences for model training/validation."""

    X, y_dir, y_fore = [], [], []
    for i in range(lookback, len(scaled_features) - horizon):
        X.append(scaled_features[i - lookback : i])

        next_logret = raw_df["LOGRET1"].iloc[i]
        y_dir.append([1 if next_logret > 0 else 0])

        if predict_returns:
            y_fore.append(raw_df["LOGRET1"].iloc[i : i + horizon].to_numpy())
        else:
            scaled_prices = price_scaler.transform(raw_df[["Close"]].iloc[i : i + horizon])
            y_fore.append(scaled_prices.flatten())

    return np.array(X), np.array(y_dir), np.array(y_fore)


def build_datasets(df: pd.DataFrame):
    train_scaled, val_scaled, train_df, val_df, price_scaler = prepare_data(df)

    X_train, y_dir_train, y_fore_train = create_sequences(
        train_scaled, train_df, LOOKBACK, FORECAST_HORIZON, PREDICT_RETURNS, price_scaler
    )
    X_val, y_dir_val, y_fore_val = create_sequences(
        val_scaled, val_df, LOOKBACK, FORECAST_HORIZON, PREDICT_RETURNS, price_scaler
    )

    return (
        X_train,
        y_dir_train,
        y_fore_train,
        X_val,
        y_dir_val,
        y_fore_val,
        price_scaler,
        val_df,
    )


# ===============================================================
# 6) Model: CNN-LSTM Backbone
# ===============================================================
def build_model(n_features: int) -> Model:
    inp = Input(shape=(LOOKBACK, n_features))
    x = inp

    if USE_CNN:
        x = Conv1D(filters=32, kernel_size=3, padding="causal", activation="relu")(x)

    if USE_BIDIR:
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
    else:
        x = LSTM(128, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)

    x = LSTM(64, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)

    x = LSTM(32, return_sequences=False)(x)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)

    dir_out = Dense(1, activation="sigmoid", name="direction")(x)
    reg_out = Dense(FORECAST_HORIZON, activation="linear", name="forecast")(x)

    model = Model(inp, [dir_out, reg_out])
    return model


# ===============================================================
# 7) Weighted Loss for Forecast Head
# ===============================================================
def weighted_mse(horizon: int, gamma: float):
    w = K.exp(-gamma * K.arange(0, horizon, dtype="float32"))
    w = w / K.sum(w)

    def loss(y_true, y_pred):
        se = K.square(y_true - y_pred)
        return K.sum(se * w, axis=1)

    return loss


# ===============================================================
# 8) Compile Model
# ===============================================================
def compile_model(model: Model) -> Model:
    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss={
            "direction": "binary_crossentropy",
            "forecast": weighted_mse(FORECAST_HORIZON, FORECAST_WEIGHT_GAMMA),
        },
        loss_weights=LOSS_WEIGHTS,
        metrics={"direction": "accuracy", "forecast": "mae"},
    )
    return model


# ===============================================================
# 9) Training
# ===============================================================
def train_model(model: Model, X_train, y_dir_train, y_fore_train, X_val, y_dir_val, y_fore_val):
    early_stop = EarlyStopping(
        monitor="val_loss", patience=EARLY_STOP_PATIENCE, restore_best_weights=True
    )
    history = model.fit(
        X_train,
        {"direction": y_dir_train, "forecast": y_fore_train},
        validation_data=(X_val, {"direction": y_dir_val, "forecast": y_fore_val}),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1,
    )
    return history


# ===============================================================
# 10) Post-processing Utilities
# ===============================================================
def logrets_to_prices(start_price: float, logrets: np.ndarray) -> np.ndarray:
    """Convert log-returns to a price path.
    Forces start_price to a scalar float and logrets to a 1-D NumPy array
    to avoid pandas alignment/shape issues.
    """
    # Coerce start_price to a plain scalar float (handles 0-d arrays / 1-len Series)
    start_price = float(np.asarray(start_price).reshape(-1)[0])
    # Ensure 1-D numpy array of floats
    logrets = np.asarray(logrets, dtype=float).reshape(-1)
    cumulative = np.exp(np.cumsum(logrets))
    return start_price * cumulative


# ===============================================================
# 11) Evaluation & Baselines
# ===============================================================
def evaluate(
    model: Model,
    X_val,
    y_dir_val,
    y_fore_val,
    price_scaler: MinMaxScaler,
    val_df: pd.DataFrame,
):
    dir_pred, fore_pred = model.predict(X_val, verbose=0)

    # Direction metrics
    dir_bin = (dir_pred > 0.5).astype(int)
    dir_acc = accuracy_score(y_dir_val, dir_bin)
    print(f"Direction Accuracy: {dir_acc:.3f}")

    # Forecast metrics in price space
    pred_prices, true_prices = [], []
    for i in range(len(fore_pred)):
        start_price = val_df["Close"].iloc[i + LOOKBACK - 1]
        if PREDICT_RETURNS:
            pred_path = logrets_to_prices(start_price, fore_pred[i])
            true_path = logrets_to_prices(start_price, y_fore_val[i])
        else:
            pred_path = price_scaler.inverse_transform(fore_pred[i].reshape(-1, 1)).flatten()
            true_path = price_scaler.inverse_transform(y_fore_val[i].reshape(-1, 1)).flatten()
        pred_prices.append(pred_path)
        true_prices.append(true_path)

    pred_prices = np.array(pred_prices)
    true_prices = np.array(true_prices)

    rmse = np.sqrt(mean_squared_error(true_prices.flatten(), pred_prices.flatten()))
    mae = mean_absolute_error(true_prices.flatten(), pred_prices.flatten())
    r2 = r2_score(true_prices.flatten(), pred_prices.flatten())
    print(f"Forecast RMSE: {rmse:.2f}")
    print(f"Forecast MAE: {mae:.2f}")
    print(f"Forecast R2: {r2:.2f}")

    # Baselines (align lengths with model sequences)
    n_seq = len(fore_pred)  # number of validation sequences predicted by the model
    start_idx = LOOKBACK - 1
    last_close = val_df["Close"].iloc[start_idx : start_idx + n_seq].to_numpy()
    baseline_hold = np.repeat(last_close[:, None], FORECAST_HORIZON, axis=1)

    # Mean log-return estimated from the same period covered by last_close anchors
    mean_logret = val_df["LOGRET1"].iloc[: start_idx + n_seq].mean()
    baseline_drift = np.array([
        logrets_to_prices(price, np.full(FORECAST_HORIZON, mean_logret))
        for price in last_close
    ])

    for name, base in {
        "Hold": baseline_hold,
        "Drift": baseline_drift,
    }.items():
        b_rmse = np.sqrt(mean_squared_error(true_prices.flatten(), base.flatten()))
        b_mae = mean_absolute_error(true_prices.flatten(), base.flatten())
        print(f"{name} Baseline RMSE: {b_rmse:.2f} | MAE: {b_mae:.2f}")

    # Confusion matrix
    cm = confusion_matrix(y_dir_val, dir_bin)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Direction Confusion Matrix")
    plt.show()

    return pred_prices, true_prices


# ===============================================================
# 12) Visualisation Helpers
# ===============================================================
def plot_history(history):
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    if "direction_loss" in history.history:
        plt.plot(history.history["direction_loss"], label="Dir Loss")
        plt.plot(history.history["val_direction_loss"], label="Val Dir Loss")
    if "forecast_loss" in history.history:
        plt.plot(history.history["forecast_loss"], label="Fore Loss")
        plt.plot(history.history["val_forecast_loss"], label="Val Fore Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training History")
    plt.show()


def plot_predictions(pred_prices, true_prices):
    plt.figure(figsize=(14, 5))
    for i in range(min(5, len(pred_prices))):
        plt.plot(true_prices[i], color="blue", alpha=0.3, label="Actual" if i == 0 else "")
        plt.plot(pred_prices[i], color="red", alpha=0.3, label="Predicted" if i == 0 else "")
    plt.title(f"{TICKER} {FORECAST_HORIZON}-Day Forecast (Validation Samples)")
    plt.xlabel("Day Ahead")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


# ===============================================================
# 13) Optional Walk-Forward Evaluation
# ===============================================================
def walk_forward_eval(df: pd.DataFrame, n_splits: int = 3):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    dir_scores, mae_scores, rmse_scores = [], [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx[0] - LOOKBACK : test_idx[-1] + 1]

        train_scaled = MinMaxScaler().fit_transform(train_df[feature_cols])
        price_scaler = MinMaxScaler().fit(train_df[["Close"]])
        test_scaled = MinMaxScaler().fit(train_df[feature_cols]).transform(
            test_df[feature_cols]
        )

        X_tr, y_dir_tr, y_fore_tr = create_sequences(
            train_scaled, train_df.reset_index(drop=True), LOOKBACK, FORECAST_HORIZON, PREDICT_RETURNS, price_scaler
        )
        X_te, y_dir_te, y_fore_te = create_sequences(
            test_scaled, test_df.reset_index(drop=True), LOOKBACK, FORECAST_HORIZON, PREDICT_RETURNS, price_scaler
        )

        model = build_model(X_tr.shape[2])
        model = compile_model(model)
        history = model.fit(
            X_tr,
            {"direction": y_dir_tr, "forecast": y_fore_tr},
            epochs=min(50, EPOCHS),
            batch_size=BATCH_SIZE,
            verbose=0,
        )

        dir_pred, fore_pred = model.predict(X_te, verbose=0)
        dir_bin = (dir_pred > 0.5).astype(int)
        dir_scores.append(accuracy_score(y_dir_te, dir_bin))

        preds, trues = [], []
        for i in range(len(fore_pred)):
            start_price = test_df["Close"].iloc[i + LOOKBACK - 1]
            if PREDICT_RETURNS:
                preds.append(logrets_to_prices(start_price, fore_pred[i]))
                trues.append(logrets_to_prices(start_price, y_fore_te[i]))
            else:
                preds.append(
                    price_scaler.inverse_transform(fore_pred[i].reshape(-1, 1)).flatten()
                )
                trues.append(
                    price_scaler.inverse_transform(y_fore_te[i].reshape(-1, 1)).flatten()
                )

        preds = np.array(preds)
        trues = np.array(trues)
        mae_scores.append(mean_absolute_error(trues.flatten(), preds.flatten()))
        rmse_scores.append(
            np.sqrt(mean_squared_error(trues.flatten(), preds.flatten()))
        )

    print(
        f"Walk-forward Direction Acc: {np.mean(dir_scores):.3f} ± {np.std(dir_scores):.3f}"
    )
    print(
        f"Walk-forward Forecast MAE: {np.mean(mae_scores):.2f} ± {np.std(mae_scores):.2f}"
    )
    print(
        f"Walk-forward Forecast RMSE: {np.mean(rmse_scores):.2f} ± {np.std(rmse_scores):.2f}"
    )


# ===============================================================
# 14) Main Execution
# ===============================================================
def main():
    df = download_and_engineer(TICKER)
    X_train, y_dir_train, y_fore_train, X_val, y_dir_val, y_fore_val, price_scaler, val_df = build_datasets(df)

    model = build_model(X_train.shape[2])
    model = compile_model(model)
    history = train_model(
        model, X_train, y_dir_train, y_fore_train, X_val, y_dir_val, y_fore_val
    )

    pred_prices, true_prices = evaluate(
        model, X_val, y_dir_val, y_fore_val, price_scaler, val_df
    )

    plot_history(history)
    plot_predictions(pred_prices, true_prices)

    model.save("stock_price_model.keras")
    print("Model saved to stock_price_model.keras")

    if RUN_WALK_FORWARD:
        walk_forward_eval(df)


if __name__ == "__main__":
    main()

