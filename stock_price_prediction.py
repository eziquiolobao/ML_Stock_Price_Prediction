import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dropout,
    Dense,
    BatchNormalization,
)
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# =============================
# USER-CONFIGURABLE PARAMETERS
# =============================
TICKER = "PLTR"
LOOKBACK = 60             # sequence length in days
FORECAST_HORIZON = 90     # days to forecast
BATCH_SIZE = 32
EPOCHS = 100
DROPOUT_RATE = 0.2
VALIDATION_SPLIT = 0.2
EARLY_STOP_PATIENCE = 10

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# =====================
# FEATURE ENGINEERING
# =====================
# Download 5 years of daily data for TICKER
df = yf.download(TICKER, period="5y")
df.to_csv(f"{TICKER}_historical_data.csv")
print(f"Saved historical data to {TICKER}_historical_data.csv")

# Keep only relevant columns
features = df[["Open", "High", "Low", "Close", "Volume"]].copy()

# Technical indicators
features["MA7"] = features["Close"].rolling(window=7).mean()
features["MA21"] = features["Close"].rolling(window=21).mean()
delta = features["Close"].diff()
up = delta.clip(lower=0)
down = -delta.clip(upper=0)
features["RSI"] = 100 - (100/(1 + up.rolling(14).mean()/down.rolling(14).mean()))

# Drop rows with NaNs after indicator calculation
features.dropna(inplace=True)

# =====================
# Data Visualization
# =====================
plt.figure(figsize=(14, 5))
plt.plot(features.index, features["Close"], label="Close Price")
plt.title(f"Historical Closing Prices for {TICKER}")
plt.xlabel("Date")
plt.ylabel("Close Price USD ($)")
plt.legend()
plt.show()

# =====================
# Data Preprocessing
# =====================
# Determine training data length
training_data_len = int(np.ceil(len(features) * 0.8))

# Split the data into training and testing datasets
train_features = features.iloc[:training_data_len]
test_features = features.iloc[training_data_len - LOOKBACK:]

# Fit the scaler on the entire dataset to avoid distribution shift issues
feature_scaler = MinMaxScaler()
feature_scaler.fit(features)
train_scaled = feature_scaler.transform(train_features)
test_scaled = feature_scaler.transform(test_features)

# Separate scaler for the 'Close' price for inverse transformation
price_scaler = MinMaxScaler()
price_scaler.fit(train_features[["Close"]])

# Function to create sequences of data
def create_dataset(dataset, lookback, target_index):
    X, y = [], []
    for i in range(lookback, len(dataset)):
        X.append(dataset[i - lookback : i])
        y.append(dataset[i, target_index])
    return np.array(X), np.array(y)

# Generate sequences for training and testing
close_idx = features.columns.get_loc("Close")
X_train, y_train = create_dataset(train_scaled, LOOKBACK, close_idx)
X_test, y_test = create_dataset(test_scaled, LOOKBACK, close_idx)

# Number of features for model input
num_features = X_train.shape[2]

# =====================

# =====================
# SEQUENCE GENERATION
# =====================
def create_sequences(data, lookback, horizon):
    X, y_dir, y_forecast = [], [], []
    close_idx = list(features.columns).index("Close")
    for i in range(lookback, len(data) - horizon):
        X.append(data[i - lookback:i])
        # Direction label: 1 if next day's close > today's close, else 0
        direction = 1 if data[i, close_idx] < data[i + 1, close_idx] else 0
        y_dir.append([direction])
        # Next 'horizon' days of normalized close prices
        y_forecast.append(data[i + 1 : i + 1 + horizon, close_idx])
    return (
        np.array(X),
        np.array(y_dir),
        np.array(y_forecast),
    )

# Create all sequences from the full normalized feature set
all_scaled = feature_scaler.transform(features)
X_all, y_dir_all, y_forecast_all = create_sequences(all_scaled, LOOKBACK, FORECAST_HORIZON)

# Split into train/val sets (80/20 split, matching earlier logic)
split_idx = int(len(X_all) * 0.8)
x_train, x_val = X_all[:split_idx], X_all[split_idx:]
y_dir_train, y_dir_val = y_dir_all[:split_idx], y_dir_all[split_idx:]
y_forecast_train, y_forecast_val = y_forecast_all[:split_idx], y_forecast_all[split_idx:]

# =====================
# Build the LSTM model (Functional API, multi-output)
# =====================
inp = Input(shape=(LOOKBACK, num_features))
x = LSTM(64, return_sequences=True)(inp)
x = BatchNormalization()(x)
x = Dropout(DROPOUT_RATE)(x)
x = LSTM(32, return_sequences=False)(x)
x = BatchNormalization()(x)
x = Dropout(DROPOUT_RATE)(x)

# Classification head: next-day direction (up/down)
dir_out = Dense(1, activation='sigmoid', name='direction')(x)

# Regression head: 90-day forecast
reg_out = Dense(FORECAST_HORIZON, activation='linear', name='forecast')(x)

model = Model(inputs=inp, outputs=[dir_out, reg_out])


# Compile the model with appropriate losses and metrics for each output
model.compile(
    optimizer='adam',
    loss={'direction': 'binary_crossentropy', 'forecast': 'mse'},
    metrics={'direction': 'accuracy', 'forecast': 'mae'}
)
model.summary()

# =====================
# Train the model
# =====================

early_stop = EarlyStopping(monitor="val_loss", patience=EARLY_STOP_PATIENCE, restore_best_weights=True)

history = model.fit(
    x_train,
    {'direction': y_dir_train, 'forecast': y_forecast_train},
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_val, {'direction': y_dir_val, 'forecast': y_forecast_val}),
    callbacks=[early_stop],
    verbose=1,
)
# Plot training and validation loss
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot total and per-output loss/metrics
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Total Training Loss')
plt.plot(history.history['val_loss'], label='Total Validation Loss')
if 'direction_loss' in history.history:
    plt.plot(history.history['direction_loss'], label='Direction Loss (Train)')
    plt.plot(history.history['val_direction_loss'], label='Direction Loss (Val)')
if 'forecast_loss' in history.history:
    plt.plot(history.history['forecast_loss'], label='Forecast Loss (Train)')
    plt.plot(history.history['val_forecast_loss'], label='Forecast Loss (Val)')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# =====================

# =====================
# Make predictions (multi-output)
# =====================
dir_pred, forecast_pred = model.predict(x_val)

# Inverse transform the forecast predictions and actuals for visualization
forecast_pred_inv = price_scaler.inverse_transform(forecast_pred)
y_forecast_val_inv = price_scaler.inverse_transform(y_forecast_val)

# =====================

# =====================
# Evaluate model performance (multi-output)
# =====================

# Direction classification metrics
from sklearn.metrics import accuracy_score
dir_pred_bin = (dir_pred > 0.5).astype(int)
dir_acc = accuracy_score(y_dir_val, dir_pred_bin)
print(f"Direction Accuracy: {dir_acc:.3f}")

# Forecast regression metrics (per horizon step, or average)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
rmse = np.sqrt(mean_squared_error(y_forecast_val_inv, forecast_pred_inv))
mae = mean_absolute_error(y_forecast_val_inv, forecast_pred_inv)
r2 = r2_score(y_forecast_val_inv, forecast_pred_inv)
print(f"Forecast RMSE: {rmse:.2f}")
print(f"Forecast MAE: {mae:.2f}")
print(f"Forecast R²: {r2:.2f}")

# =====================

# =====================
# Visualize results (multi-output)
# =====================
# Plot the first forecast horizon for a few samples
plt.figure(figsize=(14, 5))
for i in range(min(5, len(forecast_pred_inv))):
    plt.plot(range(FORECAST_HORIZON), y_forecast_val_inv[i], color='blue', alpha=0.3, label='Actual' if i==0 else "")
    plt.plot(range(FORECAST_HORIZON), forecast_pred_inv[i], color='red', alpha=0.3, label='Predicted' if i==0 else "")
plt.title(f"{TICKER} {FORECAST_HORIZON}-Day Forecast (Validation Examples)")
plt.xlabel("Forecast Day")
plt.ylabel("Close Price USD ($)")
plt.legend()
plt.show()

# =====================
# Save the model
# =====================
model.save('stock_price_model.keras')

