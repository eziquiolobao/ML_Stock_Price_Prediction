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
    Bidirectional,
    BatchNormalization,
)
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# =============================
# User-configurable parameters
# =============================
TICKER = "PLTR"  # Change this ticker symbol to download data for a different stock
LOOKBACK = 60     # Number of past days to look back (try 30, 90, 120, etc.)
EPOCHS = 100      # Training epochs (can increase to 200 for more training)
BATCH_SIZE = 32   # Batch size for training
VAL_SPLIT = 0.1   # Fraction of training data to use for validation

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# =====================
# Download stock data
# =====================
# Fetch historical stock data for the last 5 years using yfinance
df = yf.download(TICKER, period="5y")
df.to_csv(f"{TICKER}_historical_data.csv")
print(f"Saved historical data to {TICKER}_historical_data.csv")

# =====================
# Feature Engineering
# =====================
# Use relevant price columns and add technical indicators
features = df[["Open", "High", "Low", "Close", "Volume"]].copy()

# Moving averages
features["MA7"] = features["Close"].rolling(window=7).mean()
features["MA21"] = features["Close"].rolling(window=21).mean()

# Relative Strength Index (RSI)
delta = features["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
features["RSI"] = 100 - (100 / (1 + rs))

# Drop rows with NaN values created by indicators
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

# Scale features using MinMaxScaler fitted on training data only

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
# Build the Bidirectional LSTM model
# =====================
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(LOOKBACK, num_features)),
    BatchNormalization(),
    Dropout(0.1),
    Bidirectional(LSTM(64, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.1),
    Bidirectional(LSTM(32, return_sequences=False)),
    Dense(25),
    Dense(1),
])

model.compile(optimizer="adam", loss="mean_squared_error")

# =====================
# Train the model
# =====================
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

history = model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VAL_SPLIT,
    callbacks=[early_stop],
    verbose=1,
)

# Plot training and validation loss
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# =====================
# Make predictions
# =====================
predictions = model.predict(X_test)
predictions = price_scaler.inverse_transform(predictions)

# Inverse transform the actual prices
actual_prices = price_scaler.inverse_transform(y_test.reshape(-1, 1))

# =====================
# Evaluate model performance
# =====================

# LSTM model metrics
rmse = np.sqrt(mean_squared_error(actual_prices, predictions))
mae = mean_absolute_error(actual_prices, predictions)
r2 = r2_score(actual_prices, predictions)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Naive baseline: last value prediction
naive_preds = valid['Close'].shift(1).dropna().values
naive_actuals = valid['Close'].iloc[1:].values
naive_rmse = np.sqrt(mean_squared_error(naive_actuals, naive_preds))
naive_mae = mean_absolute_error(naive_actuals, naive_preds)
naive_r2 = r2_score(naive_actuals, naive_preds)
print("\nNaive Baseline (Last Value) Results:")
print(f"RMSE: {naive_rmse:.2f}")
print(f"MAE: {naive_mae:.2f}")
print(f"R² Score: {naive_r2:.2f}")

# =====================
# Visualize results
# =====================

train = features[["Close"]][:training_data_len]
valid = features[["Close"]][training_data_len:].copy()
valid["Predictions"] = predictions

# Naive baseline: last value prediction
naive_preds = valid['Close'].shift(1).dropna().values
naive_actuals = valid['Close'].iloc[1:].values
naive_rmse = np.sqrt(mean_squared_error(naive_actuals, naive_preds))
naive_mae = mean_absolute_error(naive_actuals, naive_preds)
naive_r2 = r2_score(naive_actuals, naive_preds)
print("\nNaive Baseline (Last Value) Results:")
print(f"RMSE: {naive_rmse:.2f}")
print(f"MAE: {naive_mae:.2f}")
print(f"R² Score: {naive_r2:.2f}")

plt.figure(figsize=(14, 5))
plt.plot(train.index, train['Close'], label='Training Data')
plt.plot(valid.index, valid['Close'], label='Actual Prices')
plt.plot(valid.index, valid['Predictions'], label='Predicted Prices')
plt.title(f"{TICKER} Price Prediction")
plt.xlabel("Date")
plt.ylabel("Close Price USD ($)")
plt.legend()
plt.show()

# =====================
# Save the model
# =====================
model.save('stock_price_model.keras')

