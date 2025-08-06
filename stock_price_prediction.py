import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
import tensorflow as tf

# =============================
# User-configurable parameters
# =============================
TICKER = "PLTR"  # Change this ticker symbol to download data for a different stock
LOOKBACK = 60     # Number of past days to use for prediction
EPOCHS = 50       # Number of epochs for training
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

# Use the 'Close' column for prediction
close_prices = df[['Close']]

# =====================
# Data Visualization
# =====================
plt.figure(figsize=(14, 5))
plt.plot(close_prices.index, close_prices['Close'], label='Close Price')
plt.title(f"Historical Closing Prices for {TICKER}")
plt.xlabel("Date")
plt.ylabel("Close Price USD ($)")
plt.legend()
plt.show()

# =====================
# Data Preprocessing
# =====================
# Normalize the dataset using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices.values)

# Determine training data length
training_data_len = int(np.ceil(len(scaled_data) * 0.8))

# Split the data into training and testing datasets
train_data = scaled_data[:training_data_len]
test_data = scaled_data[training_data_len - LOOKBACK:]

# Function to create sequences of data
def create_dataset(dataset, lookback):
    X, y = [], []
    for i in range(lookback, len(dataset)):
        X.append(dataset[i - lookback:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

# Generate sequences for training and testing
X_train, y_train = create_dataset(train_data, LOOKBACK)
X_test, y_test = create_dataset(test_data, LOOKBACK)

# =====================
# Reshape data for LSTM
# =====================
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# =====================
# Build the LSTM model
# =====================
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(LOOKBACK, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# =====================
# Train the model
# =====================
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VAL_SPLIT,
    verbose=1
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
predictions = scaler.inverse_transform(predictions)

# Inverse transform the actual prices
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# =====================
# Evaluate model performance
# =====================
rmse = np.sqrt(mean_squared_error(actual_prices, predictions))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# =====================
# Visualize results
# =====================
train = close_prices[:training_data_len]
valid = close_prices[training_data_len:]
valid['Predictions'] = predictions

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
model.save('stock_price_model.h5')

# ==========================================
# Optional: Predict future prices for 30 days
# ==========================================
future_predictions = []
last_sequence = scaled_data[-LOOKBACK:]
current_sequence = last_sequence.reshape(1, LOOKBACK, 1)

for _ in range(30):
    next_pred = model.predict(current_sequence)[0, 0]
    future_predictions.append(next_pred)
    # Append the prediction and slide the window
    last_sequence = np.append(last_sequence, next_pred)[-LOOKBACK:]
    current_sequence = last_sequence.reshape(1, LOOKBACK, 1)

# Inverse transform future predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Create dates for future predictions
last_date = close_prices.index[-1]
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=30)

# Plot historical and future prices
plt.figure(figsize=(14, 5))
plt.plot(close_prices.index, close_prices['Close'], label='Historical Prices')
plt.plot(future_dates, future_predictions, label='30-Day Forecast')
plt.title(f"{TICKER} Price Forecast")
plt.xlabel("Date")
plt.ylabel("Close Price USD ($)")
plt.legend()
plt.show()
