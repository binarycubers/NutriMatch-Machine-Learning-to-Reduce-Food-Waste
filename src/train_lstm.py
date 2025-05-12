import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import os

# Paths
engineered_dir = "data/engineered"
models_dir = "models"
forecast_dir = "data/forecast"
metrics_path = os.path.join(forecast_dir, "lstm_model_scores.csv")

os.makedirs(models_dir, exist_ok=True)
os.makedirs(forecast_dir, exist_ok=True)

nutrients = ["carbohydrates", "fiber", "protein", "fat"]

# Load and preprocess data
def load_data(nutrient):
    df = pd.read_csv(os.path.join(engineered_dir, f"{nutrient}_lagged.csv"))
    return df

def split_data(df):
    X = df.drop(columns=["target", "date"], errors='ignore')
    y = df["target"]
    return X[:-8], X[-8:], y[:-8], y[-8:]

def reshape_for_lstm(X):
    return np.reshape(X.values, (X.shape[0], X.shape[1], 1))

# Collect results
results = []

for nutrient in nutrients:
    df = load_data(nutrient)
    dates = df["date"].values[-8:] if "date" in df.columns else np.arange(1, 9)

    X_train, X_test, y_train, y_test = split_data(df)
    X_train_lstm = reshape_for_lstm(X_train)
    X_test_lstm = reshape_for_lstm(X_test)

    # Define LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train_lstm.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Fit model
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train_lstm, y_train, validation_split=0.2, epochs=100, batch_size=4, callbacks=[es], verbose=0)

    # Predict
    preds = model.predict(X_test_lstm).flatten()
    rmse = mean_squared_error(y_test, preds, squared=False)
    mse = mean_squared_error(y_test, preds)

    print(f"{nutrient} LSTM RMSE: {rmse:.2f}")

    # Save model
    model_path = os.path.join(models_dir, f"{nutrient}_lstm.h5")
    model.save(model_path)
    print(f"✅ Saved model: {model_path}")

    # Save forecast
    forecast_df = pd.DataFrame({
        "date": dates,
        "actual": y_test.values,
        "predicted": preds
    })
    forecast_csv_path = os.path.join(forecast_dir, f"{nutrient}_lstm_forecast.csv")
    forecast_df.to_csv(forecast_csv_path, index=False)
    print(f"📊 Saved forecast CSV: {forecast_csv_path}")

    # Store evaluation result
    results.append({
        "nutrient": nutrient,
        "model": "LSTM",
        "mse": round(mse, 4),
        "rmse": round(rmse, 4)
    })

# Save all metrics
results_df = pd.DataFrame(results)
results_df.to_csv(metrics_path, index=False)
print(f"📈 Saved model evaluation scores: {metrics_path}")
