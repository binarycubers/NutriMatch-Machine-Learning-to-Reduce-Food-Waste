import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

engineered_dir = "data/engineered"
models_dir = "models"
forecast_dir = "data/forecast"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(forecast_dir, exist_ok=True)

nutrients = ["carbohydrates", "fiber", "protein", "fat"]
model_name = "lstm"
all_scores = []

def reshape_for_lstm(data):
    return np.expand_dims(data.values, axis=2)  # shape: (samples, timesteps, features)

for nutrient in nutrients:
    df = pd.read_csv(os.path.join(engineered_dir, f"{nutrient}_lagged.csv"))
    X = df.drop(["target", "date"], axis=1, errors="ignore")
    y = df["target"]

    # Split
    X_train, X_test = X[:-8], X[-8:]
    y_train, y_test = y[:-8], y[-8:]

    X_train_lstm = reshape_for_lstm(X_train)
    X_test_lstm = reshape_for_lstm(X_test)

    # Build model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train_lstm.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train
    model.fit(X_train_lstm, y_train, epochs=50, verbose=0, 
              callbacks=[EarlyStopping(patience=5)], validation_split=0.1)

    # Predictions
    train_preds = model.predict(X_train_lstm).flatten()
    test_preds = model.predict(X_test_lstm).flatten()

    # Metrics
    train_mse = mean_squared_error(y_train, train_preds)
    train_rmse = mean_squared_error(y_train, train_preds, squared=False)
    test_mse = mean_squared_error(y_test, test_preds)
    test_rmse = mean_squared_error(y_test, test_preds, squared=False)

    all_scores.append({
        "nutrient": nutrient,
        "train_mse": train_mse,
        "train_rmse": train_rmse,
        "test_mse": test_mse,
        "test_rmse": test_rmse
    })

    # Save model
    model_path = os.path.join(models_dir, f"{nutrient}_lstm.h5")
    model.save(model_path)
    print(f"✅ Saved model: {model_path}")

    # Save combined prediction CSV
    combined_df = pd.DataFrame({
        "date": df["date"],
        "actual": y,
        "predicted_train": list(train_preds) + [None]*8,
        "predicted_test": [None]*(len(y)-8) + list(test_preds)
    })
    forecast_path = os.path.join(forecast_dir, f"{nutrient}_lstm_forecast.csv")
    combined_df.to_csv(forecast_path, index=False)
    print(f"📊 Saved forecast: {forecast_path}")

# Save score file
score_df = pd.DataFrame(all_scores)
score_df.to_csv(os.path.join(forecast_dir, "lstm_model_scores.csv"), index=False)
print("📈 Saved LSTM score file.")