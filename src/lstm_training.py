import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
# Directory setup
engineered_dir = "data/engineered"
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)
nutrients = ["carbohydrates", "fiber", "protein", "fat"]
for nutrient in nutrients:
    print(f"\n:arrows_counterclockwise: Training LSTM for {nutrient}...")
    # Load lagged data
    path = os.path.join(engineered_dir, f"{nutrient}_lagged.csv")
    if not os.path.exists(path):
        print(f":warning:  File not found: {path}")
        continue
    df = pd.read_csv(path)
    X = df.drop("target", axis=1).values
    y = df["target"].values
    # Reshape for LSTM: (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    # Split into train/test
    X_train, X_test = X[:-8], X[-8:]
    y_train, y_test = y[:-8], y[-8:]
    # Define LSTM model
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(X.shape[1], 1)))
    model.add(Dense(1))

    from tensorflow.keras.losses import MeanSquaredError
    model.compile(optimizer='adam', loss=MeanSquaredError())
    # Train model
    model.fit(X_train, y_train, epochs=50, verbose=1,
              callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
    # Evaluate
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f":white_check_mark: RMSE for {nutrient}: {rmse:.2f}")
    # Save model
    model.save(os.path.join(models_dir, f"{nutrient}_lstm_model.h5"))
    print(f":floppy_disk: Saved model: {nutrient}_lstm_model.h5")