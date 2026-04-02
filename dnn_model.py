"""
Deep Neural Network (DNN) Model Training and Testing
----------------------------------------------------
This script trains a DNN on cancer_reg.csv,
saves the model as dnn_model.h5, and includes a test_model
function to load the trained model and run predictions.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------
# Training Function
# -------------------------
def train_and_save(csv_path="cancer_reg.csv", model_path="dnn_model.h5"):
    # Load dataset
    data = pd.read_csv(csv_path, encoding="latin1")

    # Fill missing values
    for col in data.columns[:34]:
        if data[col].isna().any():
            data[col] = data[col].fillna(data[col].mean())

    # Encode categorical
    encoder = LabelEncoder()
    data["Geography"] = encoder.fit_transform(data["Geography"])
    data["binnedInc"] = encoder.fit_transform(data["binnedInc"])

    X = data.drop(columns=["TARGET_deathRate"])
    y = data["TARGET_deathRate"]

    # Log-transform for stability
    X = np.log1p(X)
    y = np.log1p(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build model
    model = Sequential([
        Dense(64, activation="relu", input_dim=X_train.shape[1]),
        Dropout(0.1),
        Dense(32, activation="relu"),
        Dropout(0.1),
        Dense(16, activation="relu"),
        Dense(1, activation="linear")
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

    # Train
    early_stop = EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
    model.fit(X_train, y_train,
              validation_split=0.2,
              epochs=200,
              batch_size=32,
              verbose=1,
              callbacks=[early_stop])

    # Evaluate
    y_pred = model.predict(X_test)
    print("Test MSE:", mean_squared_error(y_test, y_pred))
    print("Test R²:", r2_score(y_test, y_pred))

    # Save model
    model.save(model_path)
    print(f"Model saved as {model_path}")


# -------------------------
# Test Function
# -------------------------
def test_model(model_path="dnn_model.h5", csv_path="cancer_reg.csv"):
    model = load_model(model_path)

    data = pd.read_csv(csv_path, encoding="latin1")
    for col in data.columns[:34]:
        if data[col].isna().any():
            data[col] = data[col].fillna(data[col].mean())

    encoder = LabelEncoder()
    data["Geography"] = encoder.fit_transform(data["Geography"])
    data["binnedInc"] = encoder.fit_transform(data["binnedInc"])

    X = data.drop(columns=["TARGET_deathRate"])
    X = np.log1p(X)
    X = StandardScaler().fit_transform(X)

    predictions = model.predict(X)
    print("Sample predictions:", predictions[:10].flatten())
    return predictions


if __name__ == "__main__":
    train_and_save()
    test_model()
