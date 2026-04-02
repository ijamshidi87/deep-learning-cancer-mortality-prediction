"""
Linear Regression Model Training and Testing
--------------------------------------------
This script trains a Linear Regression model on cancer_reg.csv,
saves the model as linear_regression.pkl, and includes a test_model
function to load the trained model and run predictions.
"""

import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -------------------------
# Training Function
# -------------------------
def train_and_save(csv_path="cancer_reg.csv", model_path="linear_regression.pkl"):
    # Load dataset
    data = pd.read_csv(csv_path, encoding="latin1")

    # Fill missing values with column means
    for col in data.columns[:34]:
        if data[col].isna().any():
            data[col] = data[col].fillna(data[col].mean())

    # Encode categorical columns
    encoder = LabelEncoder()
    data["Geography"] = encoder.fit_transform(data["Geography"])
    data["binnedInc"] = encoder.fit_transform(data["binnedInc"])

    # Features and target
    X = data.drop(columns=["TARGET_deathRate"])
    y = data["TARGET_deathRate"]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("Test MSE:", mean_squared_error(y_test, y_pred))
    print("Test R²:", r2_score(y_test, y_pred))

    # Save model
    joblib.dump(model, model_path)
    print(f"Model saved as {model_path}")


# -------------------------
# Test Function
# -------------------------
def test_model(model_path="linear_regression.pkl", csv_path="cancer_reg.csv"):
    # Load model
    model = joblib.load(model_path)

    # Load dataset
    data = pd.read_csv(csv_path, encoding="latin1")
    for col in data.columns[:34]:
        if data[col].isna().any():
            data[col] = data[col].fillna(data[col].mean())

    encoder = LabelEncoder()
    data["Geography"] = encoder.fit_transform(data["Geography"])
    data["binnedInc"] = encoder.fit_transform(data["binnedInc"])

    X = data.drop(columns=["TARGET_deathRate"])
    y = data["TARGET_deathRate"]

    predictions = model.predict(X)
    print("Sample predictions:", predictions[:10])
    return predictions


if __name__ == "__main__":
    train_and_save()
    test_model()
