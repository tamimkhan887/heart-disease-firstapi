import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("./heart.csv")

print("Dataset shape:", df.shape)
print(df.head())


features = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]

X = df[features]
y = df["target"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train
model.fit(X_train, y_train)


# Test
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)


# Create model directory
import os

os.makedirs("model", exist_ok=True)


# Save model
joblib.dump(model, "model/heart_model.joblib")

print("Model saved successfully!")