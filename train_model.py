"""
Air Quality Prediction — Model Training Script
Trains the regression (PM2.5) and classification (AQI category) models
on data/air_quality_dataset.csv and saves them into models/ for the
Flask API (app.py) to load at startup.

Run this once before starting the server:
    python train_model.py
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
)
from xgboost import XGBRegressor, XGBClassifier

DATA_PATH = "data/air_quality_dataset.csv"
MODELS_DIR = "models"

print("Loading dataset...")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df["month"] = df["date"].dt.month
df["dayofweek"] = df["date"].dt.dayofweek
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

le_loc = LabelEncoder()
df["location_enc"] = le_loc.fit_transform(df["location"])

FEATURES = ["no2", "so2", "co", "o3", "temperature", "humidity",
            "wind_speed", "month", "dayofweek", "is_weekend", "location_enc"]

X = df[FEATURES]

# ---------------- Regression: predict PM2.5 ----------------
y_reg = df["pm25"]
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

reg_model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
reg_model.fit(Xr_train, yr_train)
reg_preds = reg_model.predict(Xr_test)

reg_metrics = {
    "rmse": round(float(np.sqrt(mean_squared_error(yr_test, reg_preds))), 2),
    "mae": round(float(mean_absolute_error(yr_test, reg_preds)), 2),
    "r2": round(float(r2_score(yr_test, reg_preds)), 3),
}
print("Regression metrics:", reg_metrics)

# ---------------- Classification: predict AQI category ----------------
le_cat = LabelEncoder()
y_cls = le_cat.fit_transform(df["aqi_category"])

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)
cls_model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                           random_state=42, eval_metric="mlogloss")
cls_model.fit(Xc_train, yc_train)
cls_preds = cls_model.predict(Xc_test)

cls_metrics = {
    "accuracy": round(float(accuracy_score(yc_test, cls_preds)), 3),
    "precision": round(float(precision_score(yc_test, cls_preds, average="weighted", zero_division=0)), 3),
    "recall": round(float(recall_score(yc_test, cls_preds, average="weighted", zero_division=0)), 3),
    "f1": round(float(f1_score(yc_test, cls_preds, average="weighted", zero_division=0)), 3),
}
print("Classification metrics:", cls_metrics)

# ---------------- Save everything the API needs ----------------
joblib.dump(reg_model, f"{MODELS_DIR}/pm25_regressor.joblib")
joblib.dump(cls_model, f"{MODELS_DIR}/aqi_classifier.joblib")
joblib.dump(le_loc, f"{MODELS_DIR}/location_encoder.joblib")
joblib.dump(le_cat, f"{MODELS_DIR}/category_encoder.joblib")

with open(f"{MODELS_DIR}/feature_list.json", "w") as f:
    json.dump(FEATURES, f)

with open(f"{MODELS_DIR}/metrics.json", "w") as f:
    json.dump({"regression": reg_metrics, "classification": cls_metrics}, f, indent=2)

print("\nModels saved to", MODELS_DIR)
