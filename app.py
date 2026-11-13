"""
Air Quality Prediction — Flask Backend
Serves the dashboard (templates/index.html) and a small REST API that
the frontend JS (static/js/dashboard.js) calls for data, charts and
live predictions.

Run:
    python train_model.py   # once, to create models/
    python app.py           # starts the server on http://127.0.0.1:5000
"""
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

# ---------------- Load dataset + trained models at startup ----------------
df = pd.read_csv("data/air_quality_dataset.csv", parse_dates=["date"])
df["month"] = df["date"].dt.month
df["dayofweek"] = df["date"].dt.dayofweek
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

reg_model = joblib.load("models/pm25_regressor.joblib")
cls_model = joblib.load("models/aqi_classifier.joblib")
le_loc = joblib.load("models/location_encoder.joblib")
le_cat = joblib.load("models/category_encoder.joblib")

with open("models/feature_list.json") as f:
    FEATURES = json.load(f)

with open("models/metrics.json") as f:
    METRICS = json.load(f)

df["location_enc"] = le_loc.transform(df["location"])

AQI_COLORS = {
    "Good": "#2ecc71",
    "Satisfactory": "#9ACD32",
    "Moderate": "#f1c40f",
    "Poor": "#e67e22",
    "Very Poor": "#e74c3c",
    "Severe": "#8e44ad",
}


# ---------------- Page ----------------
@app.route("/")
def index():
    return render_template(
        "index.html",
        locations=sorted(df["location"].unique().tolist()),
        metrics=METRICS,
    )


# ---------------- API: list of locations ----------------
@app.route("/api/locations")
def api_locations():
    return jsonify(sorted(df["location"].unique().tolist()))


# ---------------- API: latest reading per location (for the overview cards) ----------------
@app.route("/api/overview")
def api_overview():
    latest = (
        df.sort_values("date")
        .groupby("location")
        .tail(1)
        .sort_values("location")
    )
    out = []
    for _, row in latest.iterrows():
        out.append({
            "location": row["location"],
            "date": row["date"].strftime("%Y-%m-%d"),
            "pm25": round(row["pm25"], 1),
            "pm10": round(row["pm10"], 1),
            "no2": round(row["no2"], 1),
            "so2": round(row["so2"], 1),
            "co": round(row["co"], 2),
            "o3": round(row["o3"], 1),
            "aqi_category": row["aqi_category"],
            "color": AQI_COLORS.get(row["aqi_category"], "#999999"),
        })
    return jsonify(out)


# ---------------- API: historical trend for a location (for the line chart) ----------------
@app.route("/api/history")
def api_history():
    location = request.args.get("location", df["location"].unique()[0])
    days = int(request.args.get("days", 90))

    sub = df[df["location"] == location].sort_values("date").tail(days)
    return jsonify({
        "location": location,
        "dates": sub["date"].dt.strftime("%Y-%m-%d").tolist(),
        "pm25": sub["pm25"].round(1).tolist(),
        "pm10": sub["pm10"].round(1).tolist(),
        "no2": sub["no2"].round(1).tolist(),
        "o3": sub["o3"].round(1).tolist(),
    })


# ---------------- API: monthly average PM2.5 per city (for the comparison chart) ----------------
@app.route("/api/monthly-comparison")
def api_monthly_comparison():
    monthly = df.groupby(["month", "location"])["pm25"].mean().reset_index()
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    series = {}
    for loc in sorted(df["location"].unique()):
        sub = monthly[monthly["location"] == loc].sort_values("month")
        values = [None] * 12
        for _, r in sub.iterrows():
            values[int(r["month"]) - 1] = round(r["pm25"], 1)
        series[loc] = values
    return jsonify({"labels": month_labels, "series": series})


# ---------------- API: correlation matrix (for the heatmap) ----------------
@app.route("/api/correlation")
def api_correlation():
    cols = ["pm25", "pm10", "no2", "so2", "co", "o3", "temperature", "humidity", "wind_speed"]
    corr = df[cols].corr().round(2)
    return jsonify({"columns": cols, "matrix": corr.values.tolist()})


# ---------------- API: AQI category distribution (for the bar chart) ----------------
@app.route("/api/aqi-distribution")
def api_aqi_distribution():
    order = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    counts = df["aqi_category"].value_counts().reindex(order).fillna(0).astype(int)
    return jsonify({
        "labels": order,
        "counts": counts.tolist(),
        "colors": [AQI_COLORS[c] for c in order],
    })


# ---------------- API: model metrics (for the "model performance" panel) ----------------
@app.route("/api/metrics")
def api_metrics():
    return jsonify(METRICS)


# ---------------- API: live prediction ----------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(force=True)

    try:
        location = payload["location"]
        no2 = float(payload["no2"])
        so2 = float(payload["so2"])
        co = float(payload["co"])
        o3 = float(payload["o3"])
        temperature = float(payload["temperature"])
        humidity = float(payload["humidity"])
        wind_speed = float(payload["wind_speed"])
        month = int(payload["month"])
        dayofweek = int(payload["dayofweek"])
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid or missing input: {e}"}), 400

    if location not in le_loc.classes_:
        return jsonify({"error": f"Unknown location '{location}'. Choices: {list(le_loc.classes_)}"}), 400

    is_weekend = 1 if dayofweek >= 5 else 0
    location_enc = int(le_loc.transform([location])[0])

    row = pd.DataFrame([{
        "no2": no2, "so2": so2, "co": co, "o3": o3,
        "temperature": temperature, "humidity": humidity, "wind_speed": wind_speed,
        "month": month, "dayofweek": dayofweek, "is_weekend": is_weekend,
        "location_enc": location_enc,
    }])[FEATURES]

    pred_pm25 = float(reg_model.predict(row)[0])
    pred_pm25 = max(pred_pm25, 0)
    pred_cat_idx = int(cls_model.predict(row)[0])
    pred_cat = le_cat.inverse_transform([pred_cat_idx])[0]
    proba = cls_model.predict_proba(row)[0]
    confidence = float(np.max(proba))

    return jsonify({
        "predicted_pm25": round(pred_pm25, 1),
        "predicted_category": pred_cat,
        "confidence": round(confidence * 100, 1),
        "color": AQI_COLORS.get(pred_cat, "#999999"),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
