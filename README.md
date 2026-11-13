# Air Quality Prediction — Full-Stack Project

A connected frontend + backend web app for the **Air Quality Prediction** (Machine Learning for Data Analytics) project. The Flask backend trains and serves XGBoost regression/classification models on `data/air_quality_dataset.csv`; the frontend is a live dashboard (HTML/CSS/Chart.js) that shows current air quality, historical trends, correlations, and lets you request a live prediction.

## Architecture

```
Browser (dashboard.js, Chart.js)
        |  fetch() calls
        v
Flask REST API (app.py)
        |
        v
Trained models (models/*.joblib)  <-- train_model.py reads data/air_quality_dataset.csv
```

## Project structure

```
backend/
├── app.py                  # Flask app: serves the page + REST API
├── train_model.py          # Trains & saves the ML models (run once first)
├── requirements.txt
├── data/
│   └── air_quality_dataset.csv
├── models/                 # created by train_model.py
├── templates/
│   └── index.html          # dashboard page
└── static/
    ├── css/style.css
    └── js/dashboard.js
```

## Setup & run

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

python train_model.py           # trains models, writes models/*.joblib + metrics.json
python app.py                   # starts the server
```

Open **http://127.0.0.1:5000** in your browser.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/locations` | List of cities in the dataset |
| GET | `/api/overview` | Latest reading per city (for the cards) |
| GET | `/api/history?location=X&days=90` | Historical PM2.5/NO2/O3 series for one city |
| GET | `/api/monthly-comparison` | Average monthly PM2.5 per city (seasonality) |
| GET | `/api/correlation` | Correlation matrix of pollutants & weather |
| GET | `/api/aqi-distribution` | Count of days per AQI category |
| GET | `/api/metrics` | Saved regression/classification evaluation metrics |
| POST | `/api/predict` | Predicts PM2.5 + AQI category from input conditions |

`POST /api/predict` body example:

```json
{
  "location": "Chennai",
  "no2": 35, "so2": 12, "co": 0.6, "o3": 40,
  "temperature": 27, "humidity": 70, "wind_speed": 3.5,
  "month": 12, "dayofweek": 3
}
```

Response:

```json
{
  "predicted_pm25": 71.4,
  "predicted_category": "Moderate",
  "confidence": 82.3,
  "color": "#E8C93D"
}
```

## Swapping in a real dataset

Replace `data/air_quality_dataset.csv` with a real file (CPCB/EPA/your own sensor export) using the **same column names** — `date, location, pm25, pm10, no2, so2, co, o3, temperature, humidity, wind_speed, aqi_category` — then re-run `python train_model.py`. Nothing else needs to change.

## Notes for your project report / viva

- **Regression** target: `pm25` (continuous) — model: XGBoost Regressor.
- **Classification** target: `aqi_category` (6 classes) — model: XGBoost Classifier.
- Metrics from the last training run are saved to `models/metrics.json` and shown live in the dashboard header badges.
- The dashboard itself fulfils "Phase 7 — Web Application" from the project plan: Frontend → Backend/API → ML model → Dataset → Prediction → Dashboard.
