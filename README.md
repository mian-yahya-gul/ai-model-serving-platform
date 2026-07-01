# AI Model Serving & Monitoring Platform

A production style FastAPI service for serving a machine learning model, instrumented end to end with Prometheus metrics and a provisioned Grafana dashboard — built to demonstrate the operational side of MLOps.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED)
![Prometheus](https://img.shields.io/badge/Prometheus-monitoring-E6522C)
![Grafana](https://img.shields.io/badge/Grafana-dashboards-F46800)
![CI](https://github.com/mian-yahya-gul/ai-model-serving-platform/actions/workflows/ci.yml/badge.svg)

## What this project demonstrates

Training a model is one part of MLOps — **operating it reliably in production is the harder part**. This project focuses entirely on that second half:

- **REST inference API** built with FastAPI and strict Pydantic request validation (rejects malformed input before it reaches the model)
- **Prometheus instrumentation** — custom metrics for prediction counts, latency distribution, error rates, and live churn probability values, exposed on a standard `/metrics` endpoint
- **Provisioned Grafana dashboard** — auto loads on startup via Docker volume mounts, no manual dashboard setup required
- **Health checks** — a `/health` endpoint plus a Docker `HEALTHCHECK` directive, so orchestrators (Kubernetes, Docker Swarm) can detect a degraded service automatically
- **Full containerized stack** — API + Prometheus + Grafana orchestrated together with a single `docker-compose up`
- **CI** — GitHub Actions runs the full test suite and a Docker build on every push

The model itself is the churn classifier from my [churn-mlops-pipeline](https://github.com/mian-yahya-gul/churn-mlops-pipeline) project, trained on the real IBM Telco Customer Churn dataset, this project picks up where that one ends, focused on what happens *after* a model is trained.

## Architecture

```
                    ┌─────────────┐
   POST /predict ──►│   FastAPI   │──► churn_model.pkl (RandomForest)
                    │   (app)     │
                    └──────┬──────┘
                           │ /metrics
                           ▼
                    ┌─────────────┐
                    │ Prometheus  │  (scrapes every 10s)
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Grafana   │  (pre-provisioned dashboard)
                    └─────────────┘
```

## Metrics exposed

| Metric | Type | Purpose |
|---|---|---|
| `churn_predictions_total` | Counter | Total predictions, labeled by outcome (Yes/No) |
| `churn_prediction_latency_seconds` | Histogram | Inference latency distribution (enables p95/p99 tracking) |
| `churn_prediction_errors_total` | Counter | Failed prediction requests |
| `churn_last_prediction_probability` | Gauge | Most recent churn probability — useful for live monitoring |
| `churn_model_info` | Gauge | Static model version tracking |

## Running it

### Docker Compose (full stack recommended)

```bash
docker-compose up --build
```

- API: `http://localhost:8000/docs` (interactive Swagger UI)
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (login: `admin` / `admin`) — the "Churn Prediction API  Monitoring" dashboard is pre loaded

### Local (without Docker)

```bash
pip install -r requirements.txt
python models/train_export_model.py      # exports the model artifact
uvicorn app.main:app --reload --port 8000
```

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 95.50, "TotalCharges": 191.00
  }'
```

Response:
```json
{
  "churn_prediction": "Yes",
  "churn_probability": 0.9106,
  "model_version": "v1.0.0"
}
```

### Running tests

```bash
pytest tests/ -v
```

8 tests cover health checks, valid/invalid input handling, prediction sanity, and metrics exposure.

## Tech Stack

`Python` · `FastAPI` · `Pydantic` · `Docker` · `Docker Compose` · `Prometheus` · `Grafana` · `GitHub Actions` · `pytest`

## Project Structure

```
.
├── app/
│   ├── main.py           # FastAPI app + Prometheus instrumentation
│   └── schemas.py        # Pydantic request/response models
├── models/
│   └── train_export_model.py
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/provisioning/
├── tests/
│   └── test_api.py
├── .github/workflows/ci.yml
├── Dockerfile
└── docker-compose.yml
```

## Author

**Mian Yahya Gul** — [LinkedIn](https://www.linkedin.com/in/mian-yahya-gul/) · [GitHub](https://github.com/mian-yahya-gul)
