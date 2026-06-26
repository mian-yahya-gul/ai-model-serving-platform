"""
main.py
FastAPI inference service for the churn prediction model, instrumented
with Prometheus metrics for production monitoring.

Run:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health    - liveness/readiness probe
    POST /predict    - single prediction
    GET  /metrics     - Prometheus scrape endpoint
"""

import time
import os
import logging
from contextlib import asynccontextmanager
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from app.schemas import CustomerRecord, PredictionResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn-api")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "encoders.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.pkl")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    logger.info("Shutting down churn serving API")


app = FastAPI(
    title="Churn Prediction Serving API",
    description="Production-style FastAPI service for serving a customer churn classifier, with Prometheus monitoring.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Prometheus metrics ──────────────────────────────────────────────
PREDICTION_COUNT = Counter(
    "churn_predictions_total", "Total number of predictions made", ["prediction"]
)
PREDICTION_LATENCY = Histogram(
    "churn_prediction_latency_seconds", "Time taken to compute a prediction"
)
PREDICTION_ERRORS = Counter(
    "churn_prediction_errors_total", "Total number of failed prediction requests"
)
CHURN_PROBABILITY_GAUGE = Gauge(
    "churn_last_prediction_probability", "Churn probability of the most recent prediction"
)
MODEL_INFO = Gauge(
    "churn_model_info", "Static info about the loaded model", ["version"]
)

# ── Load model artifacts at startup ─────────────────────────────────
model = None
encoders = None
metadata = None


def load_model():
    global model, encoders, metadata
    try:
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        metadata = joblib.load(METADATA_PATH)
        MODEL_INFO.labels(version=metadata["model_version"]).set(1)
        logger.info(f"Model loaded successfully. Version: {metadata['model_version']}")
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {e}")
        raise


@app.get("/health", response_model=HealthResponse, tags=["Operations"])
def health():
    return HealthResponse(
        status="ok" if model is not None else "degraded",
        model_loaded=model is not None,
        model_version=metadata["model_version"] if metadata else "unknown",
    )


@app.get("/metrics", tags=["Operations"])
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(record: CustomerRecord):
    if model is None:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()
    try:
        df = pd.DataFrame([record.model_dump()])

        for col, le in encoders.items():
            if col in df.columns:
                # guard against unseen categories at inference time
                known_classes = set(le.classes_)
                if df[col].iloc[0] not in known_classes:
                    PREDICTION_ERRORS.inc()
                    raise HTTPException(
                        status_code=422,
                        detail=f"Unknown category '{df[col].iloc[0]}' for field '{col}'",
                    )
                df[col] = le.transform(df[col])

        df = df[metadata["feature_order"]]

        prediction = model.predict(df)[0]
        probability = float(model.predict_proba(df)[0][1])

        label = "Yes" if prediction == 1 else "No"

        PREDICTION_COUNT.labels(prediction=label).inc()
        CHURN_PROBABILITY_GAUGE.set(probability)

        return PredictionResponse(
            churn_prediction=label,
            churn_probability=round(probability, 4),
            model_version=metadata["model_version"],
        )

    except HTTPException:
        raise
    except Exception as e:
        PREDICTION_ERRORS.inc()
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error")
    finally:
        PREDICTION_LATENCY.observe(time.time() - start_time)


@app.get("/", tags=["Operations"])
def root():
    return {
        "service": "Churn Prediction Serving API",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }
