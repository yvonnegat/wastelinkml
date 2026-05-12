"""
Recycling Price Prediction API
FastAPI application — serves ML model predictions for the RecycleIQ frontend.

Endpoints:
  GET  /health            — liveness probe
  GET  /model/info        — model metadata & metrics
  POST /predict           — single item prediction
  POST /predict/batch     — up to 100 items
"""

import logging
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.features import engineer_features, compute_price_factors
from app.model import price_model
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("recycling-price-api")


# ---------------------------------------------------------------------------
# Lifespan — load model once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀  Loading price prediction model…")
    try:
        price_model.load()
        logger.info(f"✅  Model loaded: {price_model.info()['version']}")
    except FileNotFoundError as e:
        logger.error(f"❌  {e}")
        logger.error("    Run: python scripts/train.py   to train the model first.")
    yield
    logger.info("👋  Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RecycleIQ Price Prediction API",
    description="ML-powered pricing engine for Kenya recycling centres",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # restrict to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request timing middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    response.headers["X-Response-Time-Ms"] = f"{ms:.1f}"
    return response


# ---------------------------------------------------------------------------
# Exception handler
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error. Please try again."},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    return HealthResponse(
        status="ok" if price_model.is_loaded else "degraded",
        model_loaded=price_model.is_loaded,
        version=price_model.info().get("version", "not-loaded") if price_model.is_loaded else "not-loaded",
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["system"])
async def model_info():
    if not price_model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Run scripts/train.py first.")
    return ModelInfoResponse(**price_model.info())


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(req: PredictionRequest):
    """
    Predict price per kg for a single waste item.

    - **waste_type**: one of plastic, paper, metal, glass, e_waste, organic, textile, rubber
    - **weight_kg**: total weight in kilograms
    - **distance_km**: distance from the user to the nearest recycling centre
    - **consistency_score**: quality score from the vision module [0–1]
    - **month** / **day_of_week**: used for seasonality encoding
    """
    if not price_model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        features = engineer_features(
            waste_type=req.waste_type,
            weight_kg=req.weight_kg,
            distance_km=req.distance_km,
            consistency_score=req.consistency_score,
            month=req.month,
            day_of_week=req.day_of_week,
            market_demand_index=req.market_demand_index,
        )

        result = price_model.predict(
            feature_vector=features,
            waste_type=req.waste_type,
            weight_kg=req.weight_kg,
        )

        price_factors = compute_price_factors(
            waste_type=req.waste_type,
            weight_kg=req.weight_kg,
            distance_km=req.distance_km,
            consistency_score=req.consistency_score,
            month=req.month,
            market_demand_index=req.market_demand_index,
            predicted_price_per_kg=result["predicted_price_per_kg"],
        )

        logger.info(
            f"PREDICT  type={req.waste_type}  weight={req.weight_kg}kg  "
            f"price={result['predicted_price_per_kg']} KES/kg  "
            f"total={result['total_estimated_price']} KES"
        )

        return PredictionResponse(
            **result,
            price_factors=price_factors,
            waste_type=req.waste_type,
            weight_kg=req.weight_kg,
        )

    except Exception as exc:
        logger.error(f"Prediction error: {exc}")
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"])
async def predict_batch(req: BatchPredictionRequest):
    """
    Predict prices for up to 100 waste items in a single call.
    More efficient than calling /predict N times.
    """
    if not price_model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        feature_rows = []
        for item in req.items:
            feat = engineer_features(
                waste_type=item.waste_type,
                weight_kg=item.weight_kg,
                distance_km=item.distance_km,
                consistency_score=item.consistency_score,
                month=item.month,
                day_of_week=item.day_of_week,
                market_demand_index=item.market_demand_index,
            )
            feature_rows.append(feat.flatten())

        X = np.vstack(feature_rows)
        results = price_model.predict_batch(
            feature_matrix=X,
            waste_types=[i.waste_type for i in req.items],
            weights=[i.weight_kg for i in req.items],
        )

        predictions = []
        for item, result in zip(req.items, results):
            price_factors = compute_price_factors(
                waste_type=item.waste_type,
                weight_kg=item.weight_kg,
                distance_km=item.distance_km,
                consistency_score=item.consistency_score,
                month=item.month,
                market_demand_index=item.market_demand_index,
                predicted_price_per_kg=result["predicted_price_per_kg"],
            )
            predictions.append(PredictionResponse(
                **result,
                price_factors=price_factors,
                waste_type=item.waste_type,
                weight_kg=item.weight_kg,
            ))

        logger.info(f"BATCH_PREDICT  items={len(predictions)}")

        return BatchPredictionResponse(
            predictions=predictions,
            total_items=len(predictions),
            model_version=results[0]["model_version"] if results else "unknown",
        )

    except Exception as exc:
        logger.error(f"Batch prediction error: {exc}")
        raise HTTPException(status_code=422, detail=str(exc))
