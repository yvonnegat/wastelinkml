"""
app/main.py
Recycling Price Prediction API — WasteLink v2
FastAPI — serves ML price range predictions for waste listings.

v2 changes:
  - REMOVED: consistency_score, quality_grade from request
  - ADDED:   condition (clean / mixed / contaminated) in request
  - ADDED:   distance_km, collection_point (optional, exposed to callers)
  - Model:   price_range_models_v2.pkl  (no CV dependency)

Endpoints:
  GET  /health                — liveness probe
  POST /predict/range         — single listing price range
  POST /predict/range/batch   — up to 100 listings
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.model import price_range_model
from app.schema_range import (
    BatchRangePredictionRequest,
    BatchRangePredictionResponse,
    RangePredictionRequest,
    RangePredictionResponse,
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
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀  Loading v2 price range model…")
    try:
        price_range_model.load()
        logger.info("✅  v2 range model loaded (lower / mid / upper quantiles)")
    except FileNotFoundError as e:
        logger.error(f"❌  {e}")
        logger.error("    Run: python scripts/train_v2.py first.")
    yield
    logger.info("👋  Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="WasteLink Price Prediction API",
    description=(
        "Returns a price range (lower / recommended / upper) for a waste listing "
        "before the seller has chosen a recycler. "
        "v2: condition replaces consistency_score — no CV module required."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # lock down to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Response-Time-Ms"] = (
        f"{(time.perf_counter() - start) * 1000:.1f}"
    )
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

@app.get("/health", tags=["system"])
async def health():
    return {
        "status":       "ok" if price_range_model.is_loaded else "degraded",
        "model_loaded": price_range_model.is_loaded,
        "model_version": "2.0",
    }


@app.post(
    "/predict/range",
    response_model=RangePredictionResponse,
    tags=["prediction"],
    summary="Get a price range for a single waste listing",
)
async def predict_range(req: RangePredictionRequest):
    """
    Called at **listing creation time** — before the seller has chosen a recycler.

    The seller provides:
    - **waste_type** — top-level category (plastic, metal, paper, …)
    - **sub_type** — specific material (PET, copper, cardboard, …)
    - **weight_kg** — estimated weight
    - **condition** — clean / mixed / contaminated  *(replaces consistency_score)*
    - **county** — seller location

    Optional (sensible defaults applied if omitted):
    - **distance_km** — defaults to 5 km
    - **collection_point** — defaults to "commercial"

    Market tier (informal / semi_formal / formal) is **auto-derived** from
    weight + condition — no computer-vision module required.

    Returns:
    - **lower_bound** — floor price, don't accept less
    - **recommended** — fair market rate to list at
    - **upper_bound** — best case in current market
    - **market_tier** — auto-detected seller tier
    - **market_signal** — stable / moderate / volatile
    """
    if not price_range_model.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run scripts/train_v2.py first.",
        )
    try:
        result = price_range_model.predict(
            waste_type=req.waste_type,
            sub_type=req.sub_type,
            weight_kg=req.weight_kg,
            distance_km=req.distance_km,
            condition=req.condition,
            county=req.county,
            collection_point=req.collection_point,
        )

        pr = result["price_range"]
        mi = result["market_info"]
        logger.info(
            f"RANGE_PREDICT  {req.waste_type}/{req.sub_type}  "
            f"{req.weight_kg}kg  {req.condition}  {req.county}  "
            f"tier={mi['market_tier']}  "
            f"→ KES {pr['lower_bound']}–{pr['upper_bound']}/kg  "
            f"[{mi['market_signal']}]"
        )

        return RangePredictionResponse(
            waste_type=req.waste_type,
            sub_type=req.sub_type,
            weight_kg=req.weight_kg,
            **result,
        )

    except Exception as exc:
        logger.error(f"Prediction error: {exc}")
        raise HTTPException(status_code=422, detail=str(exc))


@app.post(
    "/predict/range/batch",
    response_model=BatchRangePredictionResponse,
    tags=["prediction"],
    summary="Get price ranges for up to 100 waste listings",
)
async def predict_range_batch(req: BatchRangePredictionRequest):
    """
    Batch version of `/predict/range`.
    Useful when a seller creates multiple listings at once.
    Each item in `items` accepts the same fields as the single endpoint.
    """
    if not price_range_model.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run scripts/train_v2.py first.",
        )
    try:
        results = price_range_model.predict_batch(
            [item.model_dump() for item in req.items]
        )

        predictions = [
            RangePredictionResponse(
                waste_type=item.waste_type,
                sub_type=item.sub_type,
                weight_kg=item.weight_kg,
                **result,
            )
            for item, result in zip(req.items, results)
        ]

        logger.info(f"RANGE_BATCH_PREDICT  items={len(predictions)}")

        return BatchRangePredictionResponse(
            predictions=predictions,
            total_items=len(predictions),
        )

    except Exception as exc:
        logger.error(f"Batch prediction error: {exc}")
        raise HTTPException(status_code=422, detail=str(exc))