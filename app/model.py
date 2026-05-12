"""
RecyclingPriceModel — wraps the trained XGBoost pipeline.

Loaded once at startup by FastAPI and reused across requests.
Provides:
  - predict()         single prediction with confidence interval
  - predict_batch()   vectorised batch prediction
  - info()            metadata dict for /model/info endpoint
"""

import json
import pickle
import uuid
from pathlib import Path
from typing import Dict, Any

import numpy as np

ROOT       = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "price_model.pkl"
META_PATH  = ROOT / "models" / "model_meta.json"

# Confidence interval half-widths per waste type (KES/kg)
# Derived from CV residual std — updated by retrain if needed
CI_HALF_WIDTH: Dict[str, float] = {
    "plastic": 3.2,  "paper": 1.6,  "metal": 6.5,  "glass": 1.4,
    "e_waste": 15.0, "organic": 1.0, "textile": 2.0, "rubber": 2.8,
}
CI_LEVEL = 0.90


class RecyclingPriceModel:
    def __init__(self):
        self._pipeline = None
        self._meta: Dict[str, Any] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self):
        """Load model + metadata from disk. Called once at startup."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run: python scripts/train.py"
            )
        with open(MODEL_PATH, "rb") as f:
            self._pipeline = pickle.load(f)

        if META_PATH.exists():
            with open(META_PATH) as f:
                self._meta = json.load(f)

        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        feature_vector: np.ndarray,   # shape (1, n_features)
        waste_type: str,
        weight_kg: float,
    ) -> Dict[str, Any]:
        """
        Run a single prediction.

        Returns a dict matching PredictionResponse (minus price_factors,
        which is computed in main.py from the request fields).
        """
        if not self._loaded:
            raise RuntimeError("Model is not loaded. Call model.load() first.")

        raw_price = float(self._pipeline.predict(feature_vector)[0])
        raw_price = max(raw_price, 1.0)  # hard floor

        half = CI_HALF_WIDTH.get(waste_type, 4.0)

        return {
            "predicted_price_per_kg": round(raw_price, 2),
            "total_estimated_price":  round(raw_price * weight_kg, 2),
            "confidence_interval": {
                "low":   round(max(raw_price - half, 0.5), 2),
                "high":  round(raw_price + half, 2),
                "level": CI_LEVEL,
            },
            "model_version": self._meta.get("version", "unknown"),
            "prediction_id": str(uuid.uuid4()),
        }

    def predict_batch(
        self,
        feature_matrix: np.ndarray,   # shape (n, n_features)
        waste_types: list[str],
        weights: list[float],
    ) -> list[Dict[str, Any]]:
        """Vectorised batch prediction — single model call, n results."""
        if not self._loaded:
            raise RuntimeError("Model is not loaded.")

        raw_prices = self._pipeline.predict(feature_matrix).tolist()

        results = []
        for raw_price, wt, wkg in zip(raw_prices, waste_types, weights):
            raw_price = max(float(raw_price), 1.0)
            half = CI_HALF_WIDTH.get(wt, 4.0)
            results.append({
                "predicted_price_per_kg": round(raw_price, 2),
                "total_estimated_price":  round(raw_price * wkg, 2),
                "confidence_interval": {
                    "low":   round(max(raw_price - half, 0.5), 2),
                    "high":  round(raw_price + half, 2),
                    "level": CI_LEVEL,
                },
                "model_version": self._meta.get("version", "unknown"),
                "prediction_id": str(uuid.uuid4()),
            })
        return results

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def info(self) -> Dict[str, Any]:
        return {
            "version":          self._meta.get("version", "unknown"),
            "algorithm":        self._meta.get("algorithm", "XGBoost"),
            "features":         self._meta.get("features", []),
            "waste_types":      self._meta.get("waste_types", []),
            "training_samples": self._meta.get("training_samples", 0),
            "metrics":          self._meta.get("metrics", {}),
            "created_at":       self._meta.get("created_at", ""),
        }


# Singleton — imported by main.py
price_model = RecyclingPriceModel()
