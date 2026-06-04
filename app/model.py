"""
app/range_model.py
Wrapper around the three XGBoost quantile models (price_range_models_v2.pkl).

v2 changes:
  - REMOVED: consistency_score, market_demand_index, quality_grade, quality_x_demand
  - ADDED:   condition (clean/mixed/contaminated) — seller provides at listing time
  - ADDED:   compute_tier() — auto-derives market_tier, tier_score, tier_multiplier
  - ADDED:   weight_x_condition interaction feature
  - ADDED:   collection_point multiplier (household/commercial/industrial)
  - Model file: price_range_models_v2.pkl
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("recycling-price-api")

_MODELS_PATH = Path(__file__).parent.parent / "models" / "price_range_models_v2.pkl"

# ---------------------------------------------------------------------------
# Informal market discounts (formal_rate × discount = street buyback rate)
# From primary recycler interview research — Kenya.
# ---------------------------------------------------------------------------
INFORMAL_DISCOUNTS: dict[str, float] = {
    "plastic":  0.62,   # PET KES 34 → street ~KES 17.5
    "paper":    0.70,   # cardboard KES 15 → street KES 10
    "metal":    0.90,   # steel KES 35 → street KES 30-40 (competitive)
    "glass":    0.55,   # clear glass KES 8.8 → street KES 3-5
    "e_waste":  0.95,   # phones KES 115 → street KES 65-100 (small gap)
    "organic":  0.85,   # food waste — recyclers sometimes pay more
    "rubber":   0.80,
    "textile":  0.80,
}

# ---------------------------------------------------------------------------
# Collection point multipliers (from primary recycler research)
# industrial: bulk volume, regular pickup → premium price
# commercial: business waste, standard pickup → baseline
# household: small quantities, irregular → discount
# ---------------------------------------------------------------------------
COLLECTION_POINT_MULT: dict[str, float] = {
    "household":   0.90,   # 10% discount for household waste
    "commercial":  1.00,   # Baseline for commercial/business waste
    "industrial":  1.12,   # 12% premium for industrial bulk waste
}

# ---------------------------------------------------------------------------
# County multipliers (regional price variations)
# ---------------------------------------------------------------------------
COUNTY_MULT: dict[str, float] = {
    "Nairobi":   1.00,   # Capital city - highest demand
    "Mombasa":   0.92,   # Port city - good market
    "Kisumu":    0.85,   # Lake region - developing market
    "Nakuru":    0.80,   # Agricultural hub
    "Eldoret":   0.78,   # Upcountry - lower prices
    "Kiambu":    0.95,   # Near Nairobi
    "Machakos":  0.88,   # Satellite town
    "Kajiado":   0.85,   # Rural-urban mix
}

# ---------------------------------------------------------------------------
# Market tier logic — auto-derived from weight + condition, no CV needed
# ---------------------------------------------------------------------------
# In app/range_model.py, replace the compute_tier function with:

def compute_tier(
    weight_kg: float,
    condition: str,
    waste_type: str,
) -> tuple[float, float, str]:
    """
    Returns (tier_multiplier, tier_score, tier_label).
    
    FIXED: Bulk = LOWER price (inverse relationship with weight)
    """
    # Weight score: 1.0 at 0kg, 0.0 at 200kg+ (inverse relationship)
    weight_score = float(np.clip(1.0 - (weight_kg / 200), 0, 1))
    
    # Condition score
    cond_score = {"clean": 1.0, "mixed": 0.5, "contaminated": 0.0}.get(condition, 0.5)
    cond_score = 0.78 + cond_score * 0.22  # Map to realistic range
    
    # Tier score: higher = more formal = better price
    tier_score = 0.5 * weight_score + 0.5 * ((cond_score - 0.78) / 0.22)
    tier_score = float(np.clip(tier_score, 0, 1))
    
    # Informal discount for this waste type
    informal_d = INFORMAL_DISCOUNTS.get(waste_type, 0.75)
    
    # Tier multiplier: informal_d (fully informal) → 1.0 (fully formal)
    tier_mult = informal_d + tier_score * (1.0 - informal_d)
    
    if tier_score >= 0.70:
        label = "formal"
    elif tier_score >= 0.35:
        label = "semi_formal"
    else:
        label = "informal"
    
    return round(tier_mult, 4), round(tier_score, 4), label

# ---------------------------------------------------------------------------
# Feature engineering — must match train_v2.py exactly
# ---------------------------------------------------------------------------
FEATURES = [
    "waste_type", "sub_type", "condition", "county", "collection_point",
    "market_tier",
    "log_weight", "log_distance", "tier_score", "tier_multiplier",
    "weight_x_condition",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
]


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_weight"]    = np.log1p(df["weight_kg"])
    df["log_distance"]  = np.log1p(df["distance_km"])
    df["month_sin"]     = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]     = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]       = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]       = np.cos(2 * np.pi * df["day_of_week"] / 7)
    cond_num = df["condition"].map({"clean": 1.0, "mixed": 0.5, "contaminated": 0.0})
    df["weight_x_condition"] = df["log_weight"] * cond_num
    return df.drop(columns=["month", "day_of_week", "weight_kg", "distance_km"],
                   errors="ignore")


def _market_signal(volatility: float) -> str:
    if volatility < 0.25:
        return "stable"
    if volatility < 0.50:
        return "moderate"
    return "volatile"


# ---------------------------------------------------------------------------
# Tier-aware seller advice
# ---------------------------------------------------------------------------
_TIER_ADVICE: dict[str, str] = {
    "informal":   (
        "You are in the informal/household seller tier. "
        "Bring more material or ensure it is clean to unlock higher rates."
    ),
    "semi_formal": (
        "You are in the mid-market tier. "
        "Sorting and cleaning your waste will improve your payout."
    ),
    "formal": (
        "You are in the formal/bulk seller tier — "
        "you are receiving wholesale market rates."
    ),
}

# ---------------------------------------------------------------------------
# Collection point advice
# ---------------------------------------------------------------------------
_COLLECTION_POINT_ADVICE: dict[str, str] = {
    "household": (
        "Household waste receives a 10% discount. "
        "For better rates, consider aggregating with neighbors or using a commercial collection point."
    ),
    "commercial": (
        "Commercial rates are baseline market price. "
        "Industrial waste receives a 12% premium."
    ),
    "industrial": (
        "Industrial waste receives a 12% premium! "
        "These are the best available rates due to volume and consistency."
    ),
}


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------
class PriceRangeModel:
    def __init__(self) -> None:
        self._models: Optional[dict] = None

    # ------------------------------------------------------------------
    def load(self, path: Path = _MODELS_PATH) -> None:
        import joblib
        if not path.exists():
            raise FileNotFoundError(
                f"Range model not found at {path}. "
                "Run: python scripts/train_v2.py"
            )
        self._models = joblib.load(path)
        logger.info(f"✅  v2 range model loaded from {path}")

    @property
    def is_loaded(self) -> bool:
        return self._models is not None

    # ------------------------------------------------------------------
    def predict(
        self,
        waste_type:        str,
        sub_type:          str,
        weight_kg:         float,
        distance_km:       float        = 5.0,
        condition:         str          = "clean",       # seller selects
        county:            str          = "Nairobi",
        collection_point:  str          = "commercial",
        month:             Optional[int] = None,          # defaults to today
        day_of_week:       Optional[int] = None,          # defaults to today
    ) -> dict:
        """
        Predict a price range for a single waste listing.

        Required from seller:
            waste_type, sub_type, weight_kg, condition, county

        Optional / internally resolved:
            distance_km, collection_point, month, day_of_week

        Returns structured dict with price_range, total_payout_range,
        market_info, and advice.
        """
        if not self.is_loaded:
            raise RuntimeError("Range model is not loaded.")

        # Resolve date fields
        now = datetime.now()
        month       = month       if month       is not None else now.month
        day_of_week = day_of_week if day_of_week is not None else now.weekday()

        # Sanitise inputs
        weight_kg   = float(np.clip(weight_kg,   0.5, 1000))
        distance_km = float(np.clip(distance_km, 0.1, 200))
        condition   = (
            condition.lower()
            if condition.lower() in ("clean", "mixed", "contaminated")
            else "mixed"
        )
        county = county if county in COUNTY_MULT else "Nairobi"
        collection_point = (
            collection_point.lower()
            if collection_point.lower() in COLLECTION_POINT_MULT
            else "commercial"
        )

        # Auto-derive market tier
        tier_mult, tier_score, tier = compute_tier(weight_kg, condition, waste_type)

        # Get multipliers
        cp_mult = COLLECTION_POINT_MULT.get(collection_point, 1.00)
        county_mult = COUNTY_MULT.get(county, 1.00)

        # Build feature row
        row = pd.DataFrame([{
            "waste_type":        waste_type,
            "sub_type":          sub_type,
            "weight_kg":         weight_kg,
            "distance_km":       distance_km,
            "condition":         condition,
            "county":            county,
            "collection_point":  collection_point,
            "market_tier":       tier,
            "tier_score":        tier_score,
            "tier_multiplier":   tier_mult,
            "month":             month,
            "day_of_week":       day_of_week,
        }])
        row_eng = _engineer(row)

        # Get base predictions from model
        lower = max(1.0, float(self._models["lower"].predict(row_eng[FEATURES])[0]))
        mid   = max(1.0, float(self._models["mid"].predict(row_eng[FEATURES])[0]))
        upper = max(1.0, float(self._models["upper"].predict(row_eng[FEATURES])[0]))

        # Apply collection point multiplier
        lower = lower * cp_mult
        mid   = mid * cp_mult
        upper = upper * cp_mult

        # Apply county multiplier
        lower = lower * county_mult
        mid   = mid * county_mult
        upper = upper * county_mult

        # Guard against quantile crossing (rare but possible)
        lower = min(lower, mid)
        upper = max(upper, mid)

        volatility = (upper - lower) / mid
        signal     = _market_signal(volatility)
        tier_advice = _TIER_ADVICE.get(tier, "")
        cp_advice = _COLLECTION_POINT_ADVICE.get(collection_point, "")
        
        combined_advice = f"{tier_advice} {cp_advice}".strip()

        return {
            "price_range": {
                "lower_bound": round(lower, 2),
                "recommended": round(mid,   2),
                "upper_bound": round(upper, 2),
                "currency":    "KES",
                "unit":        "per kg",
            },
            "total_payout_range": {
                "lower":       round(lower * weight_kg, 2),
                "recommended": round(mid   * weight_kg, 2),
                "upper":       round(upper * weight_kg, 2),
                "weight_kg":   weight_kg,
                "currency":    "KES",
            },
            "market_info": {
                "market_tier":   tier,
                "tier_score":    tier_score,
                "range_width":   round(upper - lower, 2),
                "market_signal": signal,
                "coverage":      "70%",
                "collection_point_multiplier": cp_mult,
                "county_multiplier": county_mult,
                "advice": (
                    f"Negotiate between KES {lower:.0f}–{upper:.0f}/kg. "
                    f"Fair market rate is KES {mid:.0f}/kg. {combined_advice}"
                ),
            },
        }

    # ------------------------------------------------------------------
    def predict_batch(self, items: list[dict]) -> list[dict]:
        return [self.predict(**item) for item in items]


# Module-level singleton — imported by main.py
price_range_model = PriceRangeModel()