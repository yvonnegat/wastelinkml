"""
Feature engineering for the recycling price prediction model.
Now supports waste sub-types (e.g. copper vs steel, PET vs PVC).
"""

import numpy as np
from typing import Dict, Optional

WASTE_TYPES = ["plastic", "paper", "metal", "glass", "e_waste", "organic", "textile", "rubber"]

SUBTYPES_BY_WASTE: Dict[str, list] = {
    "metal":   ["copper", "aluminum", "brass", "steel", "tin"],
    "plastic": ["PET", "HDPE", "PP", "PVC", "mixed_plastic"],
    "paper":   ["cardboard", "newspaper", "office_paper", "mixed_paper"],
    "glass":   ["clear_glass", "colored_glass", "mixed_glass"],
    "e_waste": ["computers", "phones", "batteries", "cables", "mixed_ewaste"],
    "organic": ["food_waste", "garden_waste"],
    "textile": ["clothes", "industrial_textile"],
    "rubber":  ["tyres", "mixed_rubber"],
}

# Flat ordered list — used for OHE column order
ALL_SUBTYPES = [st for subtypes in SUBTYPES_BY_WASTE.values() for st in subtypes]

# Base prices per SUBTYPE (KES/kg)
SUBTYPE_PRICES: Dict[str, float] = {
    "copper": 600.0, "aluminum": 120.0, "brass": 350.0, "steel": 30.0, "tin": 25.0,
    "PET": 25.0, "HDPE": 20.0, "PP": 15.0, "PVC": 8.0, "mixed_plastic": 10.0,
    "cardboard": 12.0, "newspaper": 8.0, "office_paper": 14.0, "mixed_paper": 7.0,
    "clear_glass": 10.0, "colored_glass": 7.0, "mixed_glass": 5.0,
    "computers": 100.0, "phones": 120.0, "batteries": 60.0, "cables": 80.0, "mixed_ewaste": 70.0,
    "food_waste": 4.0, "garden_waste": 6.0,
    "clothes": 15.0, "industrial_textile": 8.0,
    "tyres": 20.0, "mixed_rubber": 12.0,
}

# Fallback parent-type prices
BASE_PRICES: Dict[str, float] = {
    "plastic": 22.0, "paper": 10.0, "metal": 45.0, "glass": 8.0,
    "e_waste": 80.0, "organic": 5.0, "textile": 12.0, "rubber": 18.0,
}

SEASONAL_DEMAND: Dict[int, float] = {
    1: 0.80, 2: 0.82, 3: 0.95, 4: 0.85, 5: 0.88, 6: 0.90,
    7: 0.97, 8: 0.92, 9: 0.89, 10: 0.91, 11: 0.99, 12: 0.96,
}

FEATURE_NAMES = (
    [f"waste_{wt}" for wt in WASTE_TYPES]
    + [f"sub_{st}" for st in ALL_SUBTYPES]
    + [
        "has_subtype",
        "weight_kg",
        "distance_km",
        "consistency_score",
        "month_sin", "month_cos",
        "dow_sin", "dow_cos",
        "market_demand_index",
        "base_price",
        "weight_log",
        "price_x_quality",
    ]
)


def get_base_price(waste_type: str, sub_type: Optional[str] = None) -> float:
    if sub_type and sub_type in SUBTYPE_PRICES:
        return SUBTYPE_PRICES[sub_type]
    return BASE_PRICES.get(waste_type, 20.0)


def engineer_features(
    waste_type: str,
    weight_kg: float,
    distance_km: float,
    consistency_score: float,
    month: int,
    day_of_week: int,
    sub_type: Optional[str] = None,
    market_demand_index: Optional[float] = None,
    **_kwargs,
) -> np.ndarray:

    type_ohe = [1.0 if wt == waste_type else 0.0 for wt in WASTE_TYPES]
    sub_ohe = [1.0 if st == sub_type else 0.0 for st in ALL_SUBTYPES]

    has_subtype = 1.0 if sub_type else 0.0

    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    dow_sin   = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos   = np.cos(2 * np.pi * day_of_week / 7)

    if market_demand_index is None:
        market_demand_index = SEASONAL_DEMAND.get(month, 0.90)

    base_price   = get_base_price(waste_type, sub_type)
    weight_log   = np.log1p(weight_kg)
    price_x_qual = base_price * consistency_score

    features = np.array(
        type_ohe + sub_ohe + [
            has_subtype,
            weight_kg, distance_km, consistency_score,
            month_sin, month_cos, dow_sin, dow_cos,
            market_demand_index,
            base_price, weight_log, price_x_qual,
        ],
        dtype=np.float32,
    )

    return features.reshape(1, -1)


def compute_price_factors(
    waste_type: str,
    weight_kg: float,
    distance_km: float,
    consistency_score: float,
    month: int,
    sub_type: Optional[str] = None,
    market_demand_index: Optional[float] = None,
    predicted_price_per_kg: float = 0.0,
) -> Dict[str, float]:

    base = get_base_price(waste_type, sub_type)
    demand = market_demand_index if market_demand_index is not None else SEASONAL_DEMAND.get(month, 0.90)

    return {
        "base_price_kes": round(base, 2),
        "weight_multiplier": round(1.0 + np.log1p(weight_kg) * 0.02, 3),
        "quality_adjustment": round((consistency_score - 0.5) * base * 0.4, 2),
        "distance_penalty": -abs(round(min(distance_km * 0.15, base * 0.30), 2)),
        "seasonality_factor": round(SEASONAL_DEMAND.get(month, 0.90), 3),
        "demand_boost": round((demand - 0.85) * base * 0.25, 2),
    }