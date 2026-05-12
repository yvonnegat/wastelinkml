"""
Pydantic schemas for the recycling price prediction API.
All prices are in KES (Kenyan Shillings).
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import Optional, List, Dict
from enum import Enum


class WasteType(str, Enum):
    plastic  = "plastic"
    paper    = "paper"
    metal    = "metal"
    glass    = "glass"
    e_waste  = "e_waste"
    organic  = "organic"
    textile  = "textile"
    rubber   = "rubber"


class SubType(str, Enum):
    # Metal
    copper    = "copper"
    aluminum  = "aluminum"
    brass     = "brass"
    steel     = "steel"
    tin       = "tin"
    # Plastic
    PET           = "PET"
    HDPE          = "HDPE"
    PP            = "PP"
    PVC           = "PVC"
    mixed_plastic = "mixed_plastic"
    # Paper
    cardboard    = "cardboard"
    newspaper    = "newspaper"
    office_paper = "office_paper"
    mixed_paper  = "mixed_paper"
    # Glass
    clear_glass   = "clear_glass"
    colored_glass = "colored_glass"
    mixed_glass   = "mixed_glass"
    # E-waste
    computers    = "computers"
    phones       = "phones"
    batteries    = "batteries"
    cables       = "cables"
    mixed_ewaste = "mixed_ewaste"
    # Organic
    food_waste   = "food_waste"
    garden_waste = "garden_waste"
    # Textile
    clothes            = "clothes"
    industrial_textile = "industrial_textile"
    # Rubber
    tyres        = "tyres"
    mixed_rubber = "mixed_rubber"


VALID_SUBTYPES: Dict[str, List[str]] = {
    "metal":   ["copper", "aluminum", "brass", "steel", "tin"],
    "plastic": ["PET", "HDPE", "PP", "PVC", "mixed_plastic"],
    "paper":   ["cardboard", "newspaper", "office_paper", "mixed_paper"],
    "glass":   ["clear_glass", "colored_glass", "mixed_glass"],
    "e_waste": ["computers", "phones", "batteries", "cables", "mixed_ewaste"],
    "organic": ["food_waste", "garden_waste"],
    "textile": ["clothes", "industrial_textile"],
    "rubber":  ["tyres", "mixed_rubber"],
}


class PredictionRequest(BaseModel):
    waste_type:          WasteType         = Field(...)
    sub_type:            Optional[SubType] = Field(None)
    weight_kg:           float             = Field(..., gt=0, le=10_000)
    distance_km:         float             = Field(..., ge=0, le=500)
    consistency_score:   float             = Field(..., ge=0, le=1)
    month:               int               = Field(..., ge=1, le=12)
    day_of_week:         int               = Field(..., ge=0, le=6)
    centre_id:           Optional[str]     = Field(None)
    market_demand_index: Optional[float]   = Field(None, ge=0, le=1)

    @validator("weight_kg", "distance_km", "consistency_score")
    def round_floats(cls, v):
        return round(v, 4)

    @root_validator(skip_on_failure=True)
    def validate_subtype(cls, values):
        wt = values.get("waste_type")
        st = values.get("sub_type")
        if wt and st:
            allowed = VALID_SUBTYPES.get(wt, [])
            if st not in allowed:
                raise ValueError(f"sub_type '{st}' invalid for waste_type '{wt}'. Allowed: {allowed}")
        return values

    class Config:
        schema_extra = {
            "example": {
                "waste_type": "metal", "sub_type": "copper",
                "weight_kg": 5.0, "distance_km": 8.3,
                "consistency_score": 0.87, "month": 6, "day_of_week": 2,
            }
        }


class BatchPredictionRequest(BaseModel):
    items: List[PredictionRequest] = Field(..., min_items=1, max_items=100)


class PriceFactors(BaseModel):
    base_price_kes:     float
    weight_multiplier:  float
    quality_adjustment: float
    distance_penalty:   float
    seasonality_factor: float
    demand_boost:       float


class ConfidenceInterval(BaseModel):
    low:   float
    high:  float
    level: float = 0.90


class PredictionResponse(BaseModel):
    predicted_price_per_kg: float
    total_estimated_price:  float
    confidence_interval:    ConfidenceInterval
    price_factors:          PriceFactors
    waste_type:             str
    sub_type:               Optional[str] = None
    weight_kg:              float
    model_version:          str
    prediction_id:          str


class BatchPredictionResponse(BaseModel):
    predictions:   List[PredictionResponse]
    total_items:   int
    model_version: str


class ModelInfoResponse(BaseModel):
    version:          str
    algorithm:        str
    features:         List[str]
    waste_types:      List[str]
    subtypes:         Dict[str, List[str]]
    training_samples: int
    metrics:          Dict[str, float]
    created_at:       str


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    version:      str