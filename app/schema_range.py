"""
app/schema_range.py
Pydantic schemas for the v2 WasteLink price-range prediction API.

v2 changes:
  - REMOVED: consistency_score, quality_grade  (CV module dependency)
  - ADDED:   condition  (seller selects: clean / mixed / contaminated)
  - ADDED:   distance_km, collection_point  (now optionally exposed to callers)
  - ADDED:   market_tier, tier_score in RangePredictionResponse.market_info
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums / literals
# ---------------------------------------------------------------------------
WasteType = Literal[
    "plastic", "metal", "paper", "e_waste",
    "organic", "glass", "textile", "rubber",
]

Condition = Literal["clean", "mixed", "contaminated"]

County = Literal["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret"]

CollectionPoint = Literal["household", "commercial", "industrial", "dump_site"]

MarketTier = Literal["informal", "semi_formal", "formal"]

MarketSignal = Literal["stable", "moderate", "volatile"]


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------
class RangePredictionRequest(BaseModel):
    """
    Inputs the seller provides at listing creation time.
    No computer-vision or consistency-score data required.
    """

    waste_type: WasteType = Field(
        ...,
        description="Top-level waste category.",
        examples=["plastic"],
    )
    sub_type: str = Field(
        ...,
        min_length=2,
        max_length=50,
        description="Specific material sub-type (e.g. 'PET', 'copper', 'cardboard').",
        examples=["PET"],
    )
    weight_kg: float = Field(
        ...,
        ge=0.5,
        le=1000,
        description="Estimated weight in kilograms.",
        examples=[50.0],
    )
    condition: Condition = Field(
        default="clean",
        description=(
            "Seller-assessed material condition. "
            "'clean' = sorted/uncontaminated; "
            "'mixed' = partially sorted; "
            "'contaminated' = unsorted/dirty."
        ),
        examples=["clean"],
    )
    county: County = Field(
        default="Nairobi",
        description="Kenyan county where the material is located.",
        examples=["Nairobi"],
    )

    # Optional — callers may omit; model uses sensible defaults
    distance_km: float = Field(
        default=5.0,
        ge=0.1,
        le=200,
        description=(
            "Estimated distance to collection point in km. "
            "Defaults to 5 km (neutral urban estimate) if unknown."
        ),
        examples=[5.0],
    )
    collection_point: CollectionPoint = Field(
        default="commercial",
        description="Type of collection point.",
        examples=["commercial"],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "waste_type":       "plastic",
                "sub_type":         "PET",
                "weight_kg":        50.0,
                "condition":        "clean",
                "county":           "Nairobi",
                "distance_km":      5.0,
                "collection_point": "commercial",
            }
        }
    }


# ---------------------------------------------------------------------------
# Response — nested sub-models
# ---------------------------------------------------------------------------
class PriceRange(BaseModel):
    lower_bound: float = Field(..., description="Floor price — do not accept below this.")
    recommended: float = Field(..., description="Fair market rate to list at.")
    upper_bound: float = Field(..., description="Best-case price in current market.")
    currency:    str   = Field(default="KES")
    unit:        str   = Field(default="per kg")


class TotalPayoutRange(BaseModel):
    lower:       float = Field(..., description="Total payout at lower bound.")
    recommended: float = Field(..., description="Total payout at recommended rate.")
    upper:       float = Field(..., description="Total payout at upper bound.")
    weight_kg:   float
    currency:    str   = Field(default="KES")


class MarketInfo(BaseModel):
    market_tier:   MarketTier   = Field(..., description="Auto-derived seller tier.")
    tier_score:    float        = Field(..., description="Continuous tier score (0–1).")
    range_width:   float        = Field(..., description="Width of price range (KES/kg).")
    market_signal: MarketSignal = Field(..., description="Price stability indicator.")
    coverage:      str          = Field(..., description="Statistical interval coverage.")
    advice:        str          = Field(..., description="Human-readable seller advice.")


# ---------------------------------------------------------------------------
# Single prediction response
# ---------------------------------------------------------------------------
class RangePredictionResponse(BaseModel):
    waste_type:        str
    sub_type:          str
    weight_kg:         float
    price_range:       PriceRange
    total_payout_range: TotalPayoutRange
    market_info:       MarketInfo

    model_config = {
        "json_schema_extra": {
            "example": {
                "waste_type": "plastic",
                "sub_type":   "PET",
                "weight_kg":  50.0,
                "price_range": {
                    "lower_bound": 18.50,
                    "recommended": 24.00,
                    "upper_bound": 30.10,
                    "currency":    "KES",
                    "unit":        "per kg",
                },
                "total_payout_range": {
                    "lower":       925.0,
                    "recommended": 1200.0,
                    "upper":       1505.0,
                    "weight_kg":   50.0,
                    "currency":    "KES",
                },
                "market_info": {
                    "market_tier":   "semi_formal",
                    "tier_score":    0.52,
                    "range_width":   11.60,
                    "market_signal": "moderate",
                    "coverage":      "70%",
                    "advice": (
                        "Negotiate between KES 19–30/kg. "
                        "Fair market rate is KES 24/kg. "
                        "You are in the mid-market tier."
                    ),
                },
            }
        }
    }


# ---------------------------------------------------------------------------
# Batch request / response
# ---------------------------------------------------------------------------
class BatchRangePredictionRequest(BaseModel):
    items: list[RangePredictionRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Up to 100 listing prediction requests.",
    )


class BatchRangePredictionResponse(BaseModel):
    predictions: list[RangePredictionResponse]
    total_items: int