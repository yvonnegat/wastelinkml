
# ── Drop this into your FastAPI predictor.py ────────────────────────────────
import joblib, numpy as np, pandas as pd
from pathlib import Path

_models = None

def get_models():
    global _models
    if _models is None:
        _models = joblib.load(Path(__file__).parent / "models" / "price_range_models.pkl")
    return _models

def engineer(df):
    df = df.copy()
    df["log_weight"]       = np.log1p(df["weight_kg"])
    df["log_distance"]     = np.log1p(df["distance_km"])
    df["month_sin"]        = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]        = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]          = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]          = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["quality_x_demand"] = df["consistency_score"] * df["market_demand_index"]
    return df.drop(columns=["month", "day_of_week"], errors="ignore")

def predict_price_range(waste_type, sub_type, weight_kg, distance_km,
                        consistency_score, county="Nairobi",
                        collection_point="commercial", quality_grade="good",
                        market_demand_index=0.65, month=6, day_of_week=1):
    models = get_models()
    row = pd.DataFrame([{
        "waste_type": waste_type, "sub_type": sub_type,
        "weight_kg": weight_kg, "distance_km": distance_km,
        "consistency_score": consistency_score,
        "market_demand_index": market_demand_index,
        "county": county, "collection_point": collection_point,
        "quality_grade": quality_grade, "month": month, "day_of_week": day_of_week,
    }])
    row = engineer(row)
    lower = max(1.0, float(models["lower"].predict(row)[0]))
    mid   = max(1.0, float(models["mid"].predict(row)[0]))
    upper = max(1.0, float(models["upper"].predict(row)[0]))
    volatility = (upper - lower) / mid
    signal = "stable" if volatility < 0.25 else "moderate" if volatility < 0.50 else "volatile"
    return {
        "price_range": {
            "lower_bound": round(lower, 2),
            "recommended": round(mid,   2),
            "upper_bound": round(upper, 2),
            "currency": "KES", "unit": "per kg",
        },
        "total_payout_range": {
            "lower":       round(lower * weight_kg, 2),
            "recommended": round(mid   * weight_kg, 2),
            "upper":       round(upper * weight_kg, 2),
            "weight_kg":   weight_kg, "currency": "KES",
        },
        "market_info": {
            "range_width":   round(upper - lower, 2),
            "market_signal": signal,
            "coverage":      "70%",
            "advice": f"Negotiate between KES {lower:.0f}–{upper:.0f}/kg. Fair rate is KES {mid:.0f}/kg.",
        }
    }
