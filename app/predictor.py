"""
predictor_v2.py
WasteLink Pricing Engine v2 — drop into your FastAPI app/predictor.py

Changes from v1:
  - REMOVED: consistency_score, market_demand_index, quality_grade
  - ADDED:   condition (seller provides), market_tier (auto-derived)
  - Two-tier pricing: informal household vs formal bulk, auto-detected

Required inputs (all from seller at listing time):
  waste_type, sub_type, weight_kg, distance_km,
  condition, county, collection_point
  
Optional:
  month (defaults to current month), day_of_week (defaults to today)
"""

import joblib, json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent

# ── Informal market discounts (from primary recycler research) ────────────────
INFORMAL_DISCOUNTS = {
    'plastic': 0.62, 'paper': 0.70, 'metal': 0.90,
    'glass':   0.55, 'e_waste': 0.95, 'organic': 0.85,
    'rubber':  0.80, 'textile': 0.80,
}

# ── Load model once ───────────────────────────────────────────────────────────
_models = None

def get_models():
    global _models
    if _models is None:
        _models = joblib.load(ROOT / 'models' / 'price_range_models_v2.pkl')
    return _models

# ── Auto-derive market tier ───────────────────────────────────────────────────
def compute_tier(weight_kg: float, condition: str, waste_type: str):
    weight_score = float(np.clip((weight_kg - 5) / (100 - 5), 0, 1))
    cond_score   = {'clean': 1.0, 'mixed': 0.5, 'contaminated': 0.0}.get(condition, 0.5)
    tier_score   = 0.55 * weight_score + 0.45 * cond_score
    informal_d   = INFORMAL_DISCOUNTS.get(waste_type, 0.75)
    tier_mult    = informal_d + tier_score * (1.0 - informal_d)

    if   tier_score >= 0.70: tier_label = 'formal'
    elif tier_score >= 0.35: tier_label = 'semi_formal'
    else:                    tier_label = 'informal'

    return round(tier_mult, 4), round(tier_score, 4), tier_label

# ── Feature engineering ───────────────────────────────────────────────────────
FEATURES = ['waste_type', 'sub_type', 'condition', 'county', 'collection_point',
            'market_tier', 'log_weight', 'log_distance', 'tier_score',
            'tier_multiplier', 'weight_x_condition',
            'month_sin', 'month_cos', 'dow_sin', 'dow_cos']

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['log_weight']    = np.log1p(df['weight_kg'])
    df['log_distance']  = np.log1p(df['distance_km'])
    df['month_sin']     = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos']     = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin']       = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos']       = np.cos(2 * np.pi * df['day_of_week'] / 7)
    cond_num = df['condition'].map({'clean':1.0,'mixed':0.5,'contaminated':0.0})
    df['weight_x_condition'] = df['log_weight'] * cond_num
    return df.drop(columns=['month', 'day_of_week', 'weight_kg',
                             'distance_km'], errors='ignore')

# ── Main prediction function ──────────────────────────────────────────────────
def predict_price_range(
    waste_type:      str,
    sub_type:        str,
    weight_kg:       float,
    distance_km:     float,
    condition:       str   = 'clean',     # seller selects: clean/mixed/contaminated
    county:          str   = 'Nairobi',
    collection_point:str   = 'commercial',
    month:           int   = None,         # defaults to current month
    day_of_week:     int   = None,         # defaults to today
) -> dict:
    """
    Returns a full price range recommendation.

    All inputs come directly from the seller's listing form.
    No CV module, no consistency_score required.

    FastAPI usage:
        from predictor_v2 import predict_price_range
        result = predict_price_range(
            waste_type='plastic', sub_type='PET',
            weight_kg=50, distance_km=5, condition='clean',
            county='Nairobi', collection_point='commercial'
        )
    """
    if month is None:      month      = datetime.now().month
    if day_of_week is None:day_of_week = datetime.now().weekday()

    # Validate inputs
    weight_kg   = float(np.clip(weight_kg,   0.5, 1000))
    distance_km = float(np.clip(distance_km, 0.1, 200))
    condition   = condition.lower() if condition.lower() in ['clean','mixed','contaminated'] else 'mixed'

    # Auto-derive market tier
    tier_mult, tier_score, tier = compute_tier(weight_kg, condition, waste_type)

    # Build input row
    row = pd.DataFrame([{
        'waste_type':       waste_type,
        'sub_type':         sub_type,
        'weight_kg':        weight_kg,
        'distance_km':      distance_km,
        'condition':        condition,
        'county':           county,
        'collection_point': collection_point,
        'market_tier':      tier,
        'tier_score':       tier_score,
        'tier_multiplier':  tier_mult,
        'month':            month,
        'day_of_week':      day_of_week,
    }])
    row_eng = engineer(row)

    models  = get_models()
    lo  = max(1.0, float(models['lower'].predict(row_eng[FEATURES])[0]))
    mid = max(1.0, float(models['mid'].predict(row_eng[FEATURES])[0]))
    hi  = max(1.0, float(models['upper'].predict(row_eng[FEATURES])[0]))

    # Market signal
    volatility     = (hi - lo) / mid
    market_signal  = 'stable' if volatility < 0.25 else 'moderate' if volatility < 0.50 else 'volatile'

    # Tier-aware advice
    tier_advice = {
        'informal':   'You are in the informal/household seller tier. '
                      'Bring more material or ensure it is clean to unlock higher rates.',
        'semi_formal':'You are in the mid-market tier. '
                      'Sorting and cleaning your waste will improve your payout.',
        'formal':     'You are in the formal/bulk seller tier — '
                      'you are receiving wholesale market rates.',
    }.get(tier, '')

    return {
        'price_range': {
            'lower_bound': round(lo,  2),
            'recommended': round(mid, 2),
            'upper_bound': round(hi,  2),
            'currency':    'KES',
            'unit':        'per kg',
        },
        'total_payout': {
            'lower':       round(lo  * weight_kg, 2),
            'recommended': round(mid * weight_kg, 2),
            'upper':       round(hi  * weight_kg, 2),
            'weight_kg':   weight_kg,
            'currency':    'KES',
        },
        'market_info': {
            'market_tier':   tier,
            'tier_score':    tier_score,
            'market_signal': market_signal,
            'range_width':   round(hi - lo, 2),
            'coverage':      '62%',
        },
        'advice': (
            f"Negotiate between KES {lo:.0f}–{hi:.0f}/kg. "
            f"Fair market rate is KES {mid:.0f}/kg. {tier_advice}"
        ),
        'inputs_used': {
            'waste_type': waste_type, 'sub_type': sub_type,
            'weight_kg': weight_kg, 'distance_km': distance_km,
            'condition': condition, 'county': county,
            'collection_point': collection_point,
            'auto_derived': {
                'market_tier':  tier,
                'tier_score':   tier_score,
                'tier_mult':    tier_mult,
            }
        }
    }


# ── Quick demo ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    tests = [
        ("plastic", "PET",         5,  3, "contaminated", "Nairobi", "household"),
        ("plastic", "PET",        50,  5, "clean",        "Nairobi", "commercial"),
        ("plastic", "PET",       150,  8, "clean",        "Nairobi", "industrial"),
        ("metal",   "steel",     10,  4, "mixed",         "Nairobi", "household"),
        ("metal",   "steel",     80, 10, "clean",         "Nairobi", "industrial"),
        ("paper",   "cardboard",  8,  3, "mixed",         "Nairobi", "household"),
        ("paper",   "cardboard",120,  7, "clean",         "Nairobi", "commercial"),
        ("e_waste", "mixed_ewaste",5, 2, "mixed",         "Nairobi", "household"),
    ]
    print("\nWasteLink Pricing Engine v2 — Demo Predictions")
    print("=" * 70)
    for wt, st, wkg, dkm, cond, county, cp in tests:
        r    = predict_price_range(wt, st, wkg, dkm, cond, county, cp)
        pr   = r['price_range']
        tp   = r['total_payout']
        mi   = r['market_info']
        tier = mi['market_tier'].upper()
        print(f"\n  {st} ({wkg}kg, {cond}, {county})")
        print(f"    Tier   : {tier}")
        print(f"    Per kg : KES {pr['lower_bound']} – {pr['recommended']} – {pr['upper_bound']}")
        print(f"    Total  : KES {tp['lower']:,.0f} – {tp['recommended']:,.0f} – {tp['upper']:,.0f}")
        print(f"    Signal : {mi['market_signal'].upper()}")