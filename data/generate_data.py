"""
Generate synthetic training data for the Kenya recycling price model.
Now includes waste subtypes.

Run:  python data/generate_data.py
Output: data/training_data.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

SUBTYPES_BY_WASTE = {
    "metal":   ["copper", "aluminum", "brass", "steel", "tin"],
    "plastic": ["PET", "HDPE", "PP", "PVC", "mixed_plastic"],
    "paper":   ["cardboard", "newspaper", "office_paper", "mixed_paper"],
    "glass":   ["clear_glass", "colored_glass", "mixed_glass"],
    "e_waste": ["computers", "phones", "batteries", "cables", "mixed_ewaste"],
    "organic": ["food_waste", "garden_waste"],
    "textile": ["clothes", "industrial_textile"],
    "rubber":  ["tyres", "mixed_rubber"],
}

SUBTYPE_PRICES = {
    "copper": 600.0, "aluminum": 120.0, "brass": 350.0, "steel": 30.0, "tin": 25.0,
    "PET": 25.0, "HDPE": 20.0, "PP": 15.0, "PVC": 8.0, "mixed_plastic": 10.0,
    "cardboard": 12.0, "newspaper": 8.0, "office_paper": 14.0, "mixed_paper": 7.0,
    "clear_glass": 10.0, "colored_glass": 7.0, "mixed_glass": 5.0,
    "computers": 100.0, "phones": 120.0, "batteries": 60.0, "cables": 80.0, "mixed_ewaste": 70.0,
    "food_waste": 4.0, "garden_waste": 6.0,
    "clothes": 15.0, "industrial_textile": 8.0,
    "tyres": 20.0, "mixed_rubber": 12.0,
}

NOISE_STD = {
    "copper": 40.0, "aluminum": 12.0, "brass": 25.0, "steel": 3.0, "tin": 2.5,
    "PET": 2.5, "HDPE": 2.0, "PP": 1.5, "PVC": 1.0, "mixed_plastic": 1.2,
    "cardboard": 1.5, "newspaper": 1.0, "office_paper": 1.5, "mixed_paper": 1.0,
    "clear_glass": 1.2, "colored_glass": 1.0, "mixed_glass": 0.8,
    "computers": 18.0, "phones": 22.0, "batteries": 10.0, "cables": 12.0, "mixed_ewaste": 14.0,
    "food_waste": 0.6, "garden_waste": 0.8,
    "clothes": 2.0, "industrial_textile": 1.2,
    "tyres": 2.5, "mixed_rubber": 1.5,
}

SEASONAL_DEMAND = {
    1: 0.80, 2: 0.82, 3: 0.95, 4: 0.85, 5: 0.88, 6: 0.90,
    7: 0.97, 8: 0.92, 9: 0.89, 10: 0.91, 11: 0.99, 12: 0.96,
}

WASTE_TYPE_PROBS = [0.30, 0.20, 0.15, 0.05, 0.08, 0.10, 0.07, 0.05]
WASTE_TYPES = ["plastic", "paper", "metal", "glass", "e_waste", "organic", "textile", "rubber"]

SUBTYPE_PROBS = {
    "metal":   [0.05, 0.35, 0.08, 0.40, 0.12],
    "plastic": [0.35, 0.25, 0.20, 0.08, 0.12],
    "paper":   [0.40, 0.25, 0.20, 0.15],
    "glass":   [0.50, 0.30, 0.20],
    "e_waste": [0.25, 0.30, 0.20, 0.15, 0.10],
    "organic": [0.60, 0.40],
    "textile": [0.70, 0.30],
    "rubber":  [0.75, 0.25],
}


def simulate_price(sub_type, weight_kg, distance_km, consistency_score,
                   month, day_of_week, market_demand, rng):
    base     = SUBTYPE_PRICES[sub_type]
    seasonal = SEASONAL_DEMAND[month]
    demand_f = 0.85 + market_demand * 0.30
    quality  = 0.70 + consistency_score * 0.60
    dist_pen = min(distance_km * 0.12, base * 0.25)
    wt_bonus = np.log1p(weight_kg) * 0.4
    day_f    = 1.03 if day_of_week >= 5 else 1.0

    price = (base * seasonal * demand_f * quality * day_f
             - dist_pen + wt_bonus
             + rng.normal(0, NOISE_STD.get(sub_type, 5.0)))
    return max(price, 1.0)


def generate(n_samples: int = 10_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    waste_types  = rng.choice(WASTE_TYPES, size=n_samples, p=WASTE_TYPE_PROBS)
    months       = rng.integers(1, 13, size=n_samples)
    days_of_week = rng.integers(0, 7, size=n_samples)
    weights      = np.clip(rng.lognormal(1.5, 1.2, size=n_samples), 0.1, 2000)
    distances    = np.clip(rng.exponential(8, size=n_samples), 0.1, 120)
    consistency  = rng.beta(5, 2, size=n_samples)
    market_dem   = rng.beta(3, 3, size=n_samples)

    sub_types = [
        rng.choice(SUBTYPES_BY_WASTE[wt], p=SUBTYPE_PROBS[wt])
        for wt in waste_types
    ]

    prices = [
        simulate_price(st, w, d, c, m, dow, md, rng)
        for st, w, d, c, m, dow, md
        in zip(sub_types, weights, distances, consistency, months, days_of_week, market_dem)
    ]

    return pd.DataFrame({
        "waste_type":          waste_types,
        "sub_type":            sub_types,
        "weight_kg":           weights.round(3),
        "distance_km":         distances.round(3),
        "consistency_score":   consistency.round(4),
        "month":               months,
        "day_of_week":         days_of_week,
        "market_demand_index": market_dem.round(4),
        "price_per_kg":        np.round(prices, 2),
    })


if __name__ == "__main__":
    out_path = Path(__file__).parent / "training_data.csv"
    df = generate(n_samples=10_000)
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df):,} rows -> {out_path}")
    print("\nPrice summary by SUBTYPE:")
    print(
        df.groupby("sub_type")["price_per_kg"]
        .describe()[["mean", "min", "max"]]
        .round(1)
        .to_string()
    )