# RecycleIQ — ML Price Prediction Backend

FastAPI + XGBoost service that powers the recycling price prediction engine.

## Features
- **Inputs**: waste type, weight, distance, consistency score (from vision), month/seasonality, market demand
- **Output**: predicted KES/kg price + 90% confidence interval + factor breakdown
- **Fallback**: frontend falls back to rule-based estimates when the API is unreachable

---

## Quick Start

### 1. Install dependencies
```bash
cd ml-backend
pip install -r requirements.txt
```

### 2. Generate data & train the model
```bash
# Generate 10 000 synthetic rows and train XGBoost
python scripts/train.py

# Or with custom sample count:
python scripts/train.py --samples 50000

# Or with your own CSV (must have the same columns as training_data.csv):
python scripts/train.py --data path/to/real_data.csv
```

Expected output:
```
Test  MAE=2.65 KES/kg  RMSE=4.77  R²=0.9526  MAPE=10.7%
CV MAE: 2.60 ± 0.06 KES/kg
✅  Model saved → models/price_model.pkl
```

### 3. Run the API server
```bash
uvicorn app.main:app --reload --port 8000
```

Visit **http://localhost:8000/docs** for the interactive Swagger UI.

---

## Docker

```bash
# Build (trains model inside container)
docker build -t recycleiq-ml .

# Run
docker run -p 8000:8000 recycleiq-ml
```

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness probe |
| GET | `/model/info` | Model version & metrics |
| POST | `/predict` | Single item prediction |
| POST | `/predict/batch` | Up to 100 items |

### POST /predict — example request
```json
{
  "waste_type": "plastic",
  "weight_kg": 12.5,
  "distance_km": 8.3,
  "consistency_score": 0.87,
  "month": 6,
  "day_of_week": 2
}
```

### POST /predict — example response
```json
{
  "predicted_price_per_kg": 23.45,
  "total_estimated_price": 293.13,
  "confidence_interval": { "low": 20.25, "high": 26.65, "level": 0.90 },
  "price_factors": {
    "base_price_kes": 22.0,
    "weight_multiplier": 1.052,
    "quality_adjustment": 3.26,
    "distance_penalty": -1.0,
    "seasonality_factor": 0.90,
    "demand_boost": 0.11
  },
  "waste_type": "plastic",
  "weight_kg": 12.5,
  "model_version": "v20260506_0941",
  "prediction_id": "a3f2b1c4-..."
}
```

---

## Frontend Integration

Set in your `.env`:
```
VITE_ML_API_URL=http://localhost:8000
```

Then in any component:
```jsx
import { usePricing } from "./hooks/usePricing";

const { predict, prediction, loading, mlOnline } = usePricing();

// On form submit:
await predict({
  wasteType: "plastic",
  weightKg: 12.5,
  distanceKm: 8.3,
  consistencyScore: 0.87,   // ← from vision module
});
```

Render with the ready-made card:
```jsx
import MLPricingCard from "./components/pricing/MLPricingCard";

<MLPricingCard
  prediction={prediction}
  loading={loading}
  mlOnline={mlOnline}
/>
```

---

## Model Details

| Feature | Type | Notes |
|---------|------|-------|
| `waste_type` | Categorical | One-hot encoded (8 types) |
| `weight_kg` | Continuous | Log-transformed for skewed dist. |
| `distance_km` | Continuous | Penalises transport cost |
| `consistency_score` | Continuous [0–1] | From vision module |
| `month` | Cyclic | sin/cos encoded |
| `day_of_week` | Cyclic | sin/cos encoded |
| `market_demand_index` | Continuous [0–1] | Seasonal baseline if omitted |
| `base_price` | Derived | Waste-type lookup |
| `price_x_quality` | Interaction | base_price × consistency_score |

**Algorithm**: XGBoost Regressor inside a StandardScaler pipeline  
**Training data**: 10 000 synthetic rows (realistic Kenya recycling market)  
**Test MAE**: ~2.65 KES/kg | **R²**: ~0.95  

---

## Replacing Synthetic Data with Real Data

When you have real transaction data, prepare a CSV with these columns:
```
waste_type, weight_kg, distance_km, consistency_score,
month, day_of_week, market_demand_index, price_per_kg
```

Then retrain:
```bash
python scripts/train.py --data path/to/real_data.csv
```

The server hot-reloads the model on restart — no code changes needed.
