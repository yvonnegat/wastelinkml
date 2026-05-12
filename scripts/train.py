"""
Train the recycling price prediction model (with subtype support).

Usage:
    python scripts/train.py
    python scripts/train.py --data path/to/csv
    python scripts/train.py --samples 50000
"""

import argparse, json, pickle, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
except ImportError:
    print("❌  xgboost not installed. Run: pip install xgboost")
    sys.exit(1)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.features import engineer_features, FEATURE_NAMES, SUBTYPES_BY_WASTE
from data.generate_data import generate as generate_data


def load_or_generate(data_path, n_samples):
    if data_path and Path(data_path).exists():
        print(f"📂  Loading data from {data_path}")
        return pd.read_csv(data_path)
    print(f"🔧  Generating {n_samples:,} synthetic training samples…")
    return generate_data(n_samples=n_samples)


def build_feature_matrix(df):
    rows = []
    for _, row in df.iterrows():
        feat = engineer_features(
            waste_type=row["waste_type"],
            sub_type=row.get("sub_type") or None,
            weight_kg=row["weight_kg"],
            distance_km=row["distance_km"],
            consistency_score=row["consistency_score"],
            month=int(row["month"]),
            day_of_week=int(row["day_of_week"]),
            market_demand_index=float(row["market_demand_index"]),
        )
        rows.append(feat.flatten())
    return np.vstack(rows)


def evaluate(model, X, y, label=""):
    preds = model.predict(X)
    mae   = mean_absolute_error(y, preds)
    rmse  = np.sqrt(mean_squared_error(y, preds))
    r2    = r2_score(y, preds)
    mape  = np.mean(np.abs((y - preds) / np.clip(y, 1, None))) * 100
    print(f"  {label:<10}  MAE={mae:.2f} KES/kg  RMSE={rmse:.2f}  R²={r2:.4f}  MAPE={mape:.1f}%")
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "mape": round(mape, 4),
    }


def train(args):
    df = load_or_generate(args.data, args.samples)

    print(f"✅  Dataset: {len(df):,} rows | subtypes: {df['sub_type'].nunique()} unique")
    print(f"   Price range: {df['price_per_kg'].min():.1f} – {df['price_per_kg'].max():.1f} KES/kg\n")

    X = build_feature_matrix(df)
    y = df["price_per_kg"].values

    split = int(len(X) * 0.80)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", xgb.XGBRegressor(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    print("🚀  Training XGBoost model…")
    model.fit(X_train, y_train)

    print("\n📊  Evaluation:")
    train_m = evaluate(model, X_train, y_train, "Train")
    test_m  = evaluate(model, X_test,  y_test,  "Test ")

    print("\n🔄  5-fold cross-validation…")
    cv = cross_val_score(
        model,
        X,
        y,
        cv=KFold(5, shuffle=True, random_state=42),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    cv_mae = -cv.mean()
    print(f"  CV MAE: {cv_mae:.2f} ± {cv.std():.2f} KES/kg")

    print("\n📦  MAE by subtype (test set):")
    test_df = df.iloc[split:].reset_index(drop=True)
    test_preds = model.predict(X_test)

    for st in sorted(test_df["sub_type"].unique()):
        mask = test_df["sub_type"] == st
        if mask.sum() < 3:
            continue
        st_mae = mean_absolute_error(y_test[mask], test_preds[mask])
        bar = "█" * min(int(st_mae / 5), 30)
        print(f"  {st:<22} MAE={st_mae:7.2f} KES/kg  {bar}")

    importances = model.named_steps["xgb"].feature_importances_
    top = sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1])[:12]

    print("\n🔑  Top-12 features:")
    for name, imp in top:
        print(f"  {name:<26} {imp:.4f}  {'█' * int(imp * 300)}")

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    with open(models_dir / "price_model.pkl", "wb") as f:
        pickle.dump(model, f)

    version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M")

    meta = {
        "version": version,
        "algorithm": "XGBoost + StandardScaler",
        "features": FEATURE_NAMES,
        "waste_types": list(SUBTYPES_BY_WASTE.keys()),
        "subtypes": SUBTYPES_BY_WASTE,
        "training_samples": len(X_train),
        "metrics": {
            "train_mae": train_m["mae"],
            "test_mae": test_m["mae"],
            "test_rmse": test_m["rmse"],
            "test_r2": test_m["r2"],
            "test_mape": test_m["mape"],
            "cv_mae": round(cv_mae, 4),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(models_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅  Model saved  →  {models_dir / 'price_model.pkl'}")
    print(f"✅  Metadata     →  {models_dir / 'model_meta.json'}")
    print(f"🏷️   Version: {version}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None)
    p.add_argument("--samples", type=int, default=10_000)
    train(p.parse_args())