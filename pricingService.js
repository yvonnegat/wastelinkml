/**
 * pricingService.js
 *
 * Connects the RecycleIQ frontend to the ML Price Prediction API.
 * Falls back gracefully to rule-based estimates when the API is unreachable.
 *
 * Environment variables:
 *   VITE_ML_API_URL  — base URL of the FastAPI backend (e.g. http://localhost:8000)
 */

const ML_API_URL = import.meta.env.VITE_ML_API_URL || "";
const TIMEOUT_MS = 8_000;

// ---------------------------------------------------------------------------
// Base prices (KES/kg) — used for fallback only
// ---------------------------------------------------------------------------
const BASE_PRICES = {
  plastic: 22,
  paper: 10,
  metal: 45,
  glass: 8,
  e_waste: 80,
  organic: 5,
  textile: 12,
  rubber: 18,
};

const SEASONAL_DEMAND = {
  1: 0.8, 2: 0.82, 3: 0.95, 4: 0.85, 5: 0.88, 6: 0.9,
  7: 0.97, 8: 0.92, 9: 0.89, 10: 0.91, 11: 0.99, 12: 0.96,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Wrapper around fetch that enforces a timeout and returns parsed JSON.
 * @throws {Error} on network failure, timeout, or non-2xx status
 */
async function mlFetch(path, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const res = await fetch(`${ML_API_URL}${path}`, {
      ...options,
      signal: controller.signal,
      headers: { "Content-Type": "application/json", ...options.headers },
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `API error ${res.status}`);
    }

    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Rule-based fallback — same formula used in training data generation.
 * Returns a PredictionResponse-shaped object so callers don't need to branch.
 */
function fallbackPredict({
  wasteType,
  weightKg,
  distanceKm,
  consistencyScore,
  month,
  marketDemandIndex,
}) {
  const base = BASE_PRICES[wasteType] ?? 20;
  const seasonal = SEASONAL_DEMAND[month] ?? 0.9;
  const demand = marketDemandIndex ?? seasonal;
  const qualityMult = 0.7 + consistencyScore * 0.6;
  const distPenalty = Math.min(distanceKm * 0.12, base * 0.25);
  const weightBonus = Math.log1p(weightKg) * 0.4;

  const pricePerKg = Math.max(
    base * seasonal * (0.85 + demand * 0.3) * qualityMult - distPenalty + weightBonus,
    1
  );

  const half = { plastic: 3.2, paper: 1.6, metal: 6.5, glass: 1.4,
                  e_waste: 15, organic: 1, textile: 2, rubber: 2.8 }[wasteType] ?? 4;

  return {
    predicted_price_per_kg: Math.round(pricePerKg * 100) / 100,
    total_estimated_price: Math.round(pricePerKg * weightKg * 100) / 100,
    confidence_interval: {
      low: Math.max(Math.round((pricePerKg - half) * 100) / 100, 0.5),
      high: Math.round((pricePerKg + half) * 100) / 100,
      level: 0.9,
    },
    price_factors: {
      base_price_kes: base,
      weight_multiplier: Math.round((1 + Math.log1p(weightKg) * 0.02) * 1000) / 1000,
      quality_adjustment: Math.round((consistencyScore - 0.5) * base * 0.4 * 100) / 100,
      distance_penalty: -Math.abs(Math.round(distPenalty * 100) / 100),
      seasonality_factor: seasonal,
      demand_boost: Math.round((demand - 0.85) * base * 0.25 * 100) / 100,
    },
    waste_type: wasteType,
    weight_kg: weightKg,
    model_version: "fallback-rule-based",
    prediction_id: `fb-${Date.now()}`,
    _isFallback: true,
  };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Predict price for a single waste item.
 *
 * @param {object} params
 * @param {string} params.wasteType          - e.g. "plastic"
 * @param {number} params.weightKg           - e.g. 12.5
 * @param {number} params.distanceKm         - e.g. 8.3
 * @param {number} params.consistencyScore   - [0–1] from vision module
 * @param {number} [params.month]            - 1–12 (defaults to current month)
 * @param {number} [params.dayOfWeek]        - 0–6 (defaults to today)
 * @param {string} [params.centreId]         - optional centre ID
 * @param {number} [params.marketDemandIndex] - [0–1] optional override
 * @returns {Promise<PredictionResponse>}
 */
export async function predictPrice(params) {
  const {
    wasteType,
    weightKg,
    distanceKm,
    consistencyScore,
    month = new Date().getMonth() + 1,
    dayOfWeek = new Date().getDay(),
    centreId,
    marketDemandIndex,
  } = params;

  if (!ML_API_URL) {
    console.warn("[pricingService] VITE_ML_API_URL not set — using fallback.");
    return fallbackPredict({ wasteType, weightKg, distanceKm, consistencyScore, month, marketDemandIndex });
  }

  try {
    return await mlFetch("/predict", {
      method: "POST",
      body: JSON.stringify({
        waste_type: wasteType,
        weight_kg: weightKg,
        distance_km: distanceKm,
        consistency_score: consistencyScore,
        month,
        day_of_week: dayOfWeek,
        centre_id: centreId,
        market_demand_index: marketDemandIndex,
      }),
    });
  } catch (err) {
    console.warn(`[pricingService] ML API unavailable (${err.message}). Using fallback.`);
    return {
      ...fallbackPredict({ wasteType, weightKg, distanceKm, consistencyScore, month, marketDemandIndex }),
      _isFallback: true,
      _fallbackReason: err.message,
    };
  }
}

/**
 * Predict prices for multiple items in one network call.
 *
 * @param {object[]} items  — array of params objects (same shape as predictPrice)
 * @returns {Promise<PredictionResponse[]>}
 */
export async function predictPriceBatch(items) {
  if (!ML_API_URL) {
    return items.map((p) => fallbackPredict({
      wasteType: p.wasteType, weightKg: p.weightKg, distanceKm: p.distanceKm,
      consistencyScore: p.consistencyScore, month: p.month ?? new Date().getMonth() + 1,
      marketDemandIndex: p.marketDemandIndex,
    }));
  }

  try {
    const payload = items.map((p) => ({
      waste_type: p.wasteType,
      weight_kg: p.weightKg,
      distance_km: p.distanceKm,
      consistency_score: p.consistencyScore,
      month: p.month ?? new Date().getMonth() + 1,
      day_of_week: p.dayOfWeek ?? new Date().getDay(),
      centre_id: p.centreId,
      market_demand_index: p.marketDemandIndex,
    }));

    const data = await mlFetch("/predict/batch", { method: "POST", body: JSON.stringify({ items: payload }) });
    return data.predictions;
  } catch (err) {
    console.warn(`[pricingService] Batch API failed (${err.message}). Using fallback for all items.`);
    return items.map((p) => ({
      ...fallbackPredict({
        wasteType: p.wasteType, weightKg: p.weightKg, distanceKm: p.distanceKm,
        consistencyScore: p.consistencyScore, month: p.month ?? new Date().getMonth() + 1,
        marketDemandIndex: p.marketDemandIndex,
      }),
      _isFallback: true,
    }));
  }
}

/**
 * Check API health — useful for showing a "ML powered" badge vs "estimate" badge.
 * @returns {Promise<{ online: boolean, version: string }>}
 */
export async function checkMLHealth() {
  if (!ML_API_URL) return { online: false, version: null };
  try {
    const data = await mlFetch("/health");
    return { online: data.status === "ok" && data.model_loaded, version: data.version };
  } catch {
    return { online: false, version: null };
  }
}

/**
 * Fetch model metadata (version, metrics, features).
 * @returns {Promise<ModelInfo | null>}
 */
export async function getModelInfo() {
  if (!ML_API_URL) return null;
  try {
    return await mlFetch("/model/info");
  } catch {
    return null;
  }
}
