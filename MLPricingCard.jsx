/**
 * MLPricingCard.jsx
 *
 * Displays a full ML-powered price prediction result.
 * Includes: price summary, confidence interval bar, factor breakdown,
 * ML vs fallback badge, and a re-estimate button.
 *
 * Props:
 *   prediction   — result from usePricing().prediction
 *   loading      — bool
 *   error        — string | null
 *   mlOnline     — bool | null
 *   onReestimate — () => void
 */

import React from "react";

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function Badge({ online }) {
  if (online === null) {
    return (
      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-500">
        <span className="w-1.5 h-1.5 rounded-full bg-gray-400 animate-pulse" />
        Checking…
      </span>
    );
  }
  return online ? (
    <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-50 text-emerald-700 border border-emerald-200">
      <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
      ML Powered
    </span>
  ) : (
    <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-amber-50 text-amber-700 border border-amber-200">
      <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
      Rule-based estimate
    </span>
  );
}

function ConfidenceBar({ low, high, predicted }) {
  // Visualise where the predicted price sits within [low, high]
  const range = high - low || 1;
  const pct = Math.round(((predicted - low) / range) * 100);

  return (
    <div>
      <div className="flex justify-between text-xs text-gray-500 mb-1">
        <span>KES {low.toFixed(2)}</span>
        <span className="font-medium text-gray-700">90% confidence range</span>
        <span>KES {high.toFixed(2)}</span>
      </div>
      <div className="relative h-2 rounded-full bg-gray-100 overflow-visible">
        {/* Range fill */}
        <div className="absolute inset-0 rounded-full bg-emerald-100" />
        {/* Marker */}
        <div
          className="absolute top-1/2 -translate-y-1/2 w-3.5 h-3.5 rounded-full bg-emerald-600 border-2 border-white shadow"
          style={{ left: `calc(${pct}% - 7px)` }}
          title={`KES ${predicted.toFixed(2)}/kg`}
        />
      </div>
    </div>
  );
}

function FactorRow({ label, value, color = "text-gray-700" }) {
  const sign = value > 0 ? "+" : "";
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-gray-50 last:border-0">
      <span className="text-sm text-gray-600">{label}</span>
      <span className={`text-sm font-medium tabular-nums ${color}`}>
        {typeof value === "number" ? `${sign}KES ${value.toFixed(2)}` : value}
      </span>
    </div>
  );
}

function SkeletonCard() {
  return (
    <div className="rounded-2xl border border-gray-100 bg-white p-6 shadow-sm space-y-4 animate-pulse">
      <div className="h-4 w-32 bg-gray-100 rounded" />
      <div className="h-10 w-48 bg-gray-100 rounded" />
      <div className="h-2 w-full bg-gray-100 rounded" />
      <div className="space-y-2">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="h-4 w-full bg-gray-50 rounded" />
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function MLPricingCard({
  prediction,
  loading,
  error,
  mlOnline,
  onReestimate,
}) {
  if (loading) return <SkeletonCard />;

  if (error) {
    return (
      <div className="rounded-2xl border border-red-100 bg-red-50 p-6 text-center">
        <p className="text-sm text-red-600 font-medium mb-3">⚠️ {error}</p>
        {onReestimate && (
          <button
            onClick={onReestimate}
            className="text-xs text-red-500 underline hover:text-red-700"
          >
            Try again
          </button>
        )}
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="rounded-2xl border border-dashed border-gray-200 bg-gray-50 p-8 text-center text-sm text-gray-400">
        Fill in the details above to get an ML price prediction.
      </div>
    );
  }

  const {
    predicted_price_per_kg: pricePerKg,
    total_estimated_price: totalPrice,
    confidence_interval: ci,
    price_factors: factors,
    waste_type: wasteType,
    weight_kg: weightKg,
    model_version: modelVersion,
    _isFallback,
  } = prediction;

  const isFallback = _isFallback || mlOnline === false;

  return (
    <div className="rounded-2xl border border-gray-100 bg-white shadow-sm overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-6 pt-5 pb-4 border-b border-gray-50">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-0.5">
            Price Prediction
          </p>
          <p className="text-xs text-gray-400 capitalize">
            {wasteType.replace("_", " ")} · {weightKg} kg
          </p>
        </div>
        <Badge online={isFallback ? false : mlOnline} />
      </div>

      {/* Price hero */}
      <div className="px-6 py-5 bg-gradient-to-br from-emerald-50 to-teal-50">
        <div className="flex items-end gap-3">
          <div>
            <p className="text-4xl font-bold text-emerald-700 tabular-nums">
              KES {pricePerKg.toFixed(2)}
              <span className="text-base font-normal text-emerald-500">/kg</span>
            </p>
            <p className="text-sm text-emerald-600 mt-0.5 font-medium">
              Total: KES {totalPrice.toLocaleString("en-KE", { minimumFractionDigits: 2 })}
            </p>
          </div>
        </div>

        {/* Confidence bar */}
        {ci && (
          <div className="mt-4">
            <ConfidenceBar low={ci.low} high={ci.high} predicted={pricePerKg} />
          </div>
        )}
      </div>

      {/* Price factor breakdown */}
      {factors && (
        <div className="px-6 py-4">
          <p className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-3">
            Price Breakdown
          </p>
          <FactorRow label="Base market price" value={factors.base_price_kes} />
          <FactorRow
            label="Quality adjustment"
            value={factors.quality_adjustment}
            color={factors.quality_adjustment >= 0 ? "text-emerald-600" : "text-red-500"}
          />
          <FactorRow
            label="Distance penalty"
            value={factors.distance_penalty}
            color="text-red-500"
          />
          <FactorRow
            label="Seasonality factor"
            value={`×${factors.seasonality_factor}`}
            color="text-blue-600"
          />
          <FactorRow
            label="Market demand boost"
            value={factors.demand_boost}
            color={factors.demand_boost >= 0 ? "text-emerald-600" : "text-red-500"}
          />
          <FactorRow
            label="Weight multiplier"
            value={`×${factors.weight_multiplier}`}
            color="text-gray-700"
          />
        </div>
      )}

      {/* Footer */}
      <div className="flex items-center justify-between px-6 py-3 bg-gray-50 border-t border-gray-100">
        <p className="text-xs text-gray-400">
          {isFallback ? "Rule-based estimate" : `Model ${modelVersion}`}
        </p>
        {onReestimate && (
          <button
            onClick={onReestimate}
            className="text-xs font-medium text-emerald-600 hover:text-emerald-800 transition-colors"
          >
            Re-estimate ↻
          </button>
        )}
      </div>
    </div>
  );
}
