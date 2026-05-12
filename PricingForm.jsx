/**
 * PricingForm.jsx
 *
 * Form that collects waste details and triggers the ML price prediction.
 * Expects a `consistencyScore` prop from the parent (passed in by the
 * vision module after image analysis).
 *
 * Usage:
 *   <PricingForm consistencyScore={visionResult?.score ?? 0.75} distanceKm={userDistance} />
 */

import React, { useState } from "react";
import { usePricing } from "../../hooks/usePricing";
import MLPricingCard from "./MLPricingCard";

const WASTE_TYPES = [
  { value: "plastic",  label: "♻️ Plastic" },
  { value: "paper",    label: "📄 Paper" },
  { value: "metal",    label: "🔩 Metal" },
  { value: "glass",    label: "🪟 Glass" },
  { value: "e_waste",  label: "💻 E-Waste" },
  { value: "organic",  label: "🌿 Organic" },
  { value: "textile",  label: "👕 Textile" },
  { value: "rubber",   label: "🔘 Rubber" },
];

export default function PricingForm({ consistencyScore = 0.75, distanceKm = 5 }) {
  const [wasteType, setWasteType] = useState("plastic");
  const [weightKg, setWeightKg]   = useState("");

  const { predict, prediction, loading, error, mlOnline, clearPrediction } = usePricing();

  const handleSubmit = async (e) => {
    e.preventDefault();
    const weight = parseFloat(weightKg);
    if (!weight || weight <= 0) return;

    await predict({
      wasteType,
      weightKg: weight,
      distanceKm,
      consistencyScore,
    });
  };

  return (
    <div className="space-y-6">
      {/* Form */}
      <div className="rounded-2xl border border-gray-100 bg-white shadow-sm p-6">
        <h2 className="text-base font-semibold text-gray-800 mb-4">Estimate Price</h2>

        <div className="space-y-4">
          {/* Waste type */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1.5">
              Waste Type
            </label>
            <select
              value={wasteType}
              onChange={(e) => { setWasteType(e.target.value); clearPrediction(); }}
              className="w-full rounded-xl border border-gray-200 bg-gray-50 px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500"
            >
              {WASTE_TYPES.map((t) => (
                <option key={t.value} value={t.value}>{t.label}</option>
              ))}
            </select>
          </div>

          {/* Weight */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1.5">
              Weight (kg)
            </label>
            <input
              type="number"
              min="0.1"
              step="0.1"
              value={weightKg}
              onChange={(e) => { setWeightKg(e.target.value); clearPrediction(); }}
              placeholder="e.g. 12.5"
              className="w-full rounded-xl border border-gray-200 bg-gray-50 px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500"
            />
          </div>

          {/* Readonly info */}
          <div className="flex gap-3">
            <div className="flex-1 rounded-xl bg-gray-50 border border-gray-200 px-3 py-2.5">
              <p className="text-xs text-gray-400">Vision score</p>
              <p className="text-sm font-medium text-gray-700">
                {(consistencyScore * 100).toFixed(0)}%
              </p>
            </div>
            <div className="flex-1 rounded-xl bg-gray-50 border border-gray-200 px-3 py-2.5">
              <p className="text-xs text-gray-400">Distance</p>
              <p className="text-sm font-medium text-gray-700">{distanceKm} km</p>
            </div>
          </div>

          <button
            onClick={handleSubmit}
            disabled={loading || !weightKg}
            className="w-full py-3 rounded-xl bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-semibold transition-colors"
          >
            {loading ? "Calculating…" : "Get ML Price"}
          </button>
        </div>
      </div>

      {/* Result card */}
      <MLPricingCard
        prediction={prediction}
        loading={loading}
        error={error}
        mlOnline={mlOnline}
        onReestimate={prediction ? handleSubmit : null}
      />
    </div>
  );
}
