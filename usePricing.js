/**
 * usePricing.js
 *
 * Custom hook that wraps the ML pricing service.
 * Manages loading / error / result state and exposes a clean API
 * to components — they never import pricingService directly.
 *
 * Usage:
 *   const { predict, prediction, loading, error, mlOnline } = usePricing();
 *
 *   // On form submit:
 *   await predict({ wasteType: "plastic", weightKg: 5, distanceKm: 3, consistencyScore: 0.8 });
 */

import { useState, useEffect, useCallback, useRef } from "react";
import {
  predictPrice,
  predictPriceBatch,
  checkMLHealth,
  getModelInfo,
} from "../services/pricingService";

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function usePricing() {
  // Single prediction state
  const [prediction, setPrediction]   = useState(null);
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState(null);

  // Batch state
  const [batchResults, setBatchResults] = useState([]);
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchError, setBatchError]     = useState(null);

  // ML API health
  const [mlOnline, setMlOnline]         = useState(null);   // null = checking
  const [modelInfo, setModelInfo]       = useState(null);

  // Abort controller for in-flight requests
  const abortRef = useRef(null);

  // ---------------------------------------------------------------------------
  // Check ML API health on mount
  // ---------------------------------------------------------------------------
  useEffect(() => {
    let cancelled = false;

    async function init() {
      const health = await checkMLHealth();
      if (!cancelled) {
        setMlOnline(health.online);
        if (health.online) {
          const info = await getModelInfo();
          if (!cancelled) setModelInfo(info);
        }
      }
    }

    init();
    return () => { cancelled = true; };
  }, []);

  // ---------------------------------------------------------------------------
  // Single prediction
  // ---------------------------------------------------------------------------

  /**
   * @param {object} params
   * @param {string} params.wasteType
   * @param {number} params.weightKg
   * @param {number} params.distanceKm
   * @param {number} params.consistencyScore
   * @param {number} [params.month]
   * @param {number} [params.dayOfWeek]
   * @param {string} [params.centreId]
   * @param {number} [params.marketDemandIndex]
   */
  const predict = useCallback(async (params) => {
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const result = await predictPrice(params);
      setPrediction(result);
      return result;
    } catch (err) {
      const msg = err?.message || "Price prediction failed. Please try again.";
      setError(msg);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  // ---------------------------------------------------------------------------
  // Batch prediction
  // ---------------------------------------------------------------------------

  /**
   * @param {object[]} items — array of param objects (same shape as predict())
   */
  const predictBatch = useCallback(async (items) => {
    setBatchLoading(true);
    setBatchError(null);
    setBatchResults([]);

    try {
      const results = await predictPriceBatch(items);
      setBatchResults(results);
      return results;
    } catch (err) {
      const msg = err?.message || "Batch prediction failed.";
      setBatchError(msg);
      return [];
    } finally {
      setBatchLoading(false);
    }
  }, []);

  // ---------------------------------------------------------------------------
  // Reset helpers
  // ---------------------------------------------------------------------------

  const clearPrediction = useCallback(() => {
    setPrediction(null);
    setError(null);
  }, []);

  const clearBatch = useCallback(() => {
    setBatchResults([]);
    setBatchError(null);
  }, []);

  // ---------------------------------------------------------------------------
  // Derived helpers for components
  // ---------------------------------------------------------------------------

  /** Is this prediction from the ML model (true) or the fallback (false)? */
  const isMLPrediction = prediction ? !prediction._isFallback : null;

  /** Human-readable confidence range string, e.g. "KES 19.50 – 26.10" */
  const confidenceRangeLabel = prediction?.confidence_interval
    ? `KES ${prediction.confidence_interval.low.toFixed(2)} – ${prediction.confidence_interval.high.toFixed(2)}`
    : null;

  return {
    // Single prediction
    predict,
    prediction,
    loading,
    error,
    clearPrediction,
    isMLPrediction,
    confidenceRangeLabel,

    // Batch
    predictBatch,
    batchResults,
    batchLoading,
    batchError,
    clearBatch,

    // ML API status
    mlOnline,
    modelInfo,
  };
}
