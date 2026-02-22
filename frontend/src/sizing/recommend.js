import { computeSizing } from "./rooflineSizing";

function toPositiveInt(value, fallback) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return fallback;
  return Math.max(1, Math.round(n));
}

function compactNote(result) {
  const bottleneck = result.bottleneck;
  const required = result.required_to_debottleneck || {};
  if (bottleneck === "network" && required.network_bw_gbs) {
    return `Network-bound; ~${required.network_bw_gbs.toFixed(1)} GB/s needed to debottleneck`;
  }
  if (bottleneck === "memory" && required.mem_bw_gbs) {
    return `Memory-bound; ~${required.mem_bw_gbs.toFixed(1)} GB/s DRAM BW needed`;
  }
  if (bottleneck === "compute" && required.peak_tflops) {
    return `Compute-bound; ~${required.peak_tflops.toFixed(1)} TFLOPS needed`;
  }
  return `${bottleneck[0].toUpperCase()}${bottleneck.slice(1)}-bound`;
}

export function recommendSizingConfigs(baseRequest, options = {}) {
  const tpCandidates = options.tp_candidates || [1, 2, 4, 8];
  const ppCandidates = options.pp_candidates || [1, 2, 4];
  const topK = toPositiveInt(options.top_k, 3);
  const maxAsics = toPositiveInt(baseRequest?.parallel?.max_asics, 16);

  const ranked = [];

  for (const tpRaw of tpCandidates) {
    const tp = toPositiveInt(tpRaw, 1);
    for (const ppRaw of ppCandidates) {
      const pp = toPositiveInt(ppRaw, 1);
      const asics = tp * pp;
      if (asics > maxAsics) continue;

      const req = {
        ...baseRequest,
        parallel: {
          ...(baseRequest.parallel || {}),
          tp,
          pp,
          max_asics: maxAsics,
        },
      };

      const result = computeSizing(req, options);
      ranked.push({
        tp,
        pp,
        asics,
        latency_ms: result.time.end_to_end_ms,
        bottleneck: result.bottleneck,
        note: compactNote(result),
        tokens_per_s: result.time.tokens_per_s || 0,
      });
    }
  }

  ranked.sort((a, b) => (
    (a.latency_ms - b.latency_ms)
    || (a.asics - b.asics)
    || (b.tokens_per_s - a.tokens_per_s)
  ));

  return ranked.slice(0, topK);
}

