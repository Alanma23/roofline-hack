import { computeSizing } from "./rooflineSizing";

function toPositiveInt(value, fallback) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return fallback;
  return Math.max(1, Math.round(n));
}

function compactNote(result) {
  const bottleneck = result.bottleneck;
  const required = result.required_to_debottleneck || {};
  if (bottleneck === "network") {
    if (required.intranode_bw_gbs) {
      return `EP network-bound; ~${required.intranode_bw_gbs.toFixed(1)} GB/s intranode needed`;
    }
    if (required.internode_bw_gbs) {
      return `EP cross-node bottleneck; ~${required.internode_bw_gbs.toFixed(1)} GB/s internode needed`;
    }
    if (required.network_bw_gbs) {
      return `Network-bound; ~${required.network_bw_gbs.toFixed(1)} GB/s needed to debottleneck`;
    }
  }
  if (bottleneck === "memory" && required.mem_bw_gbs) {
    return `Memory-bound; ~${required.mem_bw_gbs.toFixed(1)} GB/s DRAM BW needed`;
  }
  if (bottleneck === "compute" && required.peak_tflops) {
    return `Compute-bound; ~${required.peak_tflops.toFixed(1)} TFLOPS needed`;
  }
  return `${bottleneck[0].toUpperCase()}${bottleneck.slice(1)}-bound`;
}

/**
 * Flag Pareto-optimal rows: a row is Pareto-optimal if no other row dominates it
 * on both latency (lower is better) and ASIC count (lower is better).
 */
function flagParetoOptimal(rows) {
  return rows.map((row, i) => {
    const dominated = rows.some((other, j) => {
      if (i === j) return false;
      return other.latency_ms <= row.latency_ms && other.asics <= row.asics
        && (other.latency_ms < row.latency_ms || other.asics < row.asics);
    });
    return { ...row, pareto_optimal: !dominated };
  });
}

export function recommendSizingConfigs(baseRequest, options = {}) {
  const tpCandidates = options.tp_candidates || [1, 2, 4, 8];
  const ppCandidates = options.pp_candidates || [1, 2, 4];
  const epCandidates = options.ep_candidates || [1, 2, 4, 8];
  const topK = toPositiveInt(options.top_k, 3);
  const maxAsics = toPositiveInt(baseRequest?.parallel?.max_asics, 16);

  const ranked = [];

  for (const tpRaw of tpCandidates) {
    const tp = toPositiveInt(tpRaw, 1);
    for (const ppRaw of ppCandidates) {
      const pp = toPositiveInt(ppRaw, 1);
      for (const epRaw of epCandidates) {
        const ep = toPositiveInt(epRaw, 1);
        // Total ASICs = tp * pp (ep shares the same pool via remapping)
        const asics = tp * pp;
        if (asics > maxAsics) continue;

        const req = {
          ...baseRequest,
          parallel: {
            ...(baseRequest.parallel || {}),
            tp,
            pp,
            ep,
            max_asics: maxAsics,
          },
        };

        const result = computeSizing(req, options);
        ranked.push({
          tp,
          pp,
          ep,
          asics,
          latency_ms: result.time.end_to_end_ms,
          bottleneck: result.bottleneck,
          note: compactNote(result),
          tokens_per_s: result.time.tokens_per_s || 0,
          ep_time_ms: result.metadata?.network_breakdown_ms?.ep_time_ms || 0,
          ep_uses_internode: result.metadata?.network?.ep_uses_internode || false,
        });
      }
    }
  }

  ranked.sort((a, b) => (
    (a.latency_ms - b.latency_ms)
    || (a.asics - b.asics)
    || (b.tokens_per_s - a.tokens_per_s)
  ));

  const withPareto = flagParetoOptimal(ranked);

  return withPareto.slice(0, topK);
}
