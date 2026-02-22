function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function toPositiveNumber(value, fallback) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return fallback;
  return n;
}

function toPositiveInt(value, fallback) {
  return Math.max(1, Math.round(toPositiveNumber(value, fallback)));
}

export function ringAllReduceBytes(tensorBytes, tp) {
  const rankCount = toPositiveInt(tp, 1);
  if (rankCount <= 1) return 0;
  const bytes = toPositiveNumber(tensorBytes, 0);
  return (2 * (rankCount - 1) / rankCount) * bytes;
}

export function ringAllReduceTimeSeconds(tensorBytes, tp, linkBwGBs, latencyUs) {
  const rankCount = toPositiveInt(tp, 1);
  if (rankCount <= 1) {
    return { total_s: 0, bytes: 0, latency_s: 0, bandwidth_s: 0 };
  }

  const bytes = ringAllReduceBytes(tensorBytes, rankCount);
  const latencyS = 2 * (rankCount - 1) * toPositiveNumber(latencyUs, 0) * 1e-6;
  const bw = toPositiveNumber(linkBwGBs, 0);
  const bandwidthS = bw > 0 ? bytes / (bw * 1e9) : Number.POSITIVE_INFINITY;

  return {
    total_s: latencyS + bandwidthS,
    bytes,
    latency_s: latencyS,
    bandwidth_s: bandwidthS,
  };
}

export function p2pSendTimeSeconds(bytes, linkBwGBs, latencyUs) {
  const payload = toPositiveNumber(bytes, 0);
  if (payload <= 0) {
    return { total_s: 0, bytes: 0, latency_s: 0, bandwidth_s: 0 };
  }
  const latencyS = toPositiveNumber(latencyUs, 0) * 1e-6;
  const bw = toPositiveNumber(linkBwGBs, 0);
  const bandwidthS = bw > 0 ? payload / (bw * 1e9) : Number.POSITIVE_INFINITY;
  return {
    total_s: latencyS + bandwidthS,
    bytes: payload,
    latency_s: latencyS,
    bandwidth_s: bandwidthS,
  };
}

export function partitionLayers(layerCount, pp) {
  const L = Math.max(0, Math.round(Number(layerCount) || 0));
  const stageCount = toPositiveInt(pp, 1);
  const base = stageCount > 0 ? Math.floor(L / stageCount) : 0;
  const remainder = stageCount > 0 ? (L % stageCount) : 0;

  const stages = [];
  const stageOfLayer = Array(L).fill(0);
  let cursor = 0;

  for (let stage = 0; stage < stageCount; stage += 1) {
    const count = base + (stage < remainder ? 1 : 0);
    const start = cursor;
    const end = count > 0 ? (cursor + count - 1) : (cursor - 1);

    stages.push({ stage, start, end, count });
    for (let i = start; i <= end; i += 1) {
      if (i >= 0 && i < L) stageOfLayer[i] = stage;
    }

    cursor += count;
  }

  return { stageOfLayer, stages };
}

export function computeParallelNetwork(workload, layerRecords, parallel = {}, network = {}) {
  const records = Array.isArray(layerRecords) ? layerRecords : [];
  const layerCount = records.length;

  const tp = toPositiveInt(parallel.tp, 1);
  const pp = toPositiveInt(parallel.pp, 1);
  const maxAsics = toPositiveInt(parallel.max_asics, 16);

  const tpLinkBw = toPositiveNumber(network.tp_link_bw_gbs, 900);
  const tpLinkLatencyUs = toPositiveNumber(network.tp_link_latency_us, 3);
  const ppLinkBw = toPositiveNumber(network.pp_link_bw_gbs, tpLinkBw);
  const ppLinkLatencyUs = toPositiveNumber(network.pp_link_latency_us, tpLinkLatencyUs);
  const overlapFraction = clamp(Number(network.overlap_fraction ?? 0), 0, 1);

  const partition = partitionLayers(layerCount, pp);
  const stageByLayer = partition.stageOfLayer;
  const stages = partition.stages;

  let tpAllreduceCount = 0;
  let tpAllreduceBytes = 0;
  let tpTimeS = 0;
  let tpLatencyS = 0;
  let tpBandwidthS = 0;

  let ppSendCount = 0;
  let ppSendBytes = 0;
  let ppTimeS = 0;
  let ppLatencyS = 0;
  let ppBandwidthS = 0;

  const boundaryLayer = new Set();
  for (let i = 0; i < stages.length - 1; i += 1) {
    const stage = stages[i];
    if (stage.count > 0 && stage.end >= 0) boundaryLayer.add(stage.end);
  }

  const layerCollective = records.map((layer, index) => {
    const stage = stageByLayer[index] ?? 0;
    const syncPoints = Array.isArray(layer.tp_sync_points) ? layer.tp_sync_points : [];

    let layerTPBytes = 0;
    let layerTPTimeS = 0;
    let layerTPLatencyS = 0;
    let layerTPBandwidthS = 0;

    for (const sync of syncPoints) {
      const t = ringAllReduceTimeSeconds(sync.tensor_bytes, tp, tpLinkBw, tpLinkLatencyUs);
      layerTPBytes += t.bytes;
      layerTPTimeS += t.total_s;
      layerTPLatencyS += t.latency_s;
      layerTPBandwidthS += t.bandwidth_s;
    }

    tpAllreduceCount += syncPoints.length;
    tpAllreduceBytes += layerTPBytes;
    tpTimeS += layerTPTimeS;
    tpLatencyS += layerTPLatencyS;
    tpBandwidthS += layerTPBandwidthS;

    let layerPPBytes = 0;
    let layerPPTimeS = 0;
    let layerPPLatencyS = 0;
    let layerPPBandwidthS = 0;

    if (boundaryLayer.has(index) && pp > 1) {
      const send = p2pSendTimeSeconds(layer.output_bytes, ppLinkBw, ppLinkLatencyUs);
      layerPPBytes += send.bytes;
      layerPPTimeS += send.total_s;
      layerPPLatencyS += send.latency_s;
      layerPPBandwidthS += send.bandwidth_s;

      ppSendCount += 1;
      ppSendBytes += send.bytes;
      ppTimeS += send.total_s;
      ppLatencyS += send.latency_s;
      ppBandwidthS += send.bandwidth_s;
    }

    return {
      layer: index,
      stage,
      tp_sync_count: syncPoints.length,
      tp_sync_bytes: layerTPBytes,
      tp_time_ms: layerTPTimeS * 1e3,
      pp_boundary_send_count: layerPPBytes > 0 ? 1 : 0,
      pp_boundary_send_bytes: layerPPBytes,
      pp_time_ms: layerPPTimeS * 1e3,
    };
  });

  return {
    parallel: { tp, pp, max_asics: maxAsics },
    resolved_network: {
      tp_link_bw_gbs: tpLinkBw,
      tp_link_latency_us: tpLinkLatencyUs,
      pp_link_bw_gbs: ppLinkBw,
      pp_link_latency_us: ppLinkLatencyUs,
      overlap_fraction: overlapFraction,
    },
    stage_layout: stages,
    layer_collective: layerCollective,
    totals: {
      tp_allreduce_count: tpAllreduceCount,
      tp_allreduce_bytes: tpAllreduceBytes,
      pp_send_count: ppSendCount,
      pp_send_bytes: ppSendBytes,
      tp_time_ms: tpTimeS * 1e3,
      pp_time_ms: ppTimeS * 1e3,
      tp_latency_ms: tpLatencyS * 1e3,
      pp_latency_ms: ppLatencyS * 1e3,
      tp_bandwidth_ms: tpBandwidthS * 1e3,
      pp_bandwidth_ms: ppBandwidthS * 1e3,
      network_time_ms: (tpTimeS + ppTimeS) * 1e3,
    },
    workload_phase: workload?.phase === "prefill" ? "prefill" : "decode",
  };
}

