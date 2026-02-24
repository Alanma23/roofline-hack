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

/**
 * All-to-all bytes exchanged for EP dispatch/gather.
 * Each rank sends (ep-1)/ep * tensorBytes and receives (ep-1)/ep * tensorBytes.
 * Total bytes moved = 2 * (ep-1)/ep * tensorBytes.
 */
export function allToAllBytes(tensorBytes, ep) {
  const rankCount = toPositiveInt(ep, 1);
  if (rankCount <= 1) return 0;
  const bytes = toPositiveNumber(tensorBytes, 0);
  return 2 * ((rankCount - 1) / rankCount) * bytes;
}

/**
 * All-to-all time for EP dispatch/gather.
 * Latency model: log2(ep) hops × latencyUs (bisection tree).
 * Bandwidth model: bytes / (bwGBs * 1e9).
 */
export function allToAllTimeSeconds(tensorBytes, ep, bwGBs, latencyUs) {
  const rankCount = toPositiveInt(ep, 1);
  if (rankCount <= 1) {
    return { total_s: 0, bytes: 0, latency_s: 0, bandwidth_s: 0 };
  }

  const bytes = allToAllBytes(tensorBytes, rankCount);
  // log2(ep) hops for bisection reduction
  const hops = Math.max(1, Math.ceil(Math.log2(rankCount)));
  const latencyS = hops * toPositiveNumber(latencyUs, 0) * 1e-6;
  const bw = toPositiveNumber(bwGBs, 0);
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
  const ep = toPositiveInt(parallel.ep, 1);
  const maxAsics = toPositiveInt(parallel.max_asics, 16);

  // Network topology: support both legacy flat BW and intranode/internode split
  const tpLinkBw = toPositiveNumber(network.tp_link_bw_gbs, 900);
  const tpLinkLatencyUs = toPositiveNumber(network.tp_link_latency_us, 3);
  const ppLinkBw = toPositiveNumber(network.pp_link_bw_gbs, tpLinkBw);
  const ppLinkLatencyUs = toPositiveNumber(network.pp_link_latency_us, tpLinkLatencyUs);
  const overlapFraction = clamp(Number(network.overlap_fraction ?? 0), 0, 1);

  // EP-specific network params (intranode = NVLink, internode = IB)
  const gpusPerNode = toPositiveInt(network.gpus_per_node, 8);
  const intranodeBw = toPositiveNumber(network.intranode_bw_gbs, tpLinkBw);
  const intranodeLatUs = toPositiveNumber(network.intranode_lat_us, 1);
  const internodeBw = toPositiveNumber(network.internode_bw_gbs, 25);
  const internodeLatUs = toPositiveNumber(network.internode_lat_us, 5);

  // Determine EP all-to-all BW: intranode if ep fits in one node, else internode
  const epBw = ep <= gpusPerNode ? intranodeBw : internodeBw;
  const epLatUs = ep <= gpusPerNode ? intranodeLatUs : internodeLatUs;

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

  let epAlltoallCount = 0;
  let epAlltoallBytes = 0;
  let epTimeS = 0;
  let epLatencyS = 0;
  let epBandwidthS = 0;

  const boundaryLayer = new Set();
  for (let i = 0; i < stages.length - 1; i += 1) {
    const stage = stages[i];
    if (stage.count > 0 && stage.end >= 0) boundaryLayer.add(stage.end);
  }

  const layerCollective = records.map((layer, index) => {
    const stage = stageByLayer[index] ?? 0;
    const syncPoints = Array.isArray(layer.tp_sync_points) ? layer.tp_sync_points : [];
    const expertSyncPoints = Array.isArray(layer.expert_sync_points) ? layer.expert_sync_points : [];

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

    // EP all-to-all (only on MoE layers, only when ep > 1)
    let layerEPBytes = 0;
    let layerEPTimeS = 0;
    let layerEPLatencyS = 0;
    let layerEPBandwidthS = 0;

    if (ep > 1 && expertSyncPoints.length > 0) {
      for (const esp of expertSyncPoints) {
        const t = allToAllTimeSeconds(esp.tensor_bytes, ep, epBw, epLatUs);
        layerEPBytes += t.bytes;
        layerEPTimeS += t.total_s;
        layerEPLatencyS += t.latency_s;
        layerEPBandwidthS += t.bandwidth_s;
      }
      epAlltoallCount += expertSyncPoints.length;
      epAlltoallBytes += layerEPBytes;
      epTimeS += layerEPTimeS;
      epLatencyS += layerEPLatencyS;
      epBandwidthS += layerEPBandwidthS;
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
      ep_alltoall_count: expertSyncPoints.length,
      ep_alltoall_bytes: layerEPBytes,
      ep_time_ms: layerEPTimeS * 1e3,
    };
  });

  return {
    parallel: { tp, pp, ep, max_asics: maxAsics },
    resolved_network: {
      tp_link_bw_gbs: tpLinkBw,
      tp_link_latency_us: tpLinkLatencyUs,
      pp_link_bw_gbs: ppLinkBw,
      pp_link_latency_us: ppLinkLatencyUs,
      overlap_fraction: overlapFraction,
      intranode_bw_gbs: intranodeBw,
      intranode_lat_us: intranodeLatUs,
      internode_bw_gbs: internodeBw,
      internode_lat_us: internodeLatUs,
      gpus_per_node: gpusPerNode,
      ep_uses_internode: ep > gpusPerNode,
    },
    stage_layout: stages,
    layer_collective: layerCollective,
    totals: {
      tp_allreduce_count: tpAllreduceCount,
      tp_allreduce_bytes: tpAllreduceBytes,
      pp_send_count: ppSendCount,
      pp_send_bytes: ppSendBytes,
      ep_alltoall_count: epAlltoallCount,
      ep_alltoall_bytes: epAlltoallBytes,
      tp_time_ms: tpTimeS * 1e3,
      pp_time_ms: ppTimeS * 1e3,
      ep_time_ms: epTimeS * 1e3,
      tp_latency_ms: tpLatencyS * 1e3,
      pp_latency_ms: ppLatencyS * 1e3,
      ep_latency_ms: epLatencyS * 1e3,
      tp_bandwidth_ms: tpBandwidthS * 1e3,
      pp_bandwidth_ms: ppBandwidthS * 1e3,
      ep_bandwidth_ms: epBandwidthS * 1e3,
      network_time_ms: (tpTimeS + ppTimeS + epTimeS) * 1e3,
    },
    workload_phase: workload?.phase === "prefill" ? "prefill" : "decode",
  };
}
