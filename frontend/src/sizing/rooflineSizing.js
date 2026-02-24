import { computeWorkloadModel } from "./workloadModel";
import { computeParallelNetwork } from "./parallelNetwork";

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function defaultHwFlopsKey(computeAs) {
  return computeAs || "FP16";
}

function defaultBytesPerElement() {
  return 2.0;
}

function toPositiveNumber(value, fallback) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return fallback;
  return n;
}

function resolveMemoryModel(hardware = {}) {
  const singleBw = toPositiveNumber(hardware?.mem_bw_gbs ?? hardware?.bw, 1.0);
  const model = hardware?.memory_model || {};
  const mode = model?.mode === "two_tier" ? "two_tier" : "single";

  if (mode !== "two_tier") {
    return {
      mode: "single",
      effective_bw_gbs: singleBw,
      mem_bw_gbs: singleBw,
    };
  }

  const hitRate = clamp(Number(model.onchip_hit_rate ?? 0.8), 0, 1);
  const onchipBw = toPositiveNumber(model.onchip_bw_gbs, singleBw);
  const offchipBw = toPositiveNumber(model.offchip_bw_gbs, singleBw);
  const denom = (hitRate / onchipBw) + ((1 - hitRate) / offchipBw);
  const effBw = denom > 0 ? (1 / denom) : singleBw;

  return {
    mode: "two_tier",
    effective_bw_gbs: effBw,
    mem_bw_gbs: singleBw,
    onchip_bw_gbs: onchipBw,
    offchip_bw_gbs: offchipBw,
    onchip_hit_rate: hitRate,
  };
}

function resolvePeakTflops(hardware, computeAs, hwFlopsKey) {
  const peakMap = hardware?.peak_tflops || hardware?.flops || {};
  const resolvedKey = hwFlopsKey(computeAs);
  return (
    peakMap[resolvedKey]
    ?? peakMap[computeAs]
    ?? peakMap.FP16
    ?? peakMap.BF16
    ?? 1.0
  );
}

function computeRequiredSizing(bottleneck, totals, time, collective, memoryModel, network) {
  const result = {};

  if (bottleneck === "network") {
    const targetS = Math.max(time.compute_s, time.memory_s);
    const fixedLatencyS = (
      (collective.tp_latency_ms || 0)
      + (collective.pp_latency_ms || 0)
      + (collective.ep_latency_ms || 0)
    ) * 1e-3;
    const variableBytes = (
      (collective.tp_allreduce_bytes || 0)
      + (collective.pp_send_bytes || 0)
      + (collective.ep_alltoall_bytes || 0)
    );

    if (targetS > fixedLatencyS && variableBytes > 0) {
      result.network_bw_gbs = variableBytes / (targetS - fixedLatencyS) / 1e9;
    }

    // If EP dominates: report required intranode or internode BW
    if (collective.ep_alltoall_bytes > 0) {
      const epTarget = Math.max(time.compute_s, time.memory_s);
      const epFixed = (collective.ep_latency_ms || 0) * 1e-3;
      if (epTarget > epFixed && collective.ep_alltoall_bytes > 0) {
        const reqBw = collective.ep_alltoall_bytes / (epTarget - epFixed) / 1e9;
        if (network?.ep_uses_internode) {
          result.internode_bw_gbs = reqBw;
        } else {
          result.intranode_bw_gbs = reqBw;
        }
      }
    }
  } else if (bottleneck === "memory") {
    const targetS = Math.max(time.compute_s, time.network_s);
    if (targetS > 0 && totals.bytes > 0) {
      result.mem_bw_gbs = totals.bytes / targetS / 1e9;
      if (memoryModel?.mode === "two_tier") {
        const hitRate = clamp(Number(memoryModel.onchip_hit_rate ?? 0), 0, 1);
        const oneMinus = 1 - hitRate;
        const onchipBw = toPositiveNumber(memoryModel.onchip_bw_gbs, result.mem_bw_gbs);
        const coeff = (targetS * 1e9) / totals.bytes;
        const remain = coeff - (hitRate / onchipBw);
        if (oneMinus > 0 && remain > 0) {
          result.offchip_bw_gbs = oneMinus / remain;
        }
      }
    }
  } else {
    const targetS = Math.max(time.memory_s, time.network_s);
    if (targetS > 0 && totals.flops > 0) {
      result.peak_tflops = totals.flops / targetS / 1e12;
    }
  }
  return result;
}

export function computeSizing(request, options = {}) {
  const bytesPerElement = options.bytesPerElement || defaultBytesPerElement;
  const hwFlopsKey = options.hwFlopsKey || defaultHwFlopsKey;

  const workloadData = computeWorkloadModel(request.workload, bytesPerElement);
  const peakTflops = toPositiveNumber(
    resolvePeakTflops(request.hardware, workloadData.precision.computeAs, hwFlopsKey),
    1.0,
  );
  const memoryModel = resolveMemoryModel(request.hardware);
  const memBwGBs = memoryModel.effective_bw_gbs;

  const opTimings = workloadData.ops.map((op) => {
    const computeS = op.flops / (peakTflops * 1e12);
    const memoryS = op.bytes / (memBwGBs * 1e9);
    const kernelS = Math.max(computeS, memoryS);
    return { ...op, compute_s: computeS, memory_s: memoryS, kernel_s: kernelS };
  });

  const computeS = opTimings.reduce((sum, op) => sum + op.compute_s, 0);
  const memoryS = opTimings.reduce((sum, op) => sum + op.memory_s, 0);
  const kernelS = opTimings.reduce((sum, op) => sum + op.kernel_s, 0);

  const networkData = computeParallelNetwork(
    request.workload,
    workloadData.layers,
    request.parallel,
    request.network,
  );

  const networkS = (networkData.totals.network_time_ms || 0) * 1e-3;
  const overlapFraction = clamp(networkData.resolved_network.overlap_fraction || 0, 0, 1);
  const endToEndS = kernelS + (1 - overlapFraction) * networkS;

  const bottleneckTime = {
    compute: computeS,
    memory: memoryS,
    network: networkS,
  };
  const bottleneck = Object.entries(bottleneckTime).sort((a, b) => b[1] - a[1])[0][0];

  const required = computeRequiredSizing(
    bottleneck,
    workloadData.totals,
    { compute_s: computeS, memory_s: memoryS, network_s: networkS },
    networkData.totals,
    memoryModel,
    networkData.resolved_network,
  );

  const byLayerCollective = new Map(
    networkData.layer_collective.map((entry) => [entry.layer, entry]),
  );

  const layerIO = workloadData.layers.map((layer) => {
    const c = byLayerCollective.get(layer.layer) || {};
    return {
      layer: layer.layer + 1,
      stage: (c.stage ?? 0) + 1,
      is_moe_layer: layer.is_moe_layer || false,
      input_bytes: layer.input_bytes,
      output_bytes: layer.output_bytes,
      weight_bytes: layer.weight_bytes,
      tp_sync_bytes: c.tp_sync_bytes || 0,
      pp_boundary_send_bytes: c.pp_boundary_send_bytes || 0,
      ep_alltoall_bytes: c.ep_alltoall_bytes || 0,
    };
  });

  const criticalAI = peakTflops > 0 && memBwGBs > 0
    ? (peakTflops * 1e12) / (memBwGBs * 1e9)
    : Number.POSITIVE_INFINITY;

  return {
    totals: {
      flops: workloadData.totals.flops,
      bytes: workloadData.totals.bytes,
      ai: workloadData.totals.ai,
    },
    collective: {
      tp_allreduce_count: networkData.totals.tp_allreduce_count,
      tp_allreduce_bytes: networkData.totals.tp_allreduce_bytes,
      pp_send_count: networkData.totals.pp_send_count,
      pp_send_bytes: networkData.totals.pp_send_bytes,
      ep_alltoall_count: networkData.totals.ep_alltoall_count,
      ep_alltoall_bytes: networkData.totals.ep_alltoall_bytes,
    },
    time: {
      compute_ms: computeS * 1e3,
      memory_ms: memoryS * 1e3,
      network_ms: networkS * 1e3,
      kernel_ms: kernelS * 1e3,
      end_to_end_ms: endToEndS * 1e3,
      tokens_per_s: endToEndS > 0 ? workloadData.token_count / endToEndS : 0,
      flops_per_s: endToEndS > 0 ? workloadData.totals.flops / endToEndS : 0,
    },
    bottleneck,
    layer_io: layerIO,
    recommendations: [],
    required_to_debottleneck: required,
    metadata: {
      phase: workloadData.phase,
      token_count: workloadData.token_count,
      critical_ai: criticalAI,
      peak_tflops: peakTflops,
      mem_bw_gbs: memBwGBs,
      memory_model: memoryModel,
      network: networkData.resolved_network,
      stage_layout: networkData.stage_layout,
      op_timings: opTimings,
      aggregate_ops: workloadData.aggregate_ops,
      network_breakdown_ms: {
        tp_time_ms: networkData.totals.tp_time_ms,
        pp_time_ms: networkData.totals.pp_time_ms,
        ep_time_ms: networkData.totals.ep_time_ms,
        tp_latency_ms: networkData.totals.tp_latency_ms,
        pp_latency_ms: networkData.totals.pp_latency_ms,
        ep_latency_ms: networkData.totals.ep_latency_ms,
        tp_bandwidth_ms: networkData.totals.tp_bandwidth_ms,
        pp_bandwidth_ms: networkData.totals.pp_bandwidth_ms,
        ep_bandwidth_ms: networkData.totals.ep_bandwidth_ms,
      },
    },
  };
}
