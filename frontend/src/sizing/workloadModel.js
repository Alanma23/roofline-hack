const DEFAULT_BPE_MAP = {
  FP32: 4.0,
  TF32: 4.0,
  FP16: 2.0,
  BF16: 2.0,
  FP8_E4M3: 1.0,
  FP8_E5M2: 1.0,
  INT8: 1.0,
  INT4: 0.5,
  NVFP4: 0.56640625,
  NVFP4_KV: 0.56640625,
  MXFP4: 0.53125,
  MXFP8_E4M3: 1.03125,
  MXFP8_E5M2: 1.03125,
  MXFP6_E3M2: 0.78125,
  MXFP6_E2M3: 0.78125,
  NF4: 0.53125,
};

function toPositiveInt(value, fallback) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return fallback;
  return Math.max(1, Math.round(n));
}

function defaultBytesPerElement(format) {
  return DEFAULT_BPE_MAP[format] ?? 2.0;
}

export function resolveTokenCount(workload) {
  const seqLen = toPositiveInt(workload?.seq_len, 1);
  if (workload?.phase === "prefill") {
    return toPositiveInt(workload?.prefill_tokens, seqLen);
  }
  return toPositiveInt(workload?.decode_tokens, 1);
}

export function aggregateOps(ops) {
  const grouped = {};
  for (const op of ops) {
    if (!grouped[op.name]) grouped[op.name] = { ...op, count: 0 };
    else {
      grouped[op.name].flops += op.flops;
      grouped[op.name].bytes += op.bytes;
    }
    grouped[op.name].count += 1;
    grouped[op.name].ai = grouped[op.name].bytes > 0
      ? grouped[op.name].flops / grouped[op.name].bytes
      : 0.0;
  }
  return Object.values(grouped);
}

export function computeWorkloadModel(workload, bytesPerElement = defaultBytesPerElement) {
  const model = workload?.model || {};
  const precision = workload?.precision || {};
  const moe = workload?.moe || null;

  const L = toPositiveInt(model.L, 1);
  const H = toPositiveInt(model.H, 1);
  const nh = toPositiveInt(model.nh, 1);
  const nkv = toPositiveInt(model.nkv, 1);
  const dh = toPositiveInt(model.dh, Math.max(1, Math.floor(H / nh)));
  const dff = toPositiveInt(model.dff, Math.max(1, Math.floor((8 * H) / 3)));
  const V = toPositiveInt(model.V, H);
  const gate = Boolean(model.gate ?? true);

  const B = toPositiveInt(workload?.batch, 1);
  const S = toPositiveInt(workload?.seq_len, 1);
  const T = resolveTokenCount(workload);

  const wB = Number(bytesPerElement(precision.w)) || 2.0;
  const aB = Number(bytesPerElement(precision.a)) || 2.0;
  const kvB = Number(bytesPerElement(precision.kv)) || aB;
  const oB = aB;
  const dkv = nkv * dh;
  const cp = precision.computeAs || "FP16";

  // MoE parameters (validated)
  const hasMoe = moe != null;
  const numExperts = hasMoe ? toPositiveInt(moe.num_experts, 8) : 0;
  const expertsPerToken = hasMoe ? toPositiveInt(moe.experts_per_token, 2) : 0;
  const expertFfnDim = hasMoe ? toPositiveInt(moe.expert_ffn_dim, dff) : dff;
  const capacityFactor = hasMoe ? Math.max(1.0, Number(moe.capacity_factor) || 1.25) : 1.0;
  const moeLayerFreq = hasMoe ? toPositiveInt(moe.moe_layer_freq, 1) : 1;

  const ops = [];
  const layers = [];
  const tokenActivationElements = B * T * H;
  const tokenActivationBytes = tokenActivationElements * aB;

  for (let layer = 0; layer < L; layer += 1) {
    const layerOps = [];
    let layerWeightBytes = 0;

    const addOp = (op) => {
      const full = {
        ...op,
        layer,
        ai: op.bytes > 0 ? op.flops / op.bytes : 0.0,
        cp,
      };
      layerOps.push(full);
      ops.push(full);
    };

    const addGemm = (name, M, N, K) => {
      const flops = 2 * M * N * K;
      const bytes = M * K * aB + K * N * wB + M * N * oB;
      const weightBytes = K * N * wB;
      layerWeightBytes += weightBytes;
      addOp({ name, type: "gemm", flops, bytes, weight_bytes: weightBytes });
    };

    const addAttention = (name, Sq, Skv, isScore) => {
      const flops = 2 * B * nh * Sq * Skv * dh;
      const bytes = isScore
        ? B * nh * Sq * dh * aB + B * nkv * Skv * dh * kvB + B * nh * Sq * Skv * aB
        : B * nh * Sq * Skv * aB + B * nkv * Skv * dh * kvB + B * nh * Sq * dh * oB;
      addOp({ name, type: "attention", flops, bytes, weight_bytes: 0 });
    };

    const addElementwise = (name, flops, bytes) => {
      addOp({ name, type: "elementwise", flops, bytes, weight_bytes: 0 });
    };

    // Attention sub-layers (always dense)
    addGemm("q_proj", B * T, H, H);
    addGemm("k_proj", B * T, dkv, H);
    addGemm("v_proj", B * T, dkv, H);
    addAttention("qk_score", T, S, true);
    addElementwise("softmax", 5 * B * nh * T * S, 2 * B * nh * T * S * aB);
    addAttention("sv_prod", T, S, false);
    addGemm("o_proj", B * T, H, H);
    addElementwise("rmsnorm", 5 * B * T * H, 2 * B * T * H * aB);

    // Determine if this is a MoE FFN layer
    const isMoeLayer = hasMoe && (layer % moeLayerFreq === 0);

    if (isMoeLayer) {
      // Router: small GEMM from hidden→num_experts
      addGemm("router", B * T, numExperts, H);

      // Effective tokens dispatched to each expert (accounting for capacity and top-K routing)
      const effectiveTokens = Math.max(1, Math.round(
        B * T * (expertsPerToken / numExperts) * capacityFactor,
      ));

      // Expert FFN ops (SwiGLU style within experts)
      if (gate) {
        addGemm("expert_gate", effectiveTokens, expertFfnDim, H);
        addGemm("expert_up", effectiveTokens, expertFfnDim, H);
        addElementwise(
          "expert_silu_mul",
          3 * effectiveTokens * expertFfnDim,
          3 * effectiveTokens * expertFfnDim * aB,
        );
      } else {
        addGemm("expert_up", effectiveTokens, expertFfnDim, H);
      }
      addGemm("expert_down", effectiveTokens, H, expertFfnDim);
    } else {
      // Dense FFN
      if (gate) {
        addGemm("gate_proj", B * T, dff, H);
        addGemm("up_proj", B * T, dff, H);
        addElementwise("silu_mul", 3 * B * T * dff, 3 * B * T * dff * aB);
      } else {
        addGemm("up_proj", B * T, dff, H);
      }
      addGemm("down_proj", B * T, H, dff);
    }

    addElementwise("residual", 2 * B * T * H, 6 * B * T * H * aB);

    // Expert sync points for EP all-to-all (only on MoE layers)
    const expertSyncPoints = isMoeLayer ? [
      { name: "expert_dispatch", tensor_bytes: tokenActivationBytes },
      { name: "expert_gather",   tensor_bytes: tokenActivationBytes },
    ] : [];

    layers.push({
      layer,
      is_moe_layer: isMoeLayer,
      input_bytes: tokenActivationBytes,
      output_bytes: tokenActivationBytes,
      weight_bytes: layerWeightBytes,
      tp_sync_bytes: 2 * tokenActivationBytes,
      tp_sync_points: [
        {
          name: "attn_out",
          tensor_elements: tokenActivationElements,
          tensor_bytes: tokenActivationBytes,
        },
        {
          name: "mlp_down",
          tensor_elements: tokenActivationElements,
          tensor_bytes: tokenActivationBytes,
        },
      ],
      expert_sync_points: expertSyncPoints,
      ops: layerOps,
    });
  }

  const logitFlops = 2 * B * T * V * H;
  const logitBytes = B * T * H * aB + H * V * wB + B * T * V * oB;
  ops.push({
    name: "logit_proj",
    type: "gemm",
    flops: logitFlops,
    bytes: logitBytes,
    ai: logitBytes > 0 ? logitFlops / logitBytes : 0.0,
    cp,
    layer: -1,
    weight_bytes: H * V * wB,
  });

  const totalFlops = ops.reduce((sum, op) => sum + op.flops, 0);
  const totalBytes = ops.reduce((sum, op) => sum + op.bytes, 0);

  return {
    phase: workload?.phase === "prefill" ? "prefill" : "decode",
    batch: B,
    seq_len: S,
    token_count: T,
    model: { L, H, nh, nkv, dh, dff, V, gate },
    moe: hasMoe ? { numExperts, expertsPerToken, expertFfnDim, capacityFactor, moeLayerFreq } : null,
    precision: {
      w: precision.w || "FP16",
      a: precision.a || "FP16",
      kv: precision.kv || "FP16",
      computeAs: cp,
      bytes_per_element: { w: wB, a: aB, kv: kvB },
    },
    ops,
    aggregate_ops: aggregateOps(ops),
    layers,
    totals: {
      flops: totalFlops,
      bytes: totalBytes,
      ai: totalBytes > 0 ? totalFlops / totalBytes : 0.0,
    },
  };
}
