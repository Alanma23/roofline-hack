import { describe, expect, test } from "vitest";
import { computeSizing } from "./rooflineSizing";

function bytesPerElement(fmt) {
  const map = {
    FP16: 2.0,
    FP8_E4M3: 1.0,
    NVFP4: 0.56640625,
    NVFP4_KV: 0.56640625,
    INT4: 0.5,
  };
  return map[fmt] ?? 2.0;
}

function hwFlopsKey(computeAs) {
  const map = {
    FP16: "FP16",
    BF16: "BF16",
    FP8_E4M3: "FP8_E4M3",
    NVFP4: "NVFP4",
    INT4: "INT4",
  };
  return map[computeAs] || "FP16";
}

function baselineTotals({ model, precision, phase, batch, seqLen }) {
  const B = batch;
  const S = seqLen;
  const T = phase === "prefill" ? S : 1;
  const { L, H, nh, nkv, dh, dff, V, gate } = model;
  const dkv = nkv * dh;
  const wB = bytesPerElement(precision.w);
  const aB = bytesPerElement(precision.a);
  const kvB = bytesPerElement(precision.kv);
  const oB = aB;

  const ops = [];
  const addGemm = (M, N, K) => {
    const flops = 2 * M * N * K;
    const bytes = M * K * aB + K * N * wB + M * N * oB;
    ops.push({ flops, bytes });
  };
  const addAttn = (Sq, Skv, isScore) => {
    const flops = 2 * B * nh * Sq * Skv * dh;
    const bytes = isScore
      ? B * nh * Sq * dh * aB + B * nkv * Skv * dh * kvB + B * nh * Sq * Skv * aB
      : B * nh * Sq * Skv * aB + B * nkv * Skv * dh * kvB + B * nh * Sq * dh * oB;
    ops.push({ flops, bytes });
  };
  const addElem = (flops, bytes) => ops.push({ flops, bytes });

  for (let l = 0; l < L; l += 1) {
    addGemm(B * T, H, H);
    addGemm(B * T, dkv, H);
    addGemm(B * T, dkv, H);
    addAttn(T, S, true);
    addElem(5 * B * nh * T * S, 2 * B * nh * T * S * aB);
    addAttn(T, S, false);
    addGemm(B * T, H, H);
    addElem(5 * B * T * H, 2 * B * T * H * aB);
    if (gate) {
      addGemm(B * T, dff, H);
      addGemm(B * T, dff, H);
      addElem(3 * B * T * dff, 3 * B * T * dff * aB);
    } else {
      addGemm(B * T, dff, H);
    }
    addGemm(B * T, H, dff);
    addElem(2 * B * T * H, 6 * B * T * H * aB);
  }
  addGemm(B * T, V, H);

  const flops = ops.reduce((s, o) => s + o.flops, 0);
  const bytes = ops.reduce((s, o) => s + o.bytes, 0);
  return { flops, bytes };
}

const MODEL = {
  L: 32,
  H: 4096,
  nh: 32,
  nkv: 8,
  dh: 128,
  dff: 14336,
  V: 128256,
  gate: true,
};

const REQUEST_BASE = {
  workload: {
    phase: "decode",
    batch: 1,
    seq_len: 4096,
    model: MODEL,
    precision: { w: "FP16", a: "FP16", kv: "FP16", computeAs: "FP16" },
  },
  hardware: {
    name: "GB10",
    peak_tflops: { FP16: 62, FP8_E4M3: 124, NVFP4: 1000, INT4: 248 },
    mem_bw_gbs: 287,
  },
  parallel: { tp: 1, pp: 1, max_asics: 16 },
  network: { tp_link_bw_gbs: 900, tp_link_latency_us: 3, pp_link_bw_gbs: 900, pp_link_latency_us: 3, overlap_fraction: 0.0 },
};

describe("computeSizing", () => {
  test("matches legacy aggregate FLOPs/bytes within 0.5% when tp=1, pp=1", () => {
    const got = computeSizing(REQUEST_BASE, { bytesPerElement, hwFlopsKey });
    const exp = baselineTotals({
      model: MODEL,
      precision: REQUEST_BASE.workload.precision,
      phase: REQUEST_BASE.workload.phase,
      batch: REQUEST_BASE.workload.batch,
      seqLen: REQUEST_BASE.workload.seq_len,
    });

    const flopsErr = Math.abs(got.totals.flops - exp.flops) / exp.flops;
    const bytesErr = Math.abs(got.totals.bytes - exp.bytes) / exp.bytes;
    expect(flopsErr).toBeLessThan(0.005);
    expect(bytesErr).toBeLessThan(0.005);
    expect(got.collective.tp_allreduce_bytes).toBe(0);
    expect(got.collective.pp_send_bytes).toBe(0);
  });

  test("classifies network bound when link bandwidth is tiny", () => {
    const req = {
      ...REQUEST_BASE,
      parallel: { tp: 8, pp: 2, max_asics: 16 },
      network: {
        tp_link_bw_gbs: 1,
        tp_link_latency_us: 100,
        pp_link_bw_gbs: 1,
        pp_link_latency_us: 100,
        overlap_fraction: 0.0,
      },
    };
    const result = computeSizing(req, { bytesPerElement, hwFlopsKey });
    expect(result.bottleneck).toBe("network");
    if (result.required_to_debottleneck.network_bw_gbs != null) {
      expect(result.required_to_debottleneck.network_bw_gbs).toBeGreaterThan(1);
    } else {
      // Latency-dominated network bottlenecks don't produce a BW-only fix.
      expect(result.time.network_ms).toBeGreaterThan(result.time.compute_ms);
      expect(result.time.network_ms).toBeGreaterThan(result.time.memory_ms);
    }
  });
});
