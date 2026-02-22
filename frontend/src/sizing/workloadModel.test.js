import { describe, expect, test } from "vitest";
import { computeWorkloadModel } from "./workloadModel";

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

const PRECISION = {
  w: "FP16",
  a: "FP16",
  kv: "FP16",
  computeAs: "FP16",
};

function bpe(fmt) {
  const map = {
    FP16: 2.0,
    FP8_E4M3: 1.0,
    INT4: 0.5,
    NVFP4: 0.56640625,
    NVFP4_KV: 0.56640625,
  };
  return map[fmt] ?? 2.0;
}

describe("computeWorkloadModel", () => {
  test("uses T=1 for decode and T=S for prefill", () => {
    const decode = computeWorkloadModel({
      phase: "decode",
      batch: 1,
      seq_len: 4096,
      model: MODEL,
      precision: PRECISION,
    }, bpe);
    const prefill = computeWorkloadModel({
      phase: "prefill",
      batch: 1,
      seq_len: 4096,
      model: MODEL,
      precision: PRECISION,
    }, bpe);

    expect(decode.token_count).toBe(1);
    expect(prefill.token_count).toBe(4096);
    expect(prefill.totals.flops).toBeGreaterThan(decode.totals.flops);
    expect(prefill.totals.bytes).toBeGreaterThan(decode.totals.bytes);
  });

  test("emits two TP sync points per layer with B*T*H tensor size", () => {
    const batch = 2;
    const seqLen = 128;
    const phase = "prefill";
    const r = computeWorkloadModel({
      phase,
      batch,
      seq_len: seqLen,
      model: MODEL,
      precision: PRECISION,
    }, bpe);

    const first = r.layers[0];
    const expectedTensorBytes = batch * seqLen * MODEL.H * bpe("FP16");

    expect(first.tp_sync_points).toHaveLength(2);
    expect(first.tp_sync_points[0].tensor_bytes).toBe(expectedTensorBytes);
    expect(first.tp_sync_points[1].tensor_bytes).toBe(expectedTensorBytes);
    expect(first.tp_sync_bytes).toBe(expectedTensorBytes * 2);
  });
});

