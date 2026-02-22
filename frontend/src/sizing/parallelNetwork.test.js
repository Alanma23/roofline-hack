import { describe, expect, test } from "vitest";
import {
  ringAllReduceBytes,
  partitionLayers,
  computeParallelNetwork,
} from "./parallelNetwork";

describe("ringAllReduceBytes", () => {
  test("returns zero when tp=1", () => {
    expect(ringAllReduceBytes(4096, 1)).toBe(0);
  });

  test("matches ring formula for tp=2,4,8", () => {
    const tensorBytes = 1024;
    expect(ringAllReduceBytes(tensorBytes, 2)).toBe(1024);
    expect(ringAllReduceBytes(tensorBytes, 4)).toBe(1536);
    expect(ringAllReduceBytes(tensorBytes, 8)).toBe(1792);
  });
});

describe("partitionLayers", () => {
  test("splits layers contiguously with remainder on earlier stages", () => {
    const p = partitionLayers(10, 3);
    expect(p.stages.map((s) => s.count)).toEqual([4, 3, 3]);
    expect(p.stages.map((s) => [s.start, s.end])).toEqual([[0, 3], [4, 6], [7, 9]]);
  });
});

describe("computeParallelNetwork", () => {
  test("computes PP boundary sends for pp>1", () => {
    const layers = Array.from({ length: 6 }, (_, i) => ({
      layer: i,
      output_bytes: 8192,
      tp_sync_points: [
        { name: "attn_out", tensor_bytes: 8192 },
        { name: "mlp_down", tensor_bytes: 8192 },
      ],
    }));

    const result = computeParallelNetwork(
      { phase: "decode" },
      layers,
      { tp: 1, pp: 3, max_asics: 16 },
      { tp_link_bw_gbs: 900, tp_link_latency_us: 3, pp_link_bw_gbs: 100, pp_link_latency_us: 5 },
    );

    expect(result.totals.tp_allreduce_bytes).toBe(0);
    expect(result.totals.tp_allreduce_count).toBe(12);
    expect(result.totals.pp_send_count).toBe(2);
    expect(result.totals.pp_send_bytes).toBe(16384);
  });
});

