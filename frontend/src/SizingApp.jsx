import InferenceSizingPanel from "./components/InferenceSizingPanel.jsx";

const HW_PRESETS = {
  "GB10 Blackwell": {
    bw: 287,
    flops: {
      FP16: 62,
      BF16: 62,
      FP8_E4M3: 124,
      FP8_E5M2: 124,
      NVFP4: 1000,
      MXFP4: 1000,
      INT8: 124,
      INT4: 248,
      FP32: 31,
    },
    intranode_bw_gbs: 900,
    intranode_lat_us: 1,
    internode_bw_gbs: 25,
    internode_lat_us: 5,
    gpus_per_node: 8,
  },
  B200: {
    bw: 8000,
    flops: {
      FP16: 180,
      BF16: 180,
      FP8_E4M3: 4500,
      FP8_E5M2: 4500,
      NVFP4: 9000,
      MXFP4: 9000,
      INT8: 4500,
      INT4: 9000,
      FP32: 90,
    },
    intranode_bw_gbs: 1800,
    intranode_lat_us: 1,
    internode_bw_gbs: 100,
    internode_lat_us: 3,
    gpus_per_node: 8,
  },
  "H100 DGX Node (8×H100)": {
    bw: 3350,
    flops: {
      FP16: 134,
      BF16: 134,
      FP8_E4M3: 1979,
      FP8_E5M2: 1979,
      NVFP4: 0,
      MXFP4: 0,
      INT8: 1979,
      INT4: 3958,
      FP32: 67,
    },
    intranode_bw_gbs: 900,
    intranode_lat_us: 1,
    internode_bw_gbs: 25,
    internode_lat_us: 5,
    gpus_per_node: 8,
  },
  "AMD Instinct MI325X": {
    bw: 6000,
    flops: {
      FP16: 1307,
      BF16: 1307,
      FP8_E4M3: 2615,
      FP8_E5M2: 2615,
      NVFP4: 5230,
      MXFP4: 5230,
      INT8: 2615,
      INT4: 5230,
      FP32: 160,
    },
    intranode_bw_gbs: 800,
    intranode_lat_us: 2,
    internode_bw_gbs: 25,
    internode_lat_us: 5,
    gpus_per_node: 8,
  },
  "AMD Instinct MI355X (est model)": {
    bw: 8000,
    flops: {
      FP16: 2000,
      BF16: 2000,
      FP8_E4M3: 4000,
      FP8_E5M2: 4000,
      NVFP4: 8000,
      MXFP4: 8000,
      INT8: 4000,
      INT4: 8000,
      FP32: 220,
    },
    intranode_bw_gbs: 800,
    intranode_lat_us: 2,
    internode_bw_gbs: 50,
    internode_lat_us: 5,
    gpus_per_node: 8,
  },
  "Cerebras WSE-3 (est model)": {
    bw: 21000000,
    flops: {
      FP16: 125000,
      BF16: 125000,
      FP8_E4M3: 125000,
      FP8_E5M2: 125000,
      NVFP4: 250000,
      MXFP4: 250000,
      INT8: 250000,
      INT4: 500000,
      FP32: 62500,
    },
    intranode_bw_gbs: 21000000,
    intranode_lat_us: 0,
    internode_bw_gbs: 1000,
    internode_lat_us: 1,
    gpus_per_node: 1,
  },
  "Intel Gaudi 3 (est model)": {
    bw: 3700,
    flops: {
      FP16: 1835,
      BF16: 1835,
      FP8_E4M3: 1835,
      FP8_E5M2: 1835,
      NVFP4: 3600,
      MXFP4: 3600,
      INT8: 3670,
      INT4: 7340,
      FP32: 120,
    },
    intranode_bw_gbs: 600,
    intranode_lat_us: 2,
    internode_bw_gbs: 25,
    internode_lat_us: 5,
    gpus_per_node: 8,
  },
  "Custom ASIC": {
    bw: 4000,
    flops: {
      FP16: 300,
      BF16: 300,
      FP8_E4M3: 600,
      FP8_E5M2: 600,
      NVFP4: 8000,
      MXFP4: 8000,
      INT8: 600,
      INT4: 8000,
      FP32: 150,
    },
    intranode_bw_gbs: 1600,
    intranode_lat_us: 1,
    internode_bw_gbs: 100,
    internode_lat_us: 3,
    gpus_per_node: 8,
  },
};

const MODELS = {
  // ── Dense models ──
  "Llama-3 8B": { L: 32, H: 4096, nh: 32, nkv: 8, dh: 128, dff: 14336, V: 128256, gate: true },
  "Llama-2 70B": { L: 80, H: 8192, nh: 64, nkv: 8, dh: 128, dff: 28672, V: 32000, gate: true },
  "o1/o3 reasoning": { L: 64, H: 6144, nh: 48, nkv: 8, dh: 128, dff: 16384, V: 128000, gate: true },
  "TinyLlama 1.1B": { L: 22, H: 2048, nh: 32, nkv: 4, dh: 64, dff: 5632, V: 32000, gate: true },
  // ── MoE models ──
  "Mixtral 8x7B": {
    L: 32, H: 4096, nh: 32, nkv: 8, dh: 128, dff: 14336, V: 32000, gate: true,
    moe: { num_experts: 8, experts_per_token: 2, expert_ffn_dim: 14336, capacity_factor: 1.25, moe_layer_freq: 1 },
  },
  "DeepSeek-V3 (MoE)": {
    L: 61, H: 7168, nh: 128, nkv: 128, dh: 128, dff: 2048, V: 129280, gate: true,
    moe: { num_experts: 256, experts_per_token: 8, expert_ffn_dim: 2048, capacity_factor: 1.3, moe_layer_freq: 1 },
  },
  "Grok-1 (MoE)": {
    L: 64, H: 6144, nh: 48, nkv: 8, dh: 128, dff: 32768, V: 131072, gate: true,
    moe: { num_experts: 8, experts_per_token: 2, expert_ffn_dim: 32768, capacity_factor: 1.25, moe_layer_freq: 2 },
  },
};

const CONFIGS = {
  FP16: { w: "FP16", a: "FP16", kv: "FP16", computeAs: "FP16" },
  "FP8 E4M3": { w: "FP8_E4M3", a: "FP8_E4M3", kv: "FP8_E4M3", computeAs: "FP8_E4M3" },
  "NVFP4 W4A4": { w: "NVFP4", a: "NVFP4", kv: "NVFP4_KV", computeAs: "NVFP4" },
  "NVFP4 W4A8+FP4KV": { w: "NVFP4", a: "FP8_E4M3", kv: "NVFP4_KV", computeAs: "FP8_E4M3" },
  "INT4/FP8": { w: "INT4", a: "FP8_E4M3", kv: "FP8_E4M3", computeAs: "FP8_E4M3" },
};

function bytesPerElement(fmt) {
  const b = {
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
    NF4: 0.53125,
  };
  return b[fmt] ?? 2.0;
}

function hwFlopsKey(computeAs) {
  const map = {
    FP32: "FP32",
    FP16: "FP16",
    BF16: "BF16",
    FP8_E4M3: "FP8_E4M3",
    FP8_E5M2: "FP8_E5M2",
    NVFP4: "NVFP4",
    MXFP4: "MXFP4",
    INT8: "INT8",
    INT4: "INT4",
  };
  return map[computeAs] || "FP16";
}

export default function SizingApp() {
  const page = {
    fontFamily: "'IBM Plex Sans', system-ui, sans-serif",
    minHeight: "100vh",
    background:
      "radial-gradient(1200px 600px at 12% -10%, rgba(56,189,248,0.08), transparent 50%), radial-gradient(1000px 550px at 100% 0%, rgba(34,197,94,0.07), transparent 50%), #050913",
    color: "#e2e8f0",
    padding: 20,
  };
  const shell = {
    maxWidth: 1200,
    margin: "0 auto",
  };
  const header = {
    background: "linear-gradient(135deg, #0f172a, #111827)",
    border: "1px solid #1e293b",
    borderRadius: 12,
    padding: "14px 16px",
    marginBottom: 12,
  };
  const caption = {
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: 11,
    color: "#94a3b8",
    marginTop: 5,
  };

  return (
    <div style={page}>
      <link
        href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap"
        rel="stylesheet"
      />
      <div style={shell}>
        <div style={header}>
          <div style={{ fontSize: 18, fontWeight: 700, fontFamily: "'IBM Plex Mono', monospace" }}>
            Roofline Sizing Co-Design Tool
          </div>
          <div style={caption}>
            Workload-first optimizer: MoE/GEMM/model → custom ASIC sizing → TP/PP/EP sweep → Pareto-optimal config
          </div>
        </div>

        <InferenceSizingPanel
          models={MODELS}
          configs={CONFIGS}
          hardwarePresets={HW_PRESETS}
          currentModelName="Mixtral 8x7B"
          currentConfigName="FP8 E4M3"
          currentHardwareName="Custom ASIC"
          currentPhase="decode"
          currentBatch={1}
          currentSeqLen={4096}
          bytesPerElement={bytesPerElement}
          hwFlopsKey={hwFlopsKey}
        />
      </div>
    </div>
  );
}
