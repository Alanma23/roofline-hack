import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import * as d3 from "d3";

// ═══════════════════════════════════════════════
//  PRECISION FORMAT CATALOG — the core extension
// ═══════════════════════════════════════════════
const FORMATS = {
  // ── Scalar floating point ──
  FP32:      { bits: 32, E: 8, M: 23, family: "scalar_fp", blockSize: 1, scaleBits: 0, scaleType: null,
               label: "FP32", color: "#ef4444", range: [3.4e38], codebook: null },
  TF32:      { bits: 19, E: 8, M: 10, family: "scalar_fp", blockSize: 1, scaleBits: 0, scaleType: null,
               label: "TF32", color: "#dc2626", range: [3.4e38], codebook: null },
  FP16:      { bits: 16, E: 5, M: 10, family: "scalar_fp", blockSize: 1, scaleBits: 0, scaleType: null,
               label: "FP16", color: "#f97316", range: [65504], codebook: null },
  BF16:      { bits: 16, E: 8, M: 7,  family: "scalar_fp", blockSize: 1, scaleBits: 0, scaleType: null,
               label: "BF16", color: "#fb923c", range: [3.39e38], codebook: null },
  // ── Scalar FP8 ──
  FP8_E4M3:  { bits: 8,  E: 4, M: 3,  family: "scalar_fp", blockSize: 1, scaleBits: 0, scaleType: null,
               label: "FP8 E4M3", color: "#22c55e", range: [448],
               codebook: [0,.001953125,.00390625,.005859375,.0078125,.01171875,.015625,.0234375,.03125,.046875,.0625,.09375,.125,.1875,.25,.375,.5,.75,1,1.5,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256,384,448] },
  FP8_E5M2:  { bits: 8,  E: 5, M: 2,  family: "scalar_fp", blockSize: 1, scaleBits: 0, scaleType: null,
               label: "FP8 E5M2", color: "#16a34a", range: [57344], codebook: null },
  // ── OCP MX block formats (block=32, E8M0 power-of-2 scale) ──
  MXFP8_E4M3:{ bits: 8,  E: 4, M: 3,  family: "mx_block", blockSize: 32, scaleBits: 8, scaleType: "E8M0",
               label: "MXFP8 E4M3", color: "#4ade80", range: [448], codebook: null },
  MXFP8_E5M2:{ bits: 8,  E: 5, M: 2,  family: "mx_block", blockSize: 32, scaleBits: 8, scaleType: "E8M0",
               label: "MXFP8 E5M2", color: "#86efac", range: [57344], codebook: null },
  MXFP6_E3M2:{ bits: 6,  E: 3, M: 2,  family: "mx_block", blockSize: 32, scaleBits: 8, scaleType: "E8M0",
               label: "MXFP6 E3M2", color: "#0ea5e9", range: [28], codebook: [0,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.5,3,3.5,4,5,6,7,8,10,12,14,16,20,24,28] },
  MXFP6_E2M3:{ bits: 6,  E: 2, M: 3,  family: "mx_block", blockSize: 32, scaleBits: 8, scaleType: "E8M0",
               label: "MXFP6 E2M3", color: "#38bdf8", range: [7.5],
               codebook: [0,.0625,.125,.1875,.25,.3125,.375,.4375,.5,.5625,.625,.6875,.75,.875,1,1.125,1.25,1.375,1.5,1.75,2,2.25,2.5,2.75,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5] },
  MXFP4:     { bits: 4,  E: 2, M: 1,  family: "mx_block", blockSize: 32, scaleBits: 8, scaleType: "E8M0",
               label: "MXFP4 (OCP)", color: "#8b5cf6", range: [6],
               codebook: [0, 0.5, 1, 1.5, 2, 3, 4, 6] },
  MXINT8:    { bits: 8,  E: 0, M: 7,  family: "mx_block", blockSize: 32, scaleBits: 8, scaleType: "E8M0",
               label: "MXINT8", color: "#64748b", range: [127], codebook: null },
  // ── NVIDIA NVFP4 (block=16, E4M3 scale + FP32 tensor scale) ──
  NVFP4:     { bits: 4,  E: 2, M: 1,  family: "nvfp4", blockSize: 16, scaleBits: 8, scaleType: "E4M3",
               tensorScaleBits: 32, label: "NVFP4", color: "#a855f7", range: [6],
               codebook: [0, 0.5, 1, 1.5, 2, 3, 4, 6] },
  NVFP4_KV:  { bits: 4,  E: 2, M: 1,  family: "nvfp4", blockSize: 16, scaleBits: 8, scaleType: "E4M3",
               tensorScaleBits: 32, label: "NVFP4 KV", color: "#c084fc", range: [6],
               codebook: [0, 0.5, 1, 1.5, 2, 3, 4, 6] },
  // ── Integer formats ──
  INT8:      { bits: 8,  E: 0, M: 7,  family: "scalar_int", blockSize: 1, scaleBits: 0, scaleType: null,
               label: "INT8", color: "#06b6d4", range: [127], codebook: null },
  INT4:      { bits: 4,  E: 0, M: 3,  family: "scalar_int", blockSize: 1, scaleBits: 0, scaleType: null,
               label: "INT4", color: "#3b82f6", range: [7],
               codebook: [-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7] },
  // ── Lookup / NF4 (bitsandbytes, block=64, nested quant) ──
  NF4:       { bits: 4,  E: 0, M: 0,  family: "lookup", blockSize: 64, scaleBits: 16, scaleType: "FP16_absmax",
               label: "NF4 (bnb)", color: "#ec4899", range: [1],
               codebook: [-1,-.6962,-.5251,-.3949,-.2844,-.1848,-.0911,0,.0796,.1609,.2461,.3379,.4407,.5626,.7230,1] },
  INT2:      { bits: 2,  E: 0, M: 1,  family: "scalar_int", blockSize: 1, scaleBits: 0, scaleType: null,
               label: "INT2", color: "#6366f1", range: [1], codebook: [-2,-1,0,1] },
};

// Effective bits per element including scale overhead
function effectiveBitsPerElement(fmt) {
  const f = FORMATS[fmt];
  if (!f) return 32;
  if (f.blockSize <= 1) return f.bits;
  let overhead = f.scaleBits / f.blockSize;
  if (f.tensorScaleBits) overhead += f.tensorScaleBits / 1024; // amortized over large tensors
  return f.bits + overhead;
}

function bytesPerElement(fmt) {
  return effectiveBitsPerElement(fmt) / 8;
}

// Dynamic range with block scaling
function effectiveDynamicRange(fmt) {
  const f = FORMATS[fmt];
  if (!f) return { min: 0, max: 0 };
  const elemMax = f.range[0];
  if (f.scaleType === "E8M0") return { min: Math.pow(2, -127) * elemMax * -1, max: Math.pow(2, 127) * elemMax, scaleRange: Math.pow(2, 254) };
  if (f.scaleType === "E4M3") return { min: -448 * elemMax, max: 448 * elemMax, scaleRange: 448 * 2 };
  return { min: -elemMax, max: elemMax, scaleRange: 1 };
}

// Number of representable values (positive, excluding zero)
function numLevels(fmt) {
  const f = FORMATS[fmt];
  if (f.codebook) return f.codebook.filter(v => v > 0).length;
  if (f.family.includes("int")) return Math.pow(2, f.bits - 1);
  return Math.pow(2, f.bits - 1) - 1; // approx for FP
}

// ═══════════════════════════════════════════════
//  HARDWARE PRESETS
// ═══════════════════════════════════════════════
const HW_PRESETS = {
  "A100 SXM":      { bw: 2039, bwRange:[1900,2100], flops:{FP32:19.5,FP16:312,BF16:312,FP8_E4M3:0,NVFP4:0,MXFP4:0,INT8:624,INT4:0}, note:"Ampere · 3rd-gen TC · HBM2e" },
  "H100 SXM":      { bw: 3350, bwRange:[3100,3400], flops:{FP32:67,FP16:134,BF16:134,FP8_E4M3:1979,NVFP4:0,MXFP4:0,INT8:1979,INT4:3958}, note:"Hopper · FP8 TC · HBM3" },
  "GB10 Blackwell": { bw: 287, bwRange:[250,320], flops:{FP32:31,FP16:62,BF16:62,FP8_E4M3:124,FP8_E5M2:124,NVFP4:1000,MXFP4:1000,INT8:124,INT4:248}, note:"Blackwell B10 (GX10) · 128GB LPDDR5X · measured" },
  "B200":          { bw: 8000, bwRange:[7500,8500], flops:{FP32:90,FP16:180,BF16:180,FP8_E4M3:4500,NVFP4:9000,MXFP4:9000,INT8:4500,INT4:9000}, note:"Blackwell · NVFP4/MXFP4 TC · HBM3e" },
  "B300 Ultra":    { bw: 12000, bwRange:[10000,14000], flops:{FP32:125,FP16:250,BF16:250,FP8_E4M3:7000,NVFP4:14000,MXFP4:14000,INT8:7000,INT4:14000}, note:"Blackwell Ultra · Est. specs" },
  "Custom ASIC":   { bw: 4000, bwRange:[500,20000], flops:{FP32:50,FP16:100,BF16:100,FP8_E4M3:2000,NVFP4:8000,MXFP4:8000,INT8:2000,INT4:8000}, note:"Define your own" },
};

// ═══════════════════════════════════════════════
//  MODEL PRESETS
// ═══════════════════════════════════════════════
const MODELS = {
  "TinyLlama 1.1B":    { L:22, H:2048, nh:32, nkv:4,  dh:64,  dff:5632,  V:32000,  gate:true },
  "Llama-3 8B":        { L:32, H:4096, nh:32, nkv:8,  dh:128, dff:14336, V:128256, gate:true },
  "Llama-2 70B":       { L:80, H:8192, nh:64, nkv:8,  dh:128, dff:28672, V:32000,  gate:true },
  "DeepSeek-V3 est.":  { L:61, H:7168, nh:56, nkv:8,  dh:128, dff:18432, V:129280, gate:true },
  "o1/o3 reasoning":   { L:64, H:6144, nh:48, nkv:8,  dh:128, dff:16384, V:128000, gate:true },
};

// ═══════════════════════════════════════════════
//  PRECISION CONFIGS — the interesting axis
// ═══════════════════════════════════════════════
const CONFIGS = {
  "BF16 baseline":         { w:"BF16",     a:"BF16",     kv:"BF16",     acc:"FP32", computeAs:"BF16" },
  "FP16":                  { w:"FP16",     a:"FP16",     kv:"FP16",     acc:"FP32", computeAs:"FP16" },
  "FP8 E4M3":              { w:"FP8_E4M3", a:"FP8_E4M3", kv:"FP8_E4M3", acc:"FP32", computeAs:"FP8_E4M3" },
  "FP8 + NVFP4 KV":       { w:"FP8_E4M3", a:"FP8_E4M3", kv:"NVFP4_KV", acc:"FP32", computeAs:"FP8_E4M3" },
  "MXFP8 E4M3":           { w:"MXFP8_E4M3",a:"MXFP8_E4M3",kv:"MXFP8_E4M3",acc:"FP32", computeAs:"FP8_E4M3" },
  "W4A16 NF4 (bnb)":      { w:"NF4",      a:"FP16",     kv:"FP16",     acc:"FP32", computeAs:"FP16" },
  "W4A16 INT4 (AWQ)":     { w:"INT4",     a:"FP16",     kv:"FP16",     acc:"FP32", computeAs:"FP16" },
  "W4A16 MXFP4":          { w:"MXFP4",    a:"FP16",     kv:"FP16",     acc:"FP32", computeAs:"FP16" },
  "NVFP4 W4A16":          { w:"NVFP4",    a:"FP16",     kv:"FP16",     acc:"FP32", computeAs:"FP16" },
  "NVFP4 W4A4":           { w:"NVFP4",    a:"NVFP4",    kv:"NVFP4_KV", acc:"FP32", computeAs:"NVFP4" },
  "NVFP4 W4A8dyn":        { w:"NVFP4",    a:"FP8_E4M3", kv:"FP8_E4M3", acc:"FP32", computeAs:"FP8_E4M3" },
  "NVFP4 W4A8+FP4KV":     { w:"NVFP4",    a:"FP8_E4M3", kv:"NVFP4_KV", acc:"FP32", computeAs:"FP8_E4M3" },
  "MXFP4 W4A4":           { w:"MXFP4",    a:"MXFP4",    kv:"MXFP4",    acc:"FP32", computeAs:"MXFP4" },
  "MXFP6 E3M2":           { w:"MXFP6_E3M2",a:"MXFP6_E3M2",kv:"MXFP6_E3M2",acc:"FP32", computeAs:"FP8_E4M3" },
  "Mixed: MXFP4w MXFP6a": { w:"MXFP4",    a:"MXFP6_E3M2",kv:"FP8_E4M3", acc:"FP32", computeAs:"FP8_E4M3" },
  "INT4/FP8":              { w:"INT4",     a:"FP8_E4M3", kv:"FP8_E4M3", acc:"FP32", computeAs:"FP8_E4M3" },
};

// Map compute precision to hardware flops key
function hwFlopsKey(computeAs) {
  const map = {
    "FP32":"FP32","FP16":"FP16","BF16":"BF16","FP8_E4M3":"FP8_E4M3","FP8_E5M2":"FP8_E4M3",
    "NVFP4":"NVFP4","MXFP4":"MXFP4","INT8":"INT8","INT4":"INT4",
  };
  return map[computeAs] || "FP16";
}

// ═══════════════════════════════════════════════
//  ROOFLINE MATH
// ═══════════════════════════════════════════════
function computeOps(model, cfg, phase, B, S) {
  const { L, H, nh, nkv, dh, dff, V, gate } = model;
  const T = phase === "prefill" ? S : 1;
  const wB = bytesPerElement(cfg.w), aB = bytesPerElement(cfg.a);
  const kvB = bytesPerElement(cfg.kv), oB = aB;
  const cp = cfg.computeAs;
  const dkv = nkv * dh;
  const ops = [];

  const addGemm = (name, M, N, K) => {
    const flops = 2 * M * N * K;
    const bytes = M * K * aB + K * N * wB + M * N * oB;
    ops.push({ name, flops, bytes, ai: flops / bytes, type: "gemm", cp });
  };
  const addAttn = (name, Sq, Skv, isScore) => {
    const flops = 2 * B * nh * Sq * Skv * dh;
    let bytes;
    if (isScore) bytes = B * nh * Sq * dh * aB + B * nkv * Skv * dh * kvB + B * nh * Sq * Skv * aB;
    else bytes = B * nh * Sq * Skv * aB + B * nkv * Skv * dh * kvB + B * nh * Sq * dh * oB;
    ops.push({ name, flops, bytes, ai: flops / bytes, type: "attention", cp });
  };
  const addElem = (name, flops, bytes) => {
    ops.push({ name, flops, bytes, ai: bytes > 0 ? flops / bytes : 0, type: "elementwise", cp });
  };

  for (let l = 0; l < L; l++) {
    addGemm("q_proj", B * T, H, H);
    addGemm("k_proj", B * T, dkv, H);
    addGemm("v_proj", B * T, dkv, H);
    addAttn("qk_score", T, S, true);
    addElem("softmax", 5 * B * nh * T * S, 2 * B * nh * T * S * aB);
    addAttn("sv_prod", T, S, false);
    addGemm("o_proj", B * T, H, H);
    addElem("rmsnorm", 5 * B * T * H, 2 * B * T * H * aB);
    if (gate) {
      addGemm("gate_proj", B * T, dff, H);
      addGemm("up_proj", B * T, dff, H);
      addElem("silu_mul", 3 * B * T * dff, 3 * B * T * dff * aB);
    } else {
      addGemm("up_proj", B * T, dff, H);
    }
    addGemm("down_proj", B * T, H, dff);
    addElem("residual", 2 * B * T * H, 6 * B * T * H * aB);
  }
  addGemm("logit_proj", B * T, V, H);
  return ops;
}

function aggregate(ops) {
  const g = {};
  for (const op of ops) {
    if (!g[op.name]) g[op.name] = { ...op, count: 0 };
    else { g[op.name].flops += op.flops; g[op.name].bytes += op.bytes; }
    g[op.name].count++;
  }
  return Object.values(g).map(o => ({ ...o, ai: o.flops / o.bytes }));
}

function attainable(ai, peakTFlops, bwGBs) {
  return Math.min(peakTFlops * 1e12, ai * bwGBs * 1e9);
}

// ═══════════════════════════════════════════════
//  RULE-BASED OPTIMIZATION GUIDE
// ═══════════════════════════════════════════════
const OPTIMIZATION_RULES = [
  { keywords: ["speed up decode", "faster decode", "decode faster"], message: "Decode is memory-bound. Try NVFP4 W4A16 or INT4 for ~4x speedup. Lower KV cache to NVFP4_KV for long context.", highlight: ["precision_config", "what_to_do"] },
  { keywords: ["memory bound", "why memory", "memory bottleneck"], message: "AI < Critical AI means bandwidth limits. Try lower precision (FP8, INT4) to reduce bytes.", highlight: ["what_to_do", "roofline_plot"] },
  { keywords: ["long context", "kv cache", "kv memory"], message: "KV cache grows with sequence length. Use NVFP4 KV or INT4 KV to reduce memory; see Format Reference.", highlight: ["precision_config", "what_to_do"] },
  { keywords: ["which precision", "what precision", "best precision"], message: "Check the What to do panel for recommendations. Memory-bound → lower precision; compute-bound → FP8/NVFP4.", highlight: ["what_to_do"] },
  { keywords: ["validate", "measure", "measured", "benchmark"], message: "Use Kernel Spot Check to analyze a single shape. With GPU, measured points appear on the roofline.", highlight: ["kernel_analyzer"] },
  { keywords: ["optimize", "optimization", "improve"], message: "Start with the What to do panel. For decode: lower precision. For prefill: FP8 or NVFP4 if compute-bound.", highlight: ["what_to_do"] },
  { keywords: ["bottleneck", "bound"], message: "Memory-bound = bandwidth limits; lower precision helps. Compute-bound = FLOPS limits; higher precision throughput helps.", highlight: ["what_to_do", "roofline_plot"] },
];

function getOptimizationResponse(query) {
  const q = query.toLowerCase().trim();
  for (const rule of OPTIMIZATION_RULES) {
    if (rule.keywords.some(kw => q.includes(kw))) {
      return { message: rule.message, highlightSection: rule.highlight?.[0], suggestedConfig: null };
    }
  }
  return { message: "Try: 'How can I speed up decode?' or 'Why is it memory bound?' Check the What to do panel for recommendations.", highlightSection: "what_to_do" };
}

// Ollama chat for deeper insight (what to tune, what's happening)
const OLLAMA_BASE = "";
const OLLAMA_MODEL = "llama3.2";

async function askOllama(context, query) {
  try {
    const r = await fetch(`${OLLAMA_BASE}/ollama/api/chat`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: OLLAMA_MODEL,
        stream: false,
        messages: [
          { role: "system", content: `You are a roofline/GPU performance expert. Give concise actionable advice. Be specific about precision, bottleneck, and what to try next.` },
          { role: "user", content: `Current setup:\n${context}\n\nQuestion: ${query}\n\nAnswer briefly (2-4 sentences) with specific recommendations:` },
        ],
      }),
    });
    if (!r.ok) throw new Error(`Ollama ${r.status}`);
    const d = await r.json();
    return d.message?.content?.trim() || "No response.";
  } catch {
    return null;
  }
}

// ═══════════════════════════════════════════════
//  NL PARSER
// ═══════════════════════════════════════════════
async function parseNL(text) {
  try {
    const r = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "claude-sonnet-4-20250514", max_tokens: 1000,
        messages: [{ role: "user", content: `Parse this hardware/workload description into JSON. Return ONLY valid JSON.
Fields: hwName (from: ${Object.keys(HW_PRESETS).join(", ")}), bw (GB/s or null), flopsOverrides (obj like {"NVFP4":9000} or null),
modelName (from: ${Object.keys(MODELS).join(", ")} or null), precConfig (from: ${Object.keys(CONFIGS).join(", ")} or null),
phase ("prefill"/"decode"/null), batchSize (number/null), seqLen (number/null), notes (brief interpretation).
Description: "${text}"` }] }),
    });
    const d = await r.json();
    return JSON.parse(d.content.map(c => c.text || "").join("").replace(/```json|```/g, "").trim());
  } catch(e) { return { error: e.message }; }
}

// ═══════════════════════════════════════════════
//  FORMAT DETAIL PANEL
// ═══════════════════════════════════════════════
function FormatDetail({ fmt }) {
  const f = FORMATS[fmt];
  if (!f) return null;
  const eff = effectiveBitsPerElement(fmt);
  const dr = effectiveDynamicRange(fmt);
  const cb = f.codebook;
  const levels = numLevels(fmt);

  return (
    <div style={{ background: "#0f172a", borderRadius: 6, padding: 10, border: "1px solid #1e293b", fontSize: 10, fontFamily: "mono" }}>
      <div style={{ fontWeight: 600, color: f.color, marginBottom: 4 }}>{f.label}</div>
      <div style={{ color: "#94a3b8", lineHeight: 1.8 }}>
        <div>Element: {f.E > 0 ? `E${f.E}M${f.M}` : `INT${f.bits}`} · {f.bits}b</div>
        <div>Effective: <span style={{ color: "#e2e8f0" }}>{eff.toFixed(2)}b/elem</span> · {(eff / 8).toFixed(3)} B/elem</div>
        {f.blockSize > 1 && (
          <div>Block: {f.blockSize} elems · Scale: {f.scaleType} ({f.scaleBits}b)
            {f.tensorScaleBits ? ` + FP32 tensor (${f.tensorScaleBits}b)` : ""}
          </div>
        )}
        <div>Scale overhead: <span style={{ color: "#fbbf24" }}>{(eff - f.bits).toFixed(3)}b/elem</span> ({((eff - f.bits) / eff * 100).toFixed(1)}%)</div>
        <div>Element range: ±{f.range[0]} · Levels: {levels} pos values</div>
        {dr.scaleRange > 1 && <div>Effective range with scaling: ±{(dr.max).toExponential(1)}</div>}
      </div>
      {cb && (
        <div style={{ marginTop: 6 }}>
          <div style={{ color: "#64748b", marginBottom: 2 }}>Codebook (positive):</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 2 }}>
            {cb.filter(v => v >= 0).map((v, i) => (
              <span key={i} style={{ background: "#1e293b", padding: "1px 4px", borderRadius: 2, color: "#e2e8f0", fontSize: 9 }}>{v}</span>
            ))}
          </div>
          {/* Step size visualization */}
          <div style={{ marginTop: 6 }}>
            <div style={{ color: "#64748b", marginBottom: 2 }}>Step sizes (non-uniform for FP):</div>
            <svg width="100%" height={20} viewBox="0 0 220 20">
              {cb.filter(v => v >= 0).map((v, i) => {
                const maxVal = Math.max(...cb.filter(x => x >= 0));
                const x = maxVal > 0 ? (v / maxVal) * 210 + 5 : 5;
                return <line key={i} x1={x} x2={x} y1={2} y2={18} stroke={f.color} strokeWidth={1.5} opacity={0.8} />;
              })}
              <line x1={5} x2={215} y1={10} y2={10} stroke="#334155" strokeWidth={0.5} />
            </svg>
          </div>
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════
//  FORMAT COMPARISON TABLE
// ═══════════════════════════════════════════════
function FormatCompareTable({ formats }) {
  return (
    <div style={{ overflowX: "auto", fontSize: 10, fontFamily: "mono" }}>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr style={{ color: "#64748b", borderBottom: "1px solid #1e293b" }}>
            <th style={{ textAlign: "left", padding: "3px 4px" }}>Format</th>
            <th style={{ textAlign: "right", padding: "3px 4px" }}>Raw b</th>
            <th style={{ textAlign: "right", padding: "3px 4px" }}>Eff b</th>
            <th style={{ textAlign: "right", padding: "3px 4px" }}>B/elem</th>
            <th style={{ textAlign: "right", padding: "3px 4px" }}>Overhead</th>
            <th style={{ textAlign: "center", padding: "3px 4px" }}>Block</th>
            <th style={{ textAlign: "center", padding: "3px 4px" }}>Scale</th>
            <th style={{ textAlign: "right", padding: "3px 4px" }}>Levels</th>
            <th style={{ textAlign: "right", padding: "3px 4px" }}>Elem ±</th>
          </tr>
        </thead>
        <tbody>
          {formats.map(fmt => {
            const f = FORMATS[fmt]; if (!f) return null;
            const eff = effectiveBitsPerElement(fmt);
            return (
              <tr key={fmt} style={{ borderBottom: "1px solid #1e293b15", color: "#cbd5e1" }}>
                <td style={{ padding: "2px 4px", color: f.color }}>{f.label}</td>
                <td style={{ textAlign: "right", padding: "2px 4px" }}>{f.bits}</td>
                <td style={{ textAlign: "right", padding: "2px 4px", color: "#e2e8f0" }}>{eff.toFixed(2)}</td>
                <td style={{ textAlign: "right", padding: "2px 4px" }}>{(eff / 8).toFixed(3)}</td>
                <td style={{ textAlign: "right", padding: "2px 4px", color: "#fbbf24" }}>{(eff - f.bits).toFixed(2)}</td>
                <td style={{ textAlign: "center", padding: "2px 4px" }}>{f.blockSize > 1 ? f.blockSize : "—"}</td>
                <td style={{ textAlign: "center", padding: "2px 4px" }}>{f.scaleType || "—"}</td>
                <td style={{ textAlign: "right", padding: "2px 4px" }}>{numLevels(fmt)}</td>
                <td style={{ textAlign: "right", padding: "2px 4px" }}>{f.range[0]}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ═══════════════════════════════════════════════
//  ROOFLINE PLOT (d3)
// ═══════════════════════════════════════════════
const PW = 680, PH = 400, MG = { t: 28, r: 24, b: 46, l: 62 };
const iw = PW - MG.l - MG.r, ih = PH - MG.t - MG.b;
const TC = { gemm: "#60a5fa", attention: "#f472b6", elementwise: "#a3e635" };

function RooflinePlot({ ops, hw, pinned, showBands, measuredPoints = [], simulatedPoints = [] }) {
  const ref = useRef();
  const bw = hw.bw * 1e9;
  const ceilings = useMemo(() => Object.entries(hw.flops).filter(([, v]) => v > 0).map(([p, v]) => ({
    prec: p, flops: v * 1e12, label: p.replace("FP8_E4M3", "FP8").replace("_", " "),
    color: FORMATS[p]?.color || "#94a3b8",
  })).sort((a, b) => a.flops - b.flops), [hw.flops]);

  const allPts = useMemo(() => {
    const pts = ops.map(o => ({ ...o, perf: attainable(o.ai, (hw.flops[hwFlopsKey(o.cp)] || 0), hw.bw), set: "current" }));
    for (const ps of pinned) {
      for (const o of ps.ops) {
        pts.push({ ...o, perf: attainable(o.ai, (ps.hw.flops[hwFlopsKey(o.cp)] || 0), ps.hw.bw), set: ps.label });
      }
    }
    return pts;
  }, [ops, hw, pinned]);

  const xD = [0.01, 10000];
  const yMax = Math.max(1e13, ...ceilings.map(c => c.flops * 2.5), ...allPts.map(p => (p.perf || 1) * 2));
  const yMin = 1e6;
  const xS = useMemo(() => d3.scaleLog().domain(xD).range([0, iw]), []);
  const yS = useMemo(() => d3.scaleLog().domain([yMin, yMax]).range([ih, 0]), [yMax]);

  useEffect(() => {
    const svg = d3.select(ref.current); svg.selectAll("*").remove();
    const g = svg.append("g").attr("transform", `translate(${MG.l},${MG.t})`);

    // Grid
    [0.01,0.1,1,10,100,1000,10000].forEach(x => {
      g.append("line").attr("x1",xS(x)).attr("x2",xS(x)).attr("y1",0).attr("y2",ih).attr("stroke","#1e293b").attr("stroke-width",.4);
    });

    // BW diagonal + band
    const bwPts = [0.01,0.1,1,10,100,1000,10000].map(x => ({x, y: x*bw})).filter(d => d.y>=yMin && d.y<=yMax);
    if (bwPts.length > 1) {
      g.append("path").datum(bwPts).attr("d",d3.line().x(d=>xS(d.x)).y(d=>yS(d.y)))
        .attr("stroke","#475569").attr("stroke-width",2).attr("fill","none").attr("stroke-dasharray","5,3");
    }
    if (showBands && hw.bwRange) {
      const lo = hw.bwRange[0]*1e9, hi = hw.bwRange[1]*1e9;
      const band = [0.01,0.1,1,10,100,1000,10000].map(x => ({x, yL:Math.max(yMin,Math.min(yMax,x*lo)), yH:Math.max(yMin,Math.min(yMax,x*hi))}));
      g.append("path").datum(band).attr("d",d3.area().x(d=>xS(d.x)).y0(d=>yS(Math.max(yMin,d.yL))).y1(d=>yS(Math.min(yMax,d.yH))))
        .attr("fill","#475569").attr("opacity",.06);
    }

    // Compute ceilings
    for (const c of ceilings) {
      const y = yS(c.flops); if (y < 0 || y > ih) continue;
      g.append("line").attr("x1",0).attr("x2",iw).attr("y1",y).attr("y2",y)
        .attr("stroke",c.color).attr("stroke-width",1.2).attr("opacity",.6);
      g.append("text").attr("x",iw-2).attr("y",y-4).attr("fill",c.color).attr("font-size",8)
        .attr("text-anchor","end").attr("font-family","monospace").text(`${c.label} ${(c.flops/1e12).toFixed(0)}T`);
    }

    // Points
    const pinColors = ["#fbbf24","#34d399","#c084fc","#fb7185"];
    const tip = d3.select("#tip2");
    for (const pt of allPts) {
      if (!pt.perf || pt.perf < yMin) continue;
      const cx = xS(Math.max(xD[0], Math.min(xD[1], pt.ai)));
      const cy = yS(Math.max(yMin, Math.min(yMax, pt.perf)));
      const col = pt.set === "current" ? TC[pt.type] : pinColors[pinned.findIndex(p => p.label === pt.set) % pinColors.length];
      const op = pt.set === "current" ? .85 : .4;
      const sz = pt.set === "current" ? Math.max(3, Math.min(8, Math.log10(pt.flops / 1e6))) : 3;

      const el = pt.type === "attention"
        ? g.append("rect").attr("x",cx-sz).attr("y",cy-sz).attr("width",sz*2).attr("height",sz*2).attr("transform",`rotate(45,${cx},${cy})`)
        : pt.type === "elementwise"
        ? g.append("rect").attr("x",cx-sz).attr("y",cy-sz).attr("width",sz*2).attr("height",sz*2)
        : g.append("circle").attr("cx",cx).attr("cy",cy).attr("r",sz);

      el.attr("fill",col).attr("opacity",op).attr("stroke","#0f172a").attr("stroke-width",.5).style("cursor","pointer")
        .on("mouseenter",(e)=>{
          const pk = (hw.flops[hwFlopsKey(pt.cp)]||0)*1e12;
          const critAI = pk / bw;
          const bn = pt.ai < critAI ? "MEMORY" : "COMPUTE";
          const eff = pk > 0 ? (pt.perf/pk*100).toFixed(1) : "—";
          tip.style("display","block").style("left",(e.offsetX+10)+"px").style("top",(e.offsetY-10)+"px")
            .html(`<b>${pt.name}</b>${pt.set!=="current"?` [${pt.set}]`:""}<br/>AI: ${pt.ai.toFixed(2)} · ${bn}<br/>GFLOP: ${(pt.flops/1e9).toFixed(1)} · MB: ${(pt.bytes/1e6).toFixed(1)}<br/>Eff: ${eff}% · ${(pt.flops/pt.perf*1e6).toFixed(0)}μs`);
        }).on("mouseleave",()=>tip.style("display","none"));
    }

    // Precision color mapping (for measured points)
    const PRECISION_COLORS = {
      "FP16": "#f97316",
      "BF16": "#fb923c",
      "FP8_E4M3": "#22c55e",
      "FP8_E5M2": "#16a34a",
      "NVFP4": "#a855f7",
      "INT8": "#06b6d4",
      "INT4": "#3b82f6",
    };
    // Simulated (predicted) points overlay (from GEMM Analyzer)
    for (const spt of simulatedPoints) {
      if (!spt.ai || !spt.tflops) continue;
      const perf = spt.tflops * 1e12;
      if (perf < yMin) continue;
      const cx = xS(Math.max(xD[0], Math.min(xD[1], spt.ai)));
      const cy = yS(Math.max(yMin, Math.min(yMax, perf)));
      g.append("circle").attr("cx",cx).attr("cy",cy).attr("r",5)
        .attr("fill","none").attr("stroke","#60a5fa").attr("stroke-width",1.5).attr("stroke-dasharray","3,2").attr("opacity",.9)
        .style("cursor","pointer")
        .on("mouseenter",(e)=>{
          tip.style("display","block").style("left",(e.offsetX+10)+"px").style("top",(e.offsetY-10)+"px")
            .html(`<b>${spt.label || "Predicted"}</b><br/>AI: ${spt.ai.toFixed(2)}<br/>TFLOPS: ${spt.tflops.toFixed(2)}<br/>${spt.time_us ? spt.time_us.toFixed(1)+"μs" : ""}`);
        }).on("mouseleave",()=>tip.style("display","none"));
      g.append("text").attr("x",cx+7).attr("y",cy+2).attr("fill","#60a5fa").attr("font-size",7).attr("font-family","monospace").text("P");
    }
    // Measured points overlay (from GEMM Analyzer)
    for (const mpt of measuredPoints) {
      if (!mpt.ai || !mpt.tflops) continue;
      const perf = mpt.tflops * 1e12;
      if (perf < yMin) continue;
      const cx = xS(Math.max(xD[0], Math.min(xD[1], mpt.ai)));
      const cy = yS(Math.max(yMin, Math.min(yMax, perf)));

      // Color by precision
      const color = PRECISION_COLORS[mpt.precision] || "#22c55e";

      // Measured = hollow circle with colored stroke
      g.append("circle").attr("cx",cx).attr("cy",cy).attr("r",6)
        .attr("fill","none").attr("stroke",color).attr("stroke-width",2.5).attr("opacity",.9)
        .style("cursor","pointer")
        .on("mouseenter",(e)=>{
          tip.style("display","block").style("left",(e.offsetX+10)+"px").style("top",(e.offsetY-10)+"px")
            .html(`<b>${mpt.label || "Measured"}</b><br/>AI: ${mpt.ai.toFixed(2)}<br/>TFLOPS: ${mpt.tflops.toFixed(2)}<br/>${mpt.time_us ? mpt.time_us.toFixed(1)+"μs" : ""}${mpt.bandwidth_gb_s ? "<br/>BW: "+mpt.bandwidth_gb_s.toFixed(1)+" GB/s" : ""}`);
        }).on("mouseleave",()=>tip.style("display","none"));
      // Label
      g.append("text").attr("x",cx+8).attr("y",cy+3).attr("fill",color).attr("font-size",8).attr("font-family","monospace").text("M");
    }

    // Axes
    g.append("g").attr("transform",`translate(0,${ih})`).call(d3.axisBottom(xS).tickValues([.01,.1,1,10,100,1000,10000]).tickFormat(d=>d>=1?d3.format(",")(d):d))
      .selectAll("text,line,path").attr("stroke","#475569").attr("fill","#475569");
    g.append("g").call(d3.axisLeft(yS).ticks(6).tickFormat(d=>d>=1e12?(d/1e12).toFixed(0)+"T":d>=1e9?(d/1e9).toFixed(0)+"G":d>=1e6?(d/1e6).toFixed(0)+"M":d))
      .selectAll("text,line,path").attr("stroke","#475569").attr("fill","#475569");

    g.append("text").attr("x",iw/2).attr("y",ih+38).attr("fill","#64748b").attr("text-anchor","middle").attr("font-size",10).attr("font-family","monospace").text("Arithmetic Intensity (FLOP/byte)");
    g.append("text").attr("x",-ih/2).attr("y",-46).attr("fill","#64748b").attr("text-anchor","middle").attr("font-size",10).attr("transform","rotate(-90)").attr("font-family","monospace").text("Attainable FLOP/s");
  }, [allPts, hw, bw, ceilings, xS, yS, showBands, pinned, yMax, measuredPoints, simulatedPoints]);

  return (
    <div style={{ position: "relative" }}>
      <svg ref={ref} width={PW} height={PH} style={{ background: "#080d18", borderRadius: 6 }} />
      <div id="tip2" style={{ display:"none",position:"absolute",background:"#1e293bee",border:"1px solid #334155",borderRadius:5,padding:"6px 10px",fontSize:10,color:"#e2e8f0",pointerEvents:"none",fontFamily:"monospace",lineHeight:1.6,zIndex:10,maxWidth:300 }} />
    </div>
  );
}

// ═══════════════════════════════════════════════
//  OP TABLE
// ═══════════════════════════════════════════════
function OpTable({ ops, hw }) {
  const bwVal = hw.bw;
  const sorted = [...ops].sort((a,b) => {
    const tA = a.flops / attainable(a.ai, (hw.flops[hwFlopsKey(a.cp)]||1), bwVal);
    const tB = b.flops / attainable(b.ai, (hw.flops[hwFlopsKey(b.cp)]||1), bwVal);
    return tB - tA;
  });
  const totTime = sorted.reduce((s,o) => s + o.flops / attainable(o.ai, (hw.flops[hwFlopsKey(o.cp)]||1), bwVal), 0);
  return (
    <div style={{ maxHeight: 190, overflowY: "auto", fontSize: 10, fontFamily: "monospace" }}>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead><tr style={{ color: "#64748b", borderBottom: "1px solid #1e293b", position: "sticky", top: 0, background: "#0f172a" }}>
          <th style={{textAlign:"left",padding:"3px 4px"}}>Op</th>
          <th style={{textAlign:"right",padding:"3px 4px"}}>AI</th>
          <th style={{textAlign:"right",padding:"3px 4px"}}>GFLOP</th>
          <th style={{textAlign:"right",padding:"3px 4px"}}>MB</th>
          <th style={{textAlign:"right",padding:"3px 4px"}}>μs</th>
          <th style={{textAlign:"right",padding:"3px 4px"}}>%time</th>
          <th style={{textAlign:"center",padding:"3px 4px"}}>Bnd</th>
        </tr></thead>
        <tbody>{sorted.slice(0,18).map((o,i)=>{
          const pk = (hw.flops[hwFlopsKey(o.cp)]||1)*1e12;
          const at = attainable(o.ai, (hw.flops[hwFlopsKey(o.cp)]||1), bwVal);
          const t = o.flops / at;
          const critAI = pk / (bwVal * 1e9);
          const bn = o.ai < critAI ? "M" : "C";
          return <tr key={i} style={{borderBottom:"1px solid #1e293b10",color:"#cbd5e1"}}>
            <td style={{padding:"2px 4px",display:"flex",gap:4,alignItems:"center"}}>
              <span style={{width:6,height:6,borderRadius:o.type==="gemm"?"50%":1,background:TC[o.type],display:"inline-block"}} />{o.name}{o.count>1&&<span style={{color:"#64748b"}}>×{o.count}</span>}</td>
            <td style={{textAlign:"right",padding:"2px 4px"}}>{o.ai.toFixed(1)}</td>
            <td style={{textAlign:"right",padding:"2px 4px"}}>{(o.flops/1e9).toFixed(1)}</td>
            <td style={{textAlign:"right",padding:"2px 4px"}}>{(o.bytes/1e6).toFixed(1)}</td>
            <td style={{textAlign:"right",padding:"2px 4px"}}>{(t*1e6).toFixed(0)}</td>
            <td style={{textAlign:"right",padding:"2px 4px"}}>{(t/totTime*100).toFixed(1)}</td>
            <td style={{textAlign:"center",padding:"2px 4px",color:bn==="C"?"#f87171":"#60a5fa"}}>{bn}</td>
          </tr>;
        })}</tbody>
      </table>
    </div>
  );
}

// ═══════════════════════════════════════════════
//  WHAT TO DO PANEL — roofline-driven quantization recommendation
// ═══════════════════════════════════════════════
function WhatToDoPanel({ bound, aggAI, hw, hwName, cfg, peakT }) {
  const isBlackwell = hwName.includes("B10") || hwName.includes("B200") || hwName.includes("B300") || hwName.includes("Blackwell");
  const criticalAI = peakT > 0 && hw.bw > 0 ? (peakT * 1e12) / (hw.bw * 1e9) : 200;
  const memBound = bound === "MEMORY";
  const wBits = effectiveBitsPerElement(cfg?.w || "FP16");

  let recPrec = "FP16";
  let recMethod = "—";
  let recNote = "";

  if (memBound) {
    if (isBlackwell) {
      // Blackwell has native FP8/FP4
      if (wBits > 4.5) {
        recPrec = "NVFP4 W4A4";
        recMethod = "Native FP4 tensor cores";
        recNote = "Memory-bound on Blackwell → NVFP4 gives ~4x BW reduction + 2x compute vs FP8. Native tensor core path.";
      } else {
        recPrec = "Already at FP4";
        recMethod = "Optimal";
        recNote = "Already using FP4 weights. Consider FP4 KV cache for additional savings.";
      }
    } else {
      recPrec = "FP8_E4M3 or NVFP4";
      recMethod = "Native FP8/FP4";
      recNote = "Memory-bound → lower precision = proportional speedup.";
    }
  } else {
    if (isBlackwell) {
      recPrec = "FP8_E4M3";
      recMethod = "Native FP8 tensor cores";
      recNote = "Compute-bound on Blackwell → FP8 doubles compute throughput. NVFP4 for 4x if accuracy allows.";
    } else {
      recPrec = "INT8 or FP16";
      recMethod = "PTQ";
      recNote = "Compute-bound → quantization helps less. INT8 still reduces memory.";
    }
  }

  return (
    <div style={{ fontSize: 10, fontFamily: "monospace", color: "#94a3b8", lineHeight: 1.7 }}>
      <div style={{ color: "#e2e8f0", marginBottom: 4 }}>
        <b>Bottleneck:</b> {bound}
      </div>
      <div>AI: {aggAI.toFixed(2)} · Critical: {criticalAI.toFixed(0)}</div>
      <div style={{ marginTop: 6, color: "#22c55e" }}>
        <b>Recommend:</b> {recPrec}
      </div>
      <div><b>Method:</b> {recMethod}</div>
      <div style={{ marginTop: 4, fontSize: 9, color: "#64748b" }}>{recNote}</div>
    </div>
  );
}

// ═══════════════════════════════════════════════
//  HW NAME → API KEY
// ═══════════════════════════════════════════════
const HW_NAME_TO_API_KEY = {
  "GB10 Blackwell": "b10",
  "B200": "b200",
  "H100 SXM": "h100",
  "A100 SXM": "a100",
  "B300 Ultra": "b10",
  "Custom ASIC": "b10",
};

// ═══════════════════════════════════════════════
//  GEMM ANALYZER — Kernel Spot Check
// ═══════════════════════════════════════════════
// Use VITE_API_BASE for remote B10 (e.g. http://192.168.1.100:8000). Empty = Vite proxy to localhost:8000
const API_BASE = import.meta.env.VITE_API_BASE ?? "";

const QUICK_SHAPES = [
  { label: "Decode", M: 1, N: 4096, K: 4096 },
  { label: "FFN up", M: 1, N: 4096, K: 14336 },
  { label: "Prefill", M: 2048, N: 4096, K: 4096 },
];

// GB10 iterative sweep — run across shapes/precisions, plot all points
function SweepPanel({ hwKey, onPoints }) {
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState(null);

  const runSweep = async (mode) => {
    setLoading(true);
    setError(null);
    setStatus(mode === "quick" ? "Quick sweep (3 shapes × 3 precisions)…" : "Full sweep (10 shapes × all precisions)…");
    try {
      const params = new URLSearchParams({ hardware_key: hwKey, run_measured: "true" });
      if (mode === "quick") {
        params.set("quick", "true");
        ["FP16", "FP8_E4M3", "NVFP4"].forEach(p => params.append("precisions", p));
      }
      const r = await fetch(`${API_BASE}/api/sweep?${params}`, { method: "POST" });
      if (!r.ok) throw new Error(`Sweep failed: ${r.status}`);
      const data = await r.json();
      const sim = (data.points || []).filter(p => p.source === "simulated");
      const meas = (data.points || []).filter(p => p.source === "measured");
      onPoints(sim, meas);
      setStatus(`Done: ${sim.length} predicted, ${meas.length} measured`);
    } catch (e) {
      setError(e.message);
      setStatus("");
    }
    setLoading(false);
  };

  const P = { background:"#0f172a", borderRadius:6, padding:10, border:"1px solid #1e293b", fontSize:10, fontFamily:"monospace" };
  const btn2 = { background:"#8b5cf6", color:"#fff", border:"none", borderRadius:3, padding:"6px 12px", fontSize:11, cursor:"pointer", fontFamily:"monospace" };
  return (
    <div style={P}>
      <div style={{fontSize:9,color:"#475569",textTransform:"uppercase",letterSpacing:.8,marginBottom:6}}>GB10 Iterative Sweep</div>
      <div style={{fontSize:9,color:"#64748b",marginBottom:8}}>Run benchmarks across shapes & precisions to populate the roofline with measured points.</div>
      <div style={{display:"flex",gap:6,marginBottom:6}}>
        <button onClick={()=>runSweep("quick")} disabled={loading} style={{...btn2,opacity:loading?0.6:1,flex:1}}>Quick (~30s)</button>
        <button onClick={()=>runSweep("full")} disabled={loading} style={{...btn2,opacity:loading?0.6:1,flex:1}}>Full (~2min)</button>
      </div>
      {status && <div style={{fontSize:9,color:"#94a3b8",marginBottom:4}}>{status}</div>}
      {error && <div style={{fontSize:9,color:"#f87171"}}>{error}</div>}
    </div>
  );
}

// Live GPU benchmark — saturate GPU with different formats, stream results
const BENCH_SHAPES = [
  { label: "Decode", val: "1,4096,4096" },
  { label: "Prefill", val: "2048,4096,4096" },
  { label: "Square", val: "4096,4096,4096" },
];
const BENCH_PRECISIONS = ["FP16", "BF16", "FP8_E4M3", "NVFP4", "INT8", "INT4"];

function BenchmarkPanel({ hwKey, onMeasuredPoints }) {
  const [shapes, setShapes] = useState(["1,4096,4096", "2048,4096,4096", "4096,4096,4096"]);
  const [precs, setPrecs] = useState(["FP16", "FP8_E4M3", "NVFP4", "INT8"]);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState("");
  const [results, setResults] = useState([]);
  const [error, setError] = useState(null);
  const abortRef = useRef(null);

  const toggleShape = (val) => {
    setShapes(prev => prev.includes(val) ? prev.filter(s=>s!==val) : [...prev, val]);
  };
  const togglePrec = (p) => {
    setPrecs(prev => prev.includes(p) ? prev.filter(x=>x!==p) : [...prev, p]);
  };

  const runBenchmarks = async (saturateMode = false) => {
    setRunning(true);
    setError(null);
    setResults([]);
    const shapesStr = shapes.length ? shapes.join(";") : "1,4096,4096;2048,4096,4096;4096,4096,4096";
    const precStr = precs.length ? precs.join(",") : "FP16,FP8_E4M3,NVFP4,INT8";
    let url = `${API_BASE}/api/benchmark/stream?hardware_key=${hwKey}&shapes=${encodeURIComponent(shapesStr)}&precisions=${encodeURIComponent(precStr)}`;
    if (saturateMode) url = `${API_BASE}/api/benchmark/stream?hardware_key=${hwKey}&saturate=true`;
    const collected = [];

    try {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`API ${resp.status}`);
      const reader = resp.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      abortRef.current = () => reader.cancel();

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() || "";
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.error) {
                const entry = { shape: data.shape || "?", precision: data.precision || "?", error: data.error };
                collected.push(entry);
                setResults(r => [...r, entry]);
              } else if (data.done) {
                setProgress(`Done: ${data.total} benchmarks`);
              } else {
                collected.push(data);
                setResults(r => [...r, data]);
                setProgress(`Running ${data.shape} ${data.precision}… (${data.progress || ""})`);
              }
            } catch (_) {}
          }
        }
      }
      setProgress("Complete");
      const pts = collected.filter(r => r.ai != null && !r.error).map(r => ({ ai: r.ai, tflops: r.tflops, time_us: r.time_us, bandwidth_gb_s: r.bandwidth_gb_s, label: `${r.shape} [${r.precision}]`, precision: r.precision, shape: r.shape }));
      if (onMeasuredPoints && pts.length) onMeasuredPoints(pts);
    } catch (e) {
      setError(e.message);
      setProgress("");
    }
    setRunning(false);
    abortRef.current = null;
  };

  const P = { background:"#0f172a", borderRadius:6, padding:10, border:"1px solid #1e293b", fontSize:10, fontFamily:"monospace" };
  const btn2 = { background:"#22c55e", color:"#0f172a", border:"none", borderRadius:3, padding:"6px 12px", fontSize:11, cursor:"pointer", fontFamily:"monospace", fontWeight:600 };
  const chk = { accentColor:"#22c55e" };

  return (
    <div style={P}>
      <div style={{fontSize:9,color:"#475569",textTransform:"uppercase",letterSpacing:.8,marginBottom:6}}>Live GPU Benchmark</div>
      <div style={{fontSize:9,color:"#64748b",marginBottom:8}}>Saturate the GPU with different precisions — see live TFLOPS as each benchmark completes.</div>

      <div style={{fontSize:8,color:"#64748b",marginBottom:3}}>Shapes</div>
      <div style={{display:"flex",flexWrap:"wrap",gap:4,marginBottom:8}}>
        {BENCH_SHAPES.map(s=>(
          <label key={s.val} style={{display:"flex",alignItems:"center",gap:4,cursor:"pointer",fontSize:9}}>
            <input type="checkbox" checked={shapes.includes(s.val)} onChange={()=>toggleShape(s.val)} style={chk} />
            {s.label}
          </label>
        ))}
      </div>

      <div style={{fontSize:8,color:"#64748b",marginBottom:3}}>Precisions</div>
      <div style={{display:"flex",flexWrap:"wrap",gap:6,marginBottom:8}}>
        {BENCH_PRECISIONS.map(p=>(
          <label key={p} style={{display:"flex",alignItems:"center",gap:4,cursor:"pointer",fontSize:9}}>
            <input type="checkbox" checked={precs.includes(p)} onChange={()=>togglePrec(p)} style={chk} />
            {FORMATS[p]?.label || p}
          </label>
        ))}
      </div>

      <div style={{display:"flex",gap:6,marginBottom:6,flexWrap:"wrap"}}>
        <button onClick={()=>runBenchmarks(false)} disabled={running || shapes.length===0 || precs.length===0}
          style={{...btn2,opacity:(running||!shapes.length||!precs.length)?0.6:1,flex:1,minWidth:100}}>
          {running ? "Running…" : "Run benchmarks"}
        </button>
        <button onClick={()=>runBenchmarks(true)} disabled={running}
          style={{...btn2,background:"#f59e0b",opacity:running?0.6:1,flex:1,minWidth:100}}>
          {running ? "Running…" : "Saturation sweep"}
        </button>
      </div>
      <div style={{fontSize:8,color:"#64748b",marginTop:-2,marginBottom:4}}>Saturation: larger shapes (8K³, 16K×8K) to max out GPU</div>
      {progress && <div style={{fontSize:9,color:"#94a3b8",marginBottom:4}}>{progress}</div>}
      {error && <div style={{fontSize:9,color:"#f87171",marginBottom:4}}>{error}</div>}

      {results.length > 0 && (
        <div style={{maxHeight:180,overflowY:"auto",fontSize:9,marginTop:6,border:"1px solid #1e293b",borderRadius:4,padding:4,background:"#080d18"}}>
          <table style={{width:"100%",borderCollapse:"collapse"}}>
            <thead>
              <tr style={{color:"#64748b",borderBottom:"1px solid #1e293b"}}>
                <th style={{textAlign:"left",padding:"2px 4px"}}>Shape</th>
                <th style={{textAlign:"left",padding:"2px 4px"}}>Prec</th>
                <th style={{textAlign:"right",padding:"2px 4px"}}>μs</th>
                <th style={{textAlign:"right",padding:"2px 4px"}}>TFLOPS</th>
                <th style={{textAlign:"right",padding:"2px 4px"}}>AI</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r,i)=>(
                <tr key={i} style={{borderBottom:"1px solid #1e293b15",color:r.error?"#f87171":"#cbd5e1"}}>
                  <td style={{padding:"2px 4px"}}>{r.shape||"?"}</td>
                  <td style={{padding:"2px 4px",color:FORMATS[r.precision]?.color}}>{r.precision||"?"}</td>
                  <td style={{textAlign:"right",padding:"2px 4px"}}>{r.error?"—":r.time_us?.toFixed(1)}</td>
                  <td style={{textAlign:"right",padding:"2px 4px"}}>{r.error?"—":r.tflops?.toFixed(2)}</td>
                  <td style={{textAlign:"right",padding:"2px 4px"}}>{r.error?r.error:r.ai?.toFixed(2)??"—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {results.length > 0 && !running && onMeasuredPoints && (
        <button onClick={()=>{
          const pts = results.filter(r=>r.ai!=null).map(r=>({ai:r.ai,tflops:r.tflops,time_us:r.time_us,bandwidth_gb_s:r.bandwidth_gb_s,label:`${r.shape} [${r.precision}]`,precision:r.precision,shape:r.shape}));
          if(pts.length) onMeasuredPoints(pts);
        }} style={{...btn2,background:"#8b5cf6",color:"#fff",marginTop:6,width:"100%",fontSize:9}}>
          Add measured points to roofline
        </button>
      )}
    </div>
  );
}

function GEMMAnalyzer({ hw, hwKey, onMeasuredPoints, onSimulatedPoints }) {
  const [M, setM] = useState(4096);
  const [N, setN] = useState(4096);
  const [K, setK] = useState(4096);
  const [precision, setPrecision] = useState("FP16");
  const [runAllPrecisions, setRunAllPrecisions] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [nvmlStatus, setNvmlStatus] = useState(null);
  const [nvmlUnavailable, setNvmlUnavailable] = useState(false);

  const precOptions = ["FP16","BF16","TF32","FP8_E4M3","FP8_E5M2","NVFP4","MXFP4","INT8","INT4"];

  const applyShape = (shape) => {
    setM(shape.M);
    setN(shape.N);
    setK(shape.K);
  };

  const analyze = async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch(`${API_BASE}/api/analyze?hardware_key=${hwKey}&run_all_precisions=${runAllPrecisions}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ M, N, K, precision }),
      });
      if (!resp.ok) throw new Error(`API error: ${resp.status}`);
      const data = await resp.json();
      setResult(data);
      // Pass simulated and measured points up for roofline overlay
      if (onSimulatedPoints && data.simulated?.length > 0) {
        onSimulatedPoints(data.simulated);
      }
      if (data.measured && data.measured.length > 0 && onMeasuredPoints) {
        onMeasuredPoints(data.measured);
      }
    } catch (e) {
      setError(e.message);
      // Fallback: compute locally without API
      const bpe = bytesPerElement(precision);
      const flops = 2 * M * N * K;
      const bytes = (M * K + K * N) * bpe + M * N * 2;
      const ai = flops / bytes;
      const peakKey = precision.includes("FP8") ? "FP8_E4M3" : precision.includes("FP4") || precision === "NVFP4" || precision === "MXFP4" ? "NVFP4" : precision;
      const peakT = hw.flops[peakKey] || hw.flops["FP16"] || 1;
      const perf = attainable(ai, peakT, hw.bw);
      const timeUs = flops / perf * 1e6;
      const critical = peakT > 0 && hw.bw > 0 ? (peakT * 1e12) / (hw.bw * 1e9) : 0;
      const bound = ai < critical ? "memory" : "compute";
      const simPt = { ai, tflops: perf / 1e12, time_us: timeUs, label: `${M}x${N}x${K} [${precision}]`, source: "simulated", precision, shape: `${M}x${N}x${K}`, bottleneck: bound };
      setResult({
        hardware: hw.name || hwKey,
        simulated: [simPt],
        measured: [],
        recommendation: { precision: bound === "memory" ? "NVFP4" : "FP8_E4M3", method: bound === "memory" ? "native_fp4" : "native_fp8", reason: `${bound}-bound: AI=${ai.toFixed(1)} vs critical=${critical.toFixed(0)}`, predicted_speedup: 1.0, memory_bound: bound === "memory", memory_savings_pct: 0 },
      });
      if (onSimulatedPoints) onSimulatedPoints([simPt]);
    }
    setLoading(false);
  };

  // Poll NVML status
  useEffect(() => {
    const poll = async () => {
      try {
        const resp = await fetch(`${API_BASE}/api/nvml/status`);
        if (resp.ok) {
          setNvmlStatus(await resp.json());
          setNvmlUnavailable(false);
        } else {
          setNvmlUnavailable(true);
        }
      } catch {
        setNvmlUnavailable(true);
      }
    };
    poll();
    const interval = setInterval(poll, 2000);
    return () => clearInterval(interval);
  }, []);

  const P = { background:"#0f172a", borderRadius:6, padding:10, border:"1px solid #1e293b", fontSize:10, fontFamily:"monospace" };
  const sel = { background:"#1e293b", color:"#e2e8f0", border:"1px solid #334155", borderRadius:3, padding:"4px 6px", fontSize:11, width:"100%", fontFamily:"monospace", outline:"none" };
  const btn2 = { background:"#8b5cf6", color:"#fff", border:"none", borderRadius:3, padding:"6px 12px", fontSize:11, cursor:"pointer", fontFamily:"monospace", width:"100%" };

  return (
    <div>
      {/* GEMM Input */}
      <div style={P}>
        <div style={{fontSize:9,color:"#475569",textTransform:"uppercase",letterSpacing:.8,marginBottom:4}}>Kernel Spot Check</div>
        <div style={{fontSize:9,color:"#64748b",marginBottom:6}}>Analyze a single kernel shape to validate theory vs measured performance.</div>
        <div style={{display:"flex",flexWrap:"wrap",gap:4,marginBottom:6}}>
          {QUICK_SHAPES.map((s,i)=>(
            <button key={i} onClick={()=>applyShape(s)} style={{...btn2,opacity:0.8,padding:"4px 8px",fontSize:9}}>{s.label}</button>
          ))}
        </div>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:4,marginBottom:6}}>
          {[["M",M,setM],["N",N,setN],["K",K,setK]].map(([label,val,setter])=>(
            <div key={label}>
              <div style={{fontSize:8,color:"#64748b",marginBottom:2}}>{label}</div>
              <input type="number" value={val} onChange={e=>setter(parseInt(e.target.value)||1)}
                style={{...sel,width:"100%"}} />
            </div>
          ))}
        </div>
        <div style={{marginBottom:6}}>
          <div style={{fontSize:8,color:"#64748b",marginBottom:2}}>Precision</div>
          <select value={precision} onChange={e=>setPrecision(e.target.value)} style={sel}>
            {precOptions.map(p=><option key={p} value={p}>{FORMATS[p]?.label||p}</option>)}
          </select>
        </div>
        <div style={{marginBottom:8}}>
          <label style={{fontSize:11,color:"#cbd5e1",cursor:"pointer",display:"flex",alignItems:"center"}}>
            <input
              type="checkbox"
              checked={runAllPrecisions}
              onChange={(e) => setRunAllPrecisions(e.target.checked)}
              style={{marginRight:6}}
            />
            Run all precisions (FP16, FP8, NVFP4, INT8, INT4)
          </label>
        </div>
        <button onClick={analyze} disabled={loading} style={{...btn2,opacity:loading?.6:1}}>
          {loading ? "Analyzing..." : "Analyze GEMM"}
        </button>
      </div>

      {/* Results */}
      {result && (
        <div style={{...P,marginTop:8}}>
          <div style={{fontSize:9,color:"#475569",textTransform:"uppercase",letterSpacing:.8,marginBottom:6}}>Analysis Result</div>
          {result.simulated && result.simulated[0] && (
            <div style={{color:"#94a3b8",lineHeight:1.8}}>
              <div>AI: <span style={{color:"#e2e8f0"}}>{result.simulated[0].ai.toFixed(2)}</span> FLOP/byte</div>
              <div>Predicted: <span style={{color:"#e2e8f0"}}>{result.simulated[0].time_us.toFixed(1)}</span> us</div>
              <div>Bottleneck: <span style={{color:result.simulated[0].bottleneck==="memory"?"#60a5fa":"#f87171"}}>{result.simulated[0].bottleneck?.toUpperCase()}</span></div>
            </div>
          )}
          {result.measured && result.measured.length > 0 && (
            <div style={{marginTop:4,color:"#94a3b8",lineHeight:1.8}}>
              {result.measured.length === 1 ? (
                <>
                  <div style={{color:"#22c55e"}}>Measured: {result.measured[0].time_us.toFixed(1)} μs</div>
                  <div>Throughput: {result.measured[0].tflops.toFixed(2)} TFLOPS</div>
                  {result.measured[0].bandwidth_gb_s && <div>BW: {result.measured[0].bandwidth_gb_s.toFixed(1)} GB/s</div>}
                </>
              ) : (
                <>
                  <div style={{color:"#22c55e",marginBottom:4}}>Measured {result.measured.length} precisions:</div>
                  {result.measured.map((m, idx) => (
                    <div key={idx} style={{fontSize:9,marginBottom:2}}>
                      <span style={{color:FORMATS[m.precision]?.color||"#22c55e"}}>{m.precision}</span>: {m.tflops.toFixed(2)} TFLOPS · {m.time_us.toFixed(1)} μs
                    </div>
                  ))}
                </>
              )}
            </div>
          )}
          {result.recommendation && (
            <div style={{marginTop:6,padding:6,background:"#1e293b",borderRadius:4}}>
              <div style={{color:"#22c55e",fontWeight:600}}>Recommend: {result.recommendation.precision}</div>
              <div style={{color:"#94a3b8"}}>{result.recommendation.method}</div>
              <div style={{fontSize:9,color:"#64748b",marginTop:2}}>{result.recommendation.reason}</div>
              {result.recommendation.predicted_speedup > 1 && (
                <div style={{color:"#fbbf24",marginTop:2}}>~{result.recommendation.predicted_speedup.toFixed(1)}x speedup · {result.recommendation.memory_savings_pct.toFixed(0)}% memory savings</div>
              )}
            </div>
          )}
        </div>
      )}

      {error && (
        <div style={{...P,marginTop:8,borderColor:"#fbbf24",color:"#fbbf24",fontSize:9}}>
          API offline — using local simulation. Start backend: uvicorn api.server:app
        </div>
      )}

      {/* NVML Status */}
      {nvmlStatus ? (
        <div style={{...P,marginTop:8}}>
          <div style={{fontSize:9,color:"#475569",textTransform:"uppercase",letterSpacing:.8,marginBottom:4}}>GPU Status (NVML)</div>
          <div style={{color:"#94a3b8",lineHeight:1.8}}>
            <div>{nvmlStatus.device_name}</div>
            <div>Clock: <span style={{color:"#e2e8f0"}}>{nvmlStatus.gpu_clock_mhz}</span> MHz</div>
            <div>Power: <span style={{color:"#e2e8f0"}}>{nvmlStatus.power_draw_w.toFixed(1)}</span> / {nvmlStatus.power_limit_w.toFixed(0)} W</div>
            <div>Temp: <span style={{color:"#e2e8f0"}}>{nvmlStatus.temperature_c}</span>C</div>
            <div>Mem: <span style={{color:"#e2e8f0"}}>{(nvmlStatus.mem_used_mb/1024).toFixed(1)}</span> / {(nvmlStatus.mem_total_mb/1024).toFixed(1)} GB</div>
            <div>GPU Util: <span style={{color:"#e2e8f0"}}>{nvmlStatus.gpu_utilization_pct}%</span></div>
          </div>
        </div>
      ) : nvmlUnavailable && (
        <div style={{...P,marginTop:8,borderColor:"#334155",color:"#64748b",fontSize:9}}>
          GPU status: NVML unavailable (no NVIDIA GPU)
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════
//  MAIN APP
// ═══════════════════════════════════════════════
export default function App() {
  const [hwName, setHwName] = useState("GB10 Blackwell");
  const [hw, setHw] = useState({...HW_PRESETS["GB10 Blackwell"]});
  const [modelName, setModelName] = useState("Llama-3 8B");
  const [cfgName, setCfgName] = useState("NVFP4 W4A4");
  const [phase, setPhase] = useState("decode");
  const [B, setB] = useState(1);
  const [S, setS] = useState(4096);
  const [showBands, setShowBands] = useState(true);
  const [pinned, setPinned] = useState([]);
  const [nlIn, setNlIn] = useState("");
  const [nlLoading, setNlLoading] = useState(false);
  const [nlNote, setNlNote] = useState("");
  const [fmtTab, setFmtTab] = useState("compare");
  const [selFmt, setSelFmt] = useState("NVFP4");
  const [measuredPoints, setMeasuredPoints] = useState([]);
  const [simulatedPoints, setSimulatedPoints] = useState([]);
  const [rightTab, setRightTab] = useState("kernel");
  const [chatInput, setChatInput] = useState("");
  const [chatMessages, setChatMessages] = useState([]);
  const [highlightedSection, setHighlightedSection] = useState(null);

  const applyHw = useCallback(n => { setHwName(n); setHw({...HW_PRESETS[n]}); }, []);
  const model = MODELS[modelName];
  const cfg = CONFIGS[cfgName];
  const raw = useMemo(() => computeOps(model, cfg, phase, B, S), [model, cfg, phase, B, S]);
  const agg = useMemo(() => aggregate(raw), [raw]);

  const totF = agg.reduce((s,o)=>s+o.flops,0), totB2 = agg.reduce((s,o)=>s+o.bytes,0);
  const aggAI = totF / totB2;
  const peakKey = hwFlopsKey(cfg.computeAs);
  const peakT = (hw.flops[peakKey] || 0);
  const aggPerf = attainable(aggAI, peakT, hw.bw);
  const totTimeS = totF / aggPerf;
  const toks = phase === "prefill" ? S : 1;
  const tokS = toks / totTimeS;
  const criticalAI = peakT > 0 && hw.bw > 0 ? (peakT * 1e12) / (hw.bw * 1e9) : 200;
  const bound = aggAI < criticalAI ? "MEMORY" : "COMPUTE";

  const pin = () => {
    const lbl = `${hwName.slice(0,6)}·${cfgName.slice(0,10)}·${phase[0]}${S}`;
    setPinned(p => [...p.slice(-3), { label: lbl, ops: agg, hw: {...hw} }]);
  };

  const doNl = async () => {
    if (!nlIn.trim()) return;
    setNlLoading(true); setNlNote("");
    const r = await parseNL(nlIn);
    setNlLoading(false);
    if (r && !r.error) {
      if (r.hwName && HW_PRESETS[r.hwName]) applyHw(r.hwName);
      if (r.bw) setHw(p => ({...p, bw: r.bw}));
      if (r.flopsOverrides) { setHw(p => ({...p, flops:{...p.flops,...r.flopsOverrides}})); setHwName("Custom ASIC"); }
      if (r.modelName && MODELS[r.modelName]) setModelName(r.modelName);
      if (r.precConfig && CONFIGS[r.precConfig]) setCfgName(r.precConfig);
      if (r.phase) setPhase(r.phase);
      if (r.batchSize) setB(r.batchSize);
      if (r.seqLen) setS(r.seqLen);
      if (r.notes) setNlNote(r.notes);
    }
  };

  const [chatLoading, setChatLoading] = useState(false);

  const buildChatContext = useCallback(() => {
    const lines = [
      `Hardware: ${hwName}, BW=${hw.bw} GB/s`,
      `Model: ${modelName}, phase=${phase}, batch=${B}, seq=${S}`,
      `Precision config: ${cfgName}`,
    ];
    if (agg.length > 0) {
      const totF = agg.reduce((s, o) => s + o.flops, 0);
      const totB = agg.reduce((s, o) => s + o.bytes, 0);
      const aggAI = totB > 0 ? totF / totB : 0;
      const peakT = hw.flops[hwFlopsKey(cfg?.computeAs)] || hw.flops.FP16 || 1;
      const critical = peakT > 0 && hw.bw > 0 ? (peakT * 1e12) / (hw.bw * 1e9) : 0;
      const bound = aggAI < critical ? "memory" : "compute";
      lines.push(`Aggregate: AI=${aggAI.toFixed(2)}, critical≈${critical.toFixed(0)}, bottleneck=${bound}`);
    }
    if (measuredPoints.length > 0) lines.push(`Measured points: ${measuredPoints.length} (last: ${measuredPoints[measuredPoints.length-1]?.label || "—"})`);
    if (simulatedPoints.length > 0) lines.push(`Predicted: ${simulatedPoints[0]?.label || "—"}`);
    return lines.join("\n");
  }, [hwName, hw, modelName, phase, B, S, cfgName, agg, cfg, measuredPoints, simulatedPoints]);

  const handleChatSend = async (queryOverride) => {
    const q = (queryOverride ?? chatInput).trim();
    if (!q) return;
    setChatInput("");
    setChatMessages(m => [...m, { role: "user", text: q }]);
    setChatLoading(true);
    const ruleRes = getOptimizationResponse(q);
    const ctx = buildChatContext();
    const ollamaRes = await askOllama(ctx, q);
    const text = ollamaRes || ruleRes.message;
    setChatMessages(m => [...m, { role: "assistant", text }]);
    setChatLoading(false);
    setHighlightedSection(ruleRes.highlightSection || null);
    if (ruleRes.highlightSection === "kernel_analyzer") setRightTab("kernel");
  };

  useEffect(() => {
    if (!highlightedSection) return;
    const el = document.querySelector(`[data-guide-id="${highlightedSection}"]`);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "nearest" });
    const t = setTimeout(() => setHighlightedSection(null), 4000);
    return () => clearTimeout(t);
  }, [highlightedSection]);

  const P = { background:"#0f172a", borderRadius:6, padding:12, marginBottom:8, border:"1px solid #1e293b" };
  const panelHighlight = (id) => highlightedSection === id ? { border: "2px solid #8b5cf6", boxShadow: "0 0 12px rgba(139,92,246,0.4)" } : {};
  const L = { fontSize:9, color:"#475569", textTransform:"uppercase", letterSpacing:.8, marginBottom:4, display:"block" };
  const sel = { background:"#1e293b", color:"#e2e8f0", border:"1px solid #334155", borderRadius:3, padding:"4px 6px", fontSize:11, width:"100%", fontFamily:"monospace", outline:"none" };
  const btn = a => ({ background:a?"#3b82f6":"#1e293b", color:a?"#fff":"#94a3b8", border:"1px solid "+(a?"#3b82f6":"#334155"), borderRadius:3, padding:"3px 8px", fontSize:10, cursor:"pointer", fontFamily:"monospace" });

  const FP4_FORMATS = ["MXFP4","NVFP4","NF4","INT4","INT2"];
  const ALL_FORMATS = ["FP32","BF16","FP16","FP8_E4M3","FP8_E5M2","MXFP8_E4M3","MXFP6_E3M2","MXFP6_E2M3","MXFP4","NVFP4","NF4","INT8","MXINT8","INT4","INT2"];

  return (
    <div style={{ fontFamily:"'IBM Plex Sans',system-ui,sans-serif", background:"#060a14", color:"#e2e8f0", minHeight:"100vh", padding:12 }}>
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet" />

      <div style={{ marginBottom:12, display:"flex", justifyContent:"space-between", alignItems:"center", borderBottom:"1px solid #1e293b", paddingBottom:8 }}>
        <div>
          <h1 style={{ fontSize:16, fontWeight:600, margin:0, fontFamily:"monospace", letterSpacing:-.5 }}>
            <span style={{ color:"#a855f7" }}>◈</span> Blackwell GEMM Roofline Analyzer <span style={{fontSize:10,color:"#64748b"}}>v3 — dual roofline + auto-quantizer</span>
          </h1>
          <p style={{ fontSize:10, color:"#475569", margin:"2px 0 0" }}>GB10 · FP8/FP4 native · simulated + measured roofline · NVML monitoring · auto-quantization</p>
        </div>
        <div style={{ display:"flex", gap:6, fontSize:9 }}>
          {Object.entries(TC).map(([t,c]) => <span key={t} style={{color:c}}>{t==="gemm"?"●":t==="attention"?"◆":"■"} {t}</span>)}
        </div>
      </div>

      {/* NL bar */}
      <div style={{ ...P, display:"flex", gap:6, alignItems:"center", padding:8, marginBottom:10 }}>
        <span style={{ color:"#64748b", fontSize:11 }}>⌘</span>
        <input value={nlIn} onChange={e=>setNlIn(e.target.value)} onKeyDown={e=>e.key==="Enter"&&doNl()}
          placeholder="e.g. 'o1 reasoning 32K decode on custom chip: 10TB/s BW, 12000 NVFP4 TOPS' or 'compare NVFP4 W4A4 vs MXFP4 W4A4 on B200'"
          style={{ ...sel, flex:1, background:"transparent", border:"none", fontSize:11 }} />
        <button onClick={doNl} disabled={nlLoading} style={{ ...btn(true), opacity:nlLoading?.5:1 }}>{nlLoading?"...":"Parse →"}</button>
      </div>
      {nlNote && <div style={{ fontSize:10, color:"#64748b", marginBottom:8, fontStyle:"italic", padding:"0 4px" }}>→ {nlNote}</div>}

      {/* Chat: optimization guidance */}
      <div style={{ ...P, padding:8, marginBottom:10 }} data-guide-id="chat">
        <div style={{ fontSize:9, color:"#475569", textTransform:"uppercase", letterSpacing:.8, marginBottom:6 }}>Ask about optimizations</div>
        <div style={{ display:"flex", gap:6, marginBottom:6 }}>
          <input value={chatInput} onChange={e=>setChatInput(e.target.value)} onKeyDown={e=>e.key==="Enter"&&handleChatSend()}
            placeholder="e.g. How can I speed up decode?"
            style={{ ...sel, flex:1, background:"transparent", border:"1px solid #334155", fontSize:11 }} />
          <button onClick={()=>handleChatSend()} disabled={chatLoading} style={{ ...btn(true), flexShrink:0, opacity: chatLoading ? 0.6 : 1 }}>{chatLoading ? "..." : "Send"}</button>
        </div>
        <div style={{ display:"flex", flexWrap:"wrap", gap:4, marginBottom:6 }}>
          {["How can I speed up decode?", "What should I tune?", "Why is it memory bound?", "What precision for long context?"].map((q,i)=>(
            <button key={i} onClick={()=>handleChatSend(q)} disabled={chatLoading} style={{ ...btn(false), fontSize:9, padding:"2px 6px" }}>{q}</button>
          ))}
        </div>
        <div style={{ fontSize:8, color:"#64748b" }}>Uses Ollama when available (run: ollama pull llama3.2)</div>
        {chatMessages.length > 0 && (
          <div style={{ maxHeight:120, overflowY:"auto", fontSize:10, lineHeight:1.5 }}>
            {chatMessages.slice(-4).map((m,i)=>(
              <div key={i} style={{ marginBottom:4, color:m.role==="user"?"#94a3b8":"#e2e8f0" }}>
                <span style={{ color:"#64748b", marginRight:4 }}>{m.role==="user"?"You:":"→"}</span>{m.text}
              </div>
            ))}
          </div>
        )}
      </div>

      <div style={{ display:"flex", gap:10 }}>
        {/* ═══ LEFT SIDEBAR ═══ */}
        <div style={{ width:250, flexShrink:0 }}>
          {/* HW */}
          <div style={P}>
            <span style={L}>Hardware</span>
            <select value={hwName} onChange={e=>applyHw(e.target.value)} style={{...sel,marginBottom:4}}>{Object.keys(HW_PRESETS).map(k=><option key={k}>{k}</option>)}</select>
            <div style={{fontSize:8,color:"#475569",marginBottom:6}}>{hw.note}</div>
            <SliderC l="Bandwidth" v={hw.bw} u=" GB/s" min={10} max={20000} step={10} onChange={v=>{setHw(p=>({...p,bw:v}));setHwName("Custom ASIC")}} />
            {["FP16","FP8_E4M3","NVFP4","MXFP4","INT4"].map(p => (
              <SliderC key={p} l={`Peak ${(FORMATS[p]?.label||p).replace("(OCP)","")}`} v={hw.flops[p]||0} u=" T" min={0} max={20000} step={1}
                onChange={v=>{setHw(p2=>({...p2,flops:{...p2.flops,[p]:v}}));setHwName("Custom ASIC")}} />
            ))}
            <label style={{fontSize:9,color:"#64748b",display:"flex",alignItems:"center",gap:3,cursor:"pointer",marginTop:2}}>
              <input type="checkbox" checked={showBands} onChange={e=>setShowBands(e.target.checked)} /> Uncertainty bands
            </label>
          </div>

          {/* Workload */}
          <div style={{ ...P, ...panelHighlight("workload") }} data-guide-id="workload">
            <span style={L}>Workload</span>
            <select value={modelName} onChange={e=>setModelName(e.target.value)} style={{...sel,marginBottom:4}}>{Object.keys(MODELS).map(k=><option key={k}>{k}</option>)}</select>
            <div data-guide-id="precision_config" style={highlightedSection === "precision_config" ? { ...panelHighlight("precision_config"), borderRadius:4, padding:2, marginBottom:4 } : { marginBottom:4 }}>
              <select value={cfgName} onChange={e=>{setCfgName(e.target.value); setHighlightedSection(null);}} style={{...sel,width:"100%"}}>{Object.keys(CONFIGS).map(k=><option key={k}>{k}</option>)}</select>
            </div>
            <div style={{display:"flex",gap:3,marginBottom:6}}>
              {["prefill","decode"].map(p=><button key={p} onClick={()=>setPhase(p)} style={btn(phase===p)}>{p}</button>)}
            </div>
            <SliderC l="Context" v={S} u=" tok" min={32} max={131072} log onChange={v=>setS(Math.round(v))} />
            <SliderC l="Batch" v={B} min={1} max={512} log onChange={v=>setB(Math.round(v))} />
            {/* Show active precision breakdown */}
            <div style={{fontSize:9,color:"#64748b",marginTop:4,lineHeight:1.6}}>
              <div>W: <span style={{color:FORMATS[cfg.w]?.color||"#fff"}}>{FORMATS[cfg.w]?.label}</span> ({effectiveBitsPerElement(cfg.w).toFixed(2)}b)</div>
              <div>A: <span style={{color:FORMATS[cfg.a]?.color||"#fff"}}>{FORMATS[cfg.a]?.label}</span> ({effectiveBitsPerElement(cfg.a).toFixed(2)}b)</div>
              <div>KV: <span style={{color:FORMATS[cfg.kv]?.color||"#fff"}}>{FORMATS[cfg.kv]?.label}</span> ({effectiveBitsPerElement(cfg.kv).toFixed(2)}b)</div>
              <div>Compute: <span style={{color:"#e2e8f0"}}>{cfg.computeAs}</span> → peak {peakT.toFixed(0)}T</div>
            </div>
          </div>

          {/* Pin */}
          <div style={P}>
            <span style={L}>Compare</span>
            <button onClick={pin} style={{...btn(false),width:"100%",marginBottom:4}}>📌 Pin current</button>
            {pinned.map((ps,i)=><div key={i} style={{display:"flex",justifyContent:"space-between",fontSize:9,color:"#94a3b8",padding:"1px 0"}}>
              <span>{ps.label}</span><button onClick={()=>setPinned(p=>p.filter((_,j)=>j!==i))} style={{background:"none",border:"none",color:"#64748b",cursor:"pointer",fontSize:9}}>✕</button>
            </div>)}
          </div>

          {/* What to do — quantization recommendation */}
          <div style={{ ...P, ...panelHighlight("what_to_do") }} data-guide-id="what_to_do">
            <span style={L}>What to do</span>
            <WhatToDoPanel bound={bound} aggAI={aggAI} hw={hw} hwName={hwName} cfg={cfg} peakT={peakT} />
          </div>
        </div>

        {/* ═══ CENTER ═══ */}
        <div style={{ flex:1, minWidth:0 }}>
          {/* Summary */}
          <div style={{ ...P, display:"flex", gap:16, alignItems:"center", padding:"8px 14px", flexWrap:"wrap" }}>
            {[
              {l:"Agg AI", v:aggAI.toFixed(2), u:" F/B"},
              {l:"Total", v:(totF/1e12).toFixed(2), u:" TFLOP"},
              {l:"Bytes", v:totB2>1e9?(totB2/1e9).toFixed(2)+" GB":(totB2/1e6).toFixed(1)+" MB"},
              {l:"Est Time", v:totTimeS<.001?(totTimeS*1e6).toFixed(0)+"μs":totTimeS<1?(totTimeS*1e3).toFixed(1)+"ms":totTimeS.toFixed(2)+"s"},
              {l:"Throughput", v:tokS>1e6?(tokS/1e6).toFixed(1)+"M":tokS>1e3?(tokS/1e3).toFixed(1)+"K":tokS.toFixed(1), u:" tok/s"},
              {l:"Bound", v:bound, c:bound==="COMPUTE"?"#f87171":"#60a5fa"},
              {l:"W bits/elem", v:effectiveBitsPerElement(cfg.w).toFixed(2)},
              {l:"KV bits/elem", v:effectiveBitsPerElement(cfg.kv).toFixed(2)},
            ].map((m,i)=><div key={i} style={{textAlign:"center"}}>
              <div style={{fontSize:8,color:"#475569",textTransform:"uppercase",letterSpacing:.4}}>{m.l}</div>
              <div style={{fontSize:13,fontWeight:600,fontFamily:"monospace",color:m.c||"#e2e8f0"}}>{m.v}<span style={{fontSize:9,color:"#475569"}}>{m.u||""}</span></div>
            </div>)}
          </div>

          {/* Roofline */}
          <div style={{ ...P, ...panelHighlight("roofline_plot") }} data-guide-id="roofline_plot">
            <RooflinePlot ops={agg} hw={hw} pinned={pinned} showBands={showBands} measuredPoints={measuredPoints} simulatedPoints={simulatedPoints} />
          </div>

          {/* Op Table */}
          <div style={{ ...P, ...panelHighlight("op_table") }} data-guide-id="op_table">
            <span style={L}>Operators by time</span>
            <OpTable ops={agg} hw={hw} />
          </div>
        </div>

        {/* ═══ RIGHT: TABS — Kernel Spot Check | Format Reference ═══ */}
        <div style={{ width:280, flexShrink:0 }} data-guide-id="right_sidebar">
          <div style={{display:"flex",gap:3,marginBottom:8,flexWrap:"wrap"}}>
            <button onClick={()=>setRightTab("kernel")} style={btn(rightTab==="kernel")}>Kernel</button>
            <button onClick={()=>setRightTab("sweep")} style={btn(rightTab==="sweep")}>GB10 Sweep</button>
            <button onClick={()=>setRightTab("benchmark")} style={btn(rightTab==="benchmark")}>Benchmark</button>
            <button onClick={()=>setRightTab("format")} style={btn(rightTab==="format")}>Format</button>
          </div>
          {rightTab === "benchmark" && (
            <BenchmarkPanel
              hwKey={HW_NAME_TO_API_KEY[hwName] || "b10"}
              onMeasuredPoints={(pts) => setMeasuredPoints(prev => [...prev, ...pts])}
            />
          )}
          {rightTab === "sweep" && (
            <SweepPanel
              hwKey={HW_NAME_TO_API_KEY[hwName] || "b10"}
              onPoints={(sim, meas) => { setSimulatedPoints(sim); setMeasuredPoints(meas); }}
            />
          )}
          {rightTab === "kernel" && (
            <div data-guide-id="kernel_analyzer" style={panelHighlight("kernel_analyzer")}>
              {(measuredPoints.length > 0 || simulatedPoints.length > 0) && (
                <button onClick={()=>{ setMeasuredPoints([]); setSimulatedPoints([]); }} style={{...btn(false),marginBottom:6,width:"100%",fontSize:9}}>Clear predicted & measured</button>
              )}
              <GEMMAnalyzer
                hw={hw}
                hwKey={HW_NAME_TO_API_KEY[hwName] || "b10"}
                onMeasuredPoints={(pts) => setMeasuredPoints(prev => [...prev, ...pts])}
                onSimulatedPoints={(pts) => setSimulatedPoints(pts)}
              />
            </div>
          )}
          {rightTab === "format" && (
          <div style={P}>
            <span style={L}>Format Reference</span>
            <div style={{display:"flex",gap:3,marginBottom:8}}>
              <button onClick={()=>setFmtTab("compare")} style={btn(fmtTab==="compare")}>Compare</button>
              <button onClick={()=>setFmtTab("detail")} style={btn(fmtTab==="detail")}>Detail</button>
              <button onClick={()=>setFmtTab("fp4")} style={btn(fmtTab==="fp4")}>FP4 Deep</button>
            </div>

            {fmtTab === "compare" && <FormatCompareTable formats={ALL_FORMATS} />}

            {fmtTab === "detail" && (
              <div>
                <select value={selFmt} onChange={e=>setSelFmt(e.target.value)} style={{...sel,marginBottom:6}}>
                  {ALL_FORMATS.map(f=><option key={f} value={f}>{FORMATS[f].label}</option>)}
                </select>
                <FormatDetail fmt={selFmt} />
              </div>
            )}

            {fmtTab === "fp4" && (
              <div style={{ fontSize:10, fontFamily:"monospace", color:"#94a3b8", lineHeight:1.8 }}>
                <div style={{ fontWeight:600, color:"#a855f7", marginBottom:6 }}>NVFP4 vs MXFP4 — the key differences</div>

                <div style={{ marginBottom:8 }}>
                  <div style={{ color:"#e2e8f0", marginBottom:2 }}>Block size & scale overhead:</div>
                  <div>MXFP4: 32 elem/block · E8M0 scale (power-of-2)</div>
                  <div>→ {(4 + 8/32).toFixed(3)}b/elem · {((8/32)/(4+8/32)*100).toFixed(1)}% overhead</div>
                  <div style={{marginTop:2}}>NVFP4: 16 elem/block · E4M3 scale + FP32/tensor</div>
                  <div>→ {(4 + 8/16).toFixed(3)}b/elem · {((8/16)/(4+8/16)*100).toFixed(1)}% overhead</div>
                  <div style={{marginTop:4,color:"#fbbf24"}}>NVFP4 costs ~{((4.5-4.25)/4.25*100).toFixed(1)}% more bytes</div>
                  <div style={{color:"#22c55e"}}>...but E4M3 scale gives non-power-of-2 resolution</div>
                </div>

                <div style={{ marginBottom:8 }}>
                  <div style={{ color:"#e2e8f0", marginBottom:2 }}>Dynamic range comparison:</div>
                  <div>MXFP4: E8M0 scale → ×2^(-127..127)</div>
                  <div>  Effective: ±6 × 2^127 ≈ ±1e38</div>
                  <div style={{marginTop:2}}>NVFP4: E4M3 scale → ×(0..448)</div>
                  <div>  Block: ±6 × 448 = ±2688</div>
                  <div>  + FP32 tensor: ±2688 × 3.4e38</div>
                  <div style={{marginTop:4,color:"#22c55e"}}>NVFP4 has finer granularity within blocks</div>
                  <div style={{color:"#22c55e"}}>MXFP4 has wider single-level range</div>
                </div>

                <div style={{ marginBottom:8 }}>
                  <div style={{ color:"#e2e8f0", marginBottom:2 }}>FP4 E2M1 codebook (shared):</div>
                  <div style={{display:"flex",flexWrap:"wrap",gap:2,marginTop:2}}>
                    {[0,.5,1,1.5,2,3,4,6].map((v,i)=><span key={i} style={{background:"#1e293b",padding:"1px 4px",borderRadius:2,color:"#c084fc",fontSize:9}}>{v}</span>)}
                  </div>
                  <div style={{marginTop:4}}>Step sizes: 0.5, 0.5, 0.5, 0.5, 1, 1, 2</div>
                  <div style={{color:"#fbbf24"}}>Non-uniform: fine near 0, coarse near max</div>
                  <div style={{marginTop:2}}>4/6 algorithm: some blocks scale to max=4</div>
                  <div>→ steps become 0.5,0.5,0.5,0.5,1,1</div>
                  <div style={{color:"#22c55e"}}>Trades range for resolution at boundaries</div>
                </div>

                <div style={{ marginBottom:8 }}>
                  <div style={{ color:"#e2e8f0", marginBottom:2 }}>W4A4 vs W4A16 — compute path:</div>
                  <div>W4A16: weights FP4, activations FP16</div>
                  <div style={{color:"#fbbf24"}}>→ Must dequant W to FP16 before TC</div>
                  <div>→ Compute runs at FP16 peak</div>
                  <div style={{color:"#22c55e"}}>→ Only bandwidth benefit from FP4 W</div>
                  <div style={{marginTop:4}}>W4A4: both at FP4, native TC path</div>
                  <div style={{color:"#22c55e"}}>→ 2× compute peak vs FP8</div>
                  <div style={{color:"#22c55e"}}>→ Full BW + compute benefit</div>
                  <div style={{marginTop:2,color:"#f87171"}}>But: activation quantization is harder</div>
                  <div style={{color:"#f87171"}}>Dynamic scaling needed per-token</div>
                </div>

                <div>
                  <div style={{ color:"#e2e8f0", marginBottom:2 }}>KV Cache at FP4:</div>
                  <div>NVFP4 KV: 4× compression vs FP16</div>
                  <div>  2× vs FP8 KV (already production)</div>
                  <div>  Dequant to FP8 before attention</div>
                  <div style={{color:"#22c55e"}}>Doubles effective context budget</div>
                  <div style={{color:"#22c55e"}}>Critical for o1/o3 reasoning chains</div>
                </div>
              </div>
            )}
          </div>
          )}
        </div>
      </div>
    </div>
  );
}

function SliderC({l,v,onChange,min,max,step,log,u}) {
  const act = log ? Math.log10(Math.max(v,0.01)) : v;
  const lo = log ? Math.log10(min) : min;
  const hi = log ? Math.log10(max) : max;
  const st = log ? .01 : (step||1);
  const fmt = v => v>=1e6?(v/1e6).toFixed(1)+"M":v>=1e3?(v/1e3).toFixed(1)+"K":Number.isInteger(v)?v:v.toFixed(1);
  return <div style={{marginBottom:6}}>
    <div style={{display:"flex",justifyContent:"space-between",fontSize:9,color:"#64748b",marginBottom:1}}>
      <span>{l}</span><span style={{color:"#e2e8f0",fontFamily:"monospace"}}>{fmt(v)}{u||""}</span>
    </div>
    <input type="range" min={lo} max={hi} step={st} value={act}
      onChange={e=>onChange(log?Math.pow(10,parseFloat(e.target.value)):parseFloat(e.target.value))}
      style={{width:"100%",accentColor:"#8b5cf6",height:3}} />
  </div>;
}
