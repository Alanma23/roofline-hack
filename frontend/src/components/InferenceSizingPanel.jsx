import { useEffect, useMemo, useState } from "react";
import { computeSizing } from "../sizing/rooflineSizing";
import { recommendSizingConfigs } from "../sizing/recommend";

const CUSTOM_MODEL = "Custom model";

const STAGE_COLORS = ["#38bdf8", "#22c55e", "#f59e0b", "#a855f7", "#ef4444", "#06b6d4"];

const EFFECTS_CSS = `
@keyframes podPulse {
  0% { transform: translateY(0px); box-shadow: 0 0 0 rgba(56,189,248,0.0); }
  50% { transform: translateY(-1px); box-shadow: 0 0 14px rgba(56,189,248,0.28); }
  100% { transform: translateY(0px); box-shadow: 0 0 0 rgba(56,189,248,0.0); }
}
@keyframes fabricFlow {
  0% { background-position: 0% 0%; }
  100% { background-position: 120% 0%; }
}
@keyframes linkDash {
  0% { stroke-dashoffset: 0; opacity: 0.55; }
  100% { stroke-dashoffset: -26; opacity: 1; }
}
@keyframes lanePulse {
  0% { opacity: 0.5; }
  50% { opacity: 1; }
  100% { opacity: 0.5; }
}
`;

function numberValue(value, fallback, min = 0) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, n);
}

function intValue(value, fallback, min = 1) {
  return Math.max(min, Math.round(numberValue(value, fallback, min)));
}

function fmtBytes(bytes) {
  if (!Number.isFinite(bytes)) return "-";
  if (bytes >= 1e12) return `${(bytes / 1e12).toFixed(2)} TB`;
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(2)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(2)} KB`;
  return `${bytes.toFixed(0)} B`;
}

function fmtMs(value) {
  if (!Number.isFinite(value)) return "-";
  if (value < 0.001) return `${(value * 1000).toFixed(2)} us`;
  return `${value.toFixed(3)} ms`;
}

function TimeBar({ label, value, maxValue, color }) {
  const widthPct = maxValue > 0 ? Math.max(2, (value / maxValue) * 100) : 2;
  return (
    <div style={{ marginBottom: 5 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 8, color: "#94a3b8" }}>
        <span>{label}</span>
        <span>{fmtMs(value)}</span>
      </div>
      <div style={{ background: "#1e293b", borderRadius: 3, height: 6, overflow: "hidden" }}>
        <div style={{ width: `${widthPct}%`, height: "100%", background: color }} />
      </div>
    </div>
  );
}

function stageStatsFromResult(result) {
  const stages = (result?.metadata?.stage_layout || [])
    .filter((s) => s.count > 0)
    .map((s) => ({
      stage: s.stage + 1,
      start: s.start + 1,
      end: s.end + 1,
      tpBytes: 0,
      ppBytes: 0,
    }));

  const byStage = new Map(stages.map((s, idx) => [idx + 1, s]));
  for (const row of (result?.layer_io || [])) {
    const stage = byStage.get(row.stage);
    if (!stage) continue;
    stage.tpBytes += row.tp_sync_bytes || 0;
    stage.ppBytes += row.pp_boundary_send_bytes || 0;
  }
  return stages;
}

function estimateKvCacheBytes(model, batch, contextTokens, kvBytes) {
  if (!model) return 0;
  const B = intValue(batch, 1);
  const S = intValue(contextTokens, 1);
  const L = intValue(model.L, 1);
  const nkv = intValue(model.nkv, 1);
  const dh = intValue(model.dh, 1);
  const bytes = numberValue(kvBytes, 2.0, 0);
  return B * S * L * nkv * dh * 2 * bytes;
}

function StageFlow({ stageStats, tp }) {
  if (!stageStats.length) return null;
  return (
    <div style={{ display: "flex", gap: 6, overflowX: "auto", paddingBottom: 2 }}>
      {stageStats.map((stage, idx) => (
        <div key={stage.stage} style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <div style={{ minWidth: 130, border: "1px solid #334155", borderRadius: 6, padding: 6, background: "#111827" }}>
            <div style={{ fontSize: 9, color: "#e2e8f0", fontWeight: 600 }}>Stage {stage.stage}</div>
            <div style={{ fontSize: 8, color: "#94a3b8" }}>layers {stage.start}..{stage.end}</div>
            <div style={{ fontSize: 8, color: "#60a5fa" }}>TP shards: {tp}</div>
            <div style={{ fontSize: 8, color: "#f59e0b" }}>AR bytes: {fmtBytes(stage.tpBytes)}</div>
          </div>
          {idx < stageStats.length - 1 && (
            <div style={{ minWidth: 84, textAlign: "center" }}>
              <div style={{ color: "#64748b", fontSize: 10 }}>→</div>
              <div style={{ color: "#22c55e", fontSize: 8 }}>PP send</div>
              <div style={{ color: "#94a3b8", fontSize: 8 }}>{fmtBytes(stage.ppBytes)}</div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function PodVisualizer({ podSize, tp, pp, bottleneck }) {
  const lanes = Math.max(1, intValue(tp, 1));
  const stages = Math.max(1, intValue(pp, 1));
  const used = Math.min(podSize, lanes * stages);
  const cols = 4;
  const rows = Math.max(1, Math.ceil(podSize / cols));

  const cellW = 86;
  const cellH = 52;
  const gapX = 12;
  const gapY = 12;

  const chips = Array.from({ length: podSize }, (_, i) => {
    const active = i < used;
    const stage = active ? Math.floor(i / lanes) + 1 : 0;
    const lane = active ? (i % lanes) + 1 : 0;
    const color = active ? STAGE_COLORS[(stage - 1) % STAGE_COLORS.length] : "#334155";
    const col = i % cols;
    const row = Math.floor(i / cols);
    const x = col * (cellW + gapX);
    const y = row * (cellH + gapY);
    const cx = x + (cellW / 2);
    const cy = y + (cellH / 2);
    return {
      index: i + 1,
      active,
      stage,
      lane,
      color,
      x,
      y,
      cx,
      cy,
    };
  });

  const active = chips.filter((c) => c.active);
  const svgW = cols * cellW + (cols - 1) * gapX;
  const svgH = rows * cellH + (rows - 1) * gapY;

  const tpLinks = [];
  for (let stage = 0; stage < stages; stage += 1) {
    const stageChips = active.slice(stage * lanes, stage * lanes + lanes);
    if (stageChips.length <= 1) continue;
    for (let i = 0; i < stageChips.length; i += 1) {
      const from = stageChips[i];
      const to = stageChips[(i + 1) % stageChips.length];
      tpLinks.push({
        id: `tp-${stage + 1}-${i}`,
        x1: from.cx,
        y1: from.cy,
        x2: to.cx,
        y2: to.cy,
        color: STAGE_COLORS[stage % STAGE_COLORS.length],
      });
    }
  }

  const ppLinks = [];
  for (let stage = 0; stage < stages - 1; stage += 1) {
    for (let lane = 0; lane < lanes; lane += 1) {
      const from = active[stage * lanes + lane];
      const to = active[(stage + 1) * lanes + lane];
      if (!from || !to) continue;
      ppLinks.push({
        id: `pp-${stage + 1}-${lane + 1}`,
        x1: from.cx,
        y1: from.cy,
        x2: to.cx,
        y2: to.cy,
      });
    }
  }

  return (
    <div style={{ position: "relative" }}>
      <style>{EFFECTS_CSS}</style>
      <div
        style={{
          position: "absolute",
          inset: 0,
          borderRadius: 8,
          background:
            "linear-gradient(90deg, rgba(56,189,248,0.03), rgba(34,197,94,0.04), rgba(168,85,247,0.03), rgba(56,189,248,0.03))",
          backgroundSize: "200% 100%",
          animation: "fabricFlow 6s linear infinite",
          pointerEvents: "none",
        }}
      />
      <div style={{ position: "relative", zIndex: 1 }}>
        <div style={{ fontSize: 8, color: "#94a3b8", marginBottom: 6 }}>
          Pod utilization: {used}/{podSize} ASICs · layout {rows}x4
        </div>
        <div style={{ position: "relative", minHeight: svgH }}>
          <svg
            width="100%"
            viewBox={`0 0 ${svgW} ${svgH}`}
            preserveAspectRatio="none"
            style={{ position: "absolute", inset: 0, pointerEvents: "none" }}
          >
            {ppLinks.map((link, idx) => (
              <line
                key={link.id}
                x1={link.x1}
                y1={link.y1}
                x2={link.x2}
                y2={link.y2}
                stroke="#22c55e"
                strokeWidth="1.2"
                strokeDasharray="4 8"
                style={{ animation: "lanePulse 2.4s ease-in-out infinite", animationDelay: `${idx * 80}ms` }}
              />
            ))}
            {tpLinks.map((link, idx) => (
              <line
                key={link.id}
                x1={link.x1}
                y1={link.y1}
                x2={link.x2}
                y2={link.y2}
                stroke={link.color}
                strokeWidth="1.3"
                strokeDasharray="6 10"
                style={{ animation: "linkDash 1.3s linear infinite", animationDelay: `${idx * 60}ms` }}
              />
            ))}
          </svg>
          <div style={{ position: "relative", display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: 6 }}>
          {chips.map((chip) => (
            <div
              key={chip.index}
              style={{
                border: `1px solid ${chip.color}`,
                borderRadius: 6,
                padding: 6,
                minHeight: 50,
                background: chip.active ? "rgba(15,23,42,0.9)" : "rgba(15,23,42,0.55)",
                animation: chip.active ? "podPulse 2.8s ease-in-out infinite" : "none",
                animationDelay: `${chip.index * 90}ms`,
              }}
            >
              <div style={{ fontSize: 8, color: "#e2e8f0", fontWeight: 600 }}>ASIC {chip.index}</div>
              {chip.active ? (
                <>
                  <div style={{ fontSize: 8, color: chip.color }}>Stage {chip.stage}</div>
                  <div style={{ fontSize: 8, color: "#94a3b8" }}>TP lane {chip.lane}</div>
                </>
              ) : (
                <div style={{ fontSize: 8, color: "#64748b" }}>idle</div>
              )}
            </div>
          ))}
          </div>
        </div>
        <div style={{ fontSize: 8, color: "#94a3b8", marginTop: 6 }}>
          TP ring links: {tpLinks.length} · PP links: {ppLinks.length} · Bottleneck focus: <span style={{ color: bottleneck === "network" ? "#f59e0b" : bottleneck === "compute" ? "#ef4444" : "#60a5fa" }}>{bottleneck}</span>
        </div>
      </div>
    </div>
  );
}

export default function InferenceSizingPanel({
  models = {},
  configs = {},
  hardwarePresets = {},
  currentModelName = "Llama-3 8B",
  currentConfigName = "NVFP4 W4A4",
  currentHardwareName = "GB10 Blackwell",
  currentPhase = "decode",
  currentBatch = 1,
  currentSeqLen = 4096,
  bytesPerElement,
  hwFlopsKey,
}) {
  const modelNames = useMemo(() => Object.keys(models), [models]);
  const configNames = useMemo(() => Object.keys(configs), [configs]);
  const hardwareNames = useMemo(() => Object.keys(hardwarePresets), [hardwarePresets]);

  const fallbackModelName = modelNames[0] || CUSTOM_MODEL;
  const fallbackConfigName = configNames[0] || "";
  const fallbackHardwareName = hardwareNames[0] || "";

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showLayerDetails, setShowLayerDetails] = useState(false);
  const [showDualPhase, setShowDualPhase] = useState(true);
  const [enableE2E, setEnableE2E] = useState(true);
  const [viewMode, setViewMode] = useState(currentPhase === "prefill" ? "prefill" : "decode");

  const [modelChoice, setModelChoice] = useState(
    modelNames.includes(currentModelName) ? currentModelName : fallbackModelName,
  );
  const [customModel, setCustomModel] = useState(
    models[currentModelName] || models[fallbackModelName] || {
      L: 32, H: 4096, nh: 32, nkv: 8, dh: 128, dff: 14336, V: 128256, gate: true,
    },
  );

  const [configChoice, setConfigChoice] = useState(
    configNames.includes(currentConfigName) ? currentConfigName : fallbackConfigName,
  );

  const [phase, setPhase] = useState(currentPhase === "prefill" ? "prefill" : "decode");
  const [batch, setBatch] = useState(intValue(currentBatch, 1));
  const [seqLen, setSeqLen] = useState(intValue(currentSeqLen, 4096));
  const [decodeContextTokens, setDecodeContextTokens] = useState(intValue(currentSeqLen, 4096));
  const [promptTokens, setPromptTokens] = useState(intValue(currentSeqLen, 4096));
  const [decodeTokens, setDecodeTokens] = useState(256);
  const [decodeUsesPrefillContext, setDecodeUsesPrefillContext] = useState(true);

  const [hardwareChoice, setHardwareChoice] = useState(
    hardwareNames.includes(currentHardwareName) ? currentHardwareName : fallbackHardwareName,
  );
  const [memBw, setMemBw] = useState(numberValue(hardwarePresets[currentHardwareName]?.bw, 287, 1));
  const [hardwareFlops, setHardwareFlops] = useState(
    { ...(hardwarePresets[currentHardwareName]?.flops || hardwarePresets[fallbackHardwareName]?.flops || { FP16: 62 }) },
  );

  const [memoryMode, setMemoryMode] = useState("single");
  const [onchipBw, setOnchipBw] = useState(120000);
  const [offchipBw, setOffchipBw] = useState(numberValue(hardwarePresets[currentHardwareName]?.bw, 287, 1));
  const [onchipHitRate, setOnchipHitRate] = useState(0.85);

  const [tp, setTp] = useState(2);
  const [pp, setPp] = useState(1);
  const [podSize, setPodSize] = useState(8);
  const [constrainToPod, setConstrainToPod] = useState(true);
  const [podTargetMode, setPodTargetMode] = useState("at_most");
  const [maxAsics, setMaxAsics] = useState(16);
  const [tpLinkBw, setTpLinkBw] = useState(900);
  const [ppLinkBw, setPpLinkBw] = useState(900);
  const [tpLinkLatUs, setTpLinkLatUs] = useState(3);
  const [ppLinkLatUs, setPpLinkLatUs] = useState(3);
  const [overlap, setOverlap] = useState(0);

  const [e2ePrefillHardware, setE2ePrefillHardware] = useState(
    hardwareNames.includes(currentHardwareName) ? currentHardwareName : fallbackHardwareName,
  );
  const [e2eDecodeHardware, setE2eDecodeHardware] = useState(
    hardwareNames.includes(currentHardwareName) ? currentHardwareName : fallbackHardwareName,
  );
  const [handoffBwGBs, setHandoffBwGBs] = useState(900);
  const [handoffLatencyUs, setHandoffLatencyUs] = useState(5);

  useEffect(() => {
    const preset = hardwarePresets[hardwareChoice];
    if (!preset) return;
    const baseBw = numberValue(preset.bw, 287, 1);
    setMemBw(baseBw);
    setOffchipBw(baseBw);
    setOnchipBw(Math.max(baseBw * 32, 10000));
    setHardwareFlops({ ...(preset.flops || { FP16: 62 }) });
  }, [hardwareChoice, hardwarePresets]);

  useEffect(() => {
    if (viewMode === "prefill" && phase !== "prefill") setPhase("prefill");
    if (viewMode === "decode" && phase !== "decode") setPhase("decode");
  }, [viewMode, phase]);

  const activeModel = useMemo(() => (
    modelChoice === CUSTOM_MODEL ? customModel : (models[modelChoice] || customModel)
  ), [modelChoice, models, customModel]);

  const activePrecision = useMemo(
    () => configs[configChoice] || { w: "FP16", a: "FP16", kv: "FP16", computeAs: "FP16" },
    [configs, configChoice],
  );

  const computeKey = useMemo(() => (
    typeof hwFlopsKey === "function" ? hwFlopsKey(activePrecision.computeAs) : activePrecision.computeAs
  ), [activePrecision, hwFlopsKey]);

  const kvBytesPerElement = useMemo(
    () => (
      typeof bytesPerElement === "function"
        ? numberValue(bytesPerElement(activePrecision.kv), 2.0, 0)
        : 2.0
    ),
    [bytesPerElement, activePrecision],
  );

  const effectiveDecodeContext = useMemo(
    () => intValue(decodeContextTokens, seqLen, 1),
    [decodeContextTokens, seqLen],
  );

  const effectiveSeqLen = useMemo(
    () => (phase === "decode" ? effectiveDecodeContext : intValue(seqLen, 1)),
    [phase, effectiveDecodeContext, seqLen],
  );

  const maxAsicsEffective = constrainToPod ? intValue(podSize, 8) : intValue(maxAsics, 16);

  const presetHardwareSpec = (choice, fallbackHw) => {
    const preset = hardwarePresets[choice] || {};
    return {
      name: choice,
      peak_tflops: { ...(preset.flops || fallbackHw?.peak_tflops || hardwareFlops) },
      mem_bw_gbs: numberValue(preset.bw, fallbackHw?.mem_bw_gbs || memBw, 0.1),
    };
  };

  const request = useMemo(() => ({
    workload: {
      phase,
      batch: intValue(batch, 1),
      seq_len: effectiveSeqLen,
      model: {
        L: intValue(activeModel.L, 1),
        H: intValue(activeModel.H, 1),
        nh: intValue(activeModel.nh, 1),
        nkv: intValue(activeModel.nkv, 1),
        dh: intValue(activeModel.dh, 1),
        dff: intValue(activeModel.dff, 1),
        V: intValue(activeModel.V, 1),
        gate: Boolean(activeModel.gate),
      },
      precision: activePrecision,
    },
    hardware: {
      name: hardwareChoice,
      peak_tflops: hardwareFlops,
      mem_bw_gbs: numberValue(memBw, 1, 0.1),
      memory_model: memoryMode === "two_tier"
        ? {
            mode: "two_tier",
            onchip_bw_gbs: numberValue(onchipBw, 1, 0.1),
            offchip_bw_gbs: numberValue(offchipBw, 1, 0.1),
            onchip_hit_rate: Math.max(0, Math.min(1, numberValue(onchipHitRate, 0.85, 0))),
          }
        : undefined,
    },
    parallel: {
      tp: intValue(tp, 1),
      pp: intValue(pp, 1),
      max_asics: maxAsicsEffective,
    },
    network: {
      tp_link_bw_gbs: numberValue(tpLinkBw, 900, 0.1),
      tp_link_latency_us: numberValue(tpLinkLatUs, 3, 0),
      pp_link_bw_gbs: numberValue(ppLinkBw, tpLinkBw, 0.1),
      pp_link_latency_us: numberValue(ppLinkLatUs, tpLinkLatUs, 0),
      overlap_fraction: Math.max(0, Math.min(1, numberValue(overlap, 0, 0))),
    },
  }), [
    phase, batch, effectiveSeqLen, activeModel, activePrecision,
    hardwareChoice, hardwareFlops, memBw,
    memoryMode, onchipBw, offchipBw, onchipHitRate,
    tp, pp, maxAsicsEffective, tpLinkBw, ppLinkBw, tpLinkLatUs, ppLinkLatUs, overlap,
  ]);

  const sizingOptions = useMemo(() => ({ bytesPerElement, hwFlopsKey }), [bytesPerElement, hwFlopsKey]);
  const result = useMemo(() => computeSizing(request, sizingOptions), [request, sizingOptions]);

  const oppositePhase = phase === "decode" ? "prefill" : "decode";
  const oppositeRequest = useMemo(
    () => ({ ...request, workload: { ...request.workload, phase: oppositePhase } }),
    [request, oppositePhase],
  );
  const oppositeResult = useMemo(
    () => computeSizing(oppositeRequest, sizingOptions),
    [oppositeRequest, sizingOptions],
  );

  const rawRecommendations = useMemo(
    () => recommendSizingConfigs(request, {
      ...sizingOptions,
      tp_candidates: [1, 2, 4, 8, 16],
      pp_candidates: [1, 2, 4, 8],
      top_k: 64,
    }),
    [request, sizingOptions],
  );

  const recommendations = useMemo(() => {
    let rows = rawRecommendations;
    if (constrainToPod && podTargetMode === "exact") {
      const exactAsics = intValue(podSize, 8);
      rows = rows.filter((row) => row.asics === exactAsics);
    }
    return rows.slice(0, 3);
  }, [rawRecommendations, constrainToPod, podTargetMode, podSize]);

  const stageStats = useMemo(() => stageStatsFromResult(result), [result]);

  const splitPrefillRequest = useMemo(() => {
    const prompt = intValue(promptTokens, seqLen, 1);
    return {
      ...request,
      workload: {
        ...request.workload,
        phase: "prefill",
        seq_len: prompt,
        prefill_tokens: prompt,
      },
    };
  }, [request, promptTokens, seqLen]);

  const splitDecodeRequest = useMemo(() => {
    const decodeContextStart = intValue(decodeContextTokens, seqLen, 1);
    return {
      ...request,
      workload: {
        ...request.workload,
        phase: "decode",
        seq_len: decodeContextStart,
        decode_tokens: 1,
      },
    };
  }, [request, decodeContextTokens, seqLen]);

  const splitPrefillResult = useMemo(
    () => computeSizing(splitPrefillRequest, sizingOptions),
    [splitPrefillRequest, sizingOptions],
  );
  const splitDecodeResult = useMemo(
    () => computeSizing(splitDecodeRequest, sizingOptions),
    [splitDecodeRequest, sizingOptions],
  );
  const splitPrefillStageStats = useMemo(() => stageStatsFromResult(splitPrefillResult), [splitPrefillResult]);
  const splitDecodeStageStats = useMemo(() => stageStatsFromResult(splitDecodeResult), [splitDecodeResult]);

  const endToEnd = useMemo(() => {
    if (!enableE2E && viewMode !== "end_to_end") return null;
    const prompt = intValue(promptTokens, seqLen, 1);
    const gen = intValue(decodeTokens, 1, 1);
    const decodeContextStart = decodeUsesPrefillContext ? prompt : intValue(decodeContextTokens, prompt, 1);

    const prefillHardware = presetHardwareSpec(e2ePrefillHardware, request.hardware);
    const decodeHardware = presetHardwareSpec(e2eDecodeHardware, request.hardware);

    const prefillReq = {
      ...request,
      hardware: prefillHardware,
      workload: {
        ...request.workload,
        phase: "prefill",
        seq_len: prompt,
        prefill_tokens: prompt,
      },
    };
    const decodeStartReq = {
      ...request,
      hardware: decodeHardware,
      workload: {
        ...request.workload,
        phase: "decode",
        seq_len: decodeContextStart,
        decode_tokens: 1,
      },
    };
    const decodeEndReq = {
      ...decodeStartReq,
      workload: {
        ...decodeStartReq.workload,
        seq_len: decodeContextStart + Math.max(0, gen - 1),
        decode_tokens: 1,
      },
    };

    const prefill = computeSizing(prefillReq, sizingOptions);
    const decodeStart = computeSizing(decodeStartReq, sizingOptions);
    const decodeEnd = computeSizing(decodeEndReq, sizingOptions);
    const decodeAvgPerTokenMs = (decodeStart.time.end_to_end_ms + decodeEnd.time.end_to_end_ms) / 2;
    const decodeTotalMs = decodeAvgPerTokenMs * gen;

    const kvTransferBytes = estimateKvCacheBytes(activeModel, batch, decodeContextStart, kvBytesPerElement);
    const handoffMs = numberValue(handoffLatencyUs, 5, 0) * 1e-3
      + (kvTransferBytes / (numberValue(handoffBwGBs, 900, 0.1) * 1e9)) * 1e3;
    const totalMs = prefill.time.end_to_end_ms + decodeTotalMs + handoffMs;

    return {
      prompt,
      gen,
      decode_context_start: decodeContextStart,
      prefill_hw: e2ePrefillHardware,
      decode_hw: e2eDecodeHardware,
      prefill_ms: prefill.time.end_to_end_ms,
      decode_start_ms: decodeStart.time.end_to_end_ms,
      decode_end_ms: decodeEnd.time.end_to_end_ms,
      decode_avg_ms: decodeAvgPerTokenMs,
      decode_total_ms: decodeTotalMs,
      handoff_ms: handoffMs,
      kv_transfer_bytes: kvTransferBytes,
      total_ms: totalMs,
      e2e_tok_s: totalMs > 0 ? (gen / (totalMs / 1000)) : 0,
      prefill_bottleneck: prefill.bottleneck,
      decode_bottleneck: decodeStart.bottleneck,
      prefill_result: prefill,
      decode_result: decodeStart,
      prefill_stage_stats: stageStatsFromResult(prefill),
      decode_stage_stats: stageStatsFromResult(decodeStart),
    };
  }, [
    viewMode,
    enableE2E,
    promptTokens,
    decodeTokens,
    seqLen,
    decodeUsesPrefillContext,
    decodeContextTokens,
    request,
    sizingOptions,
    e2ePrefillHardware,
    e2eDecodeHardware,
    handoffBwGBs,
    handoffLatencyUs,
    activeModel,
    batch,
    kvBytesPerElement,
  ]);

  const maxBar = Math.max(result.time.compute_ms, result.time.memory_ms, result.time.network_ms, 1e-9);
  const splitPrefillMaxBar = Math.max(
    splitPrefillResult.time.compute_ms,
    splitPrefillResult.time.memory_ms,
    splitPrefillResult.time.network_ms,
    1e-9,
  );
  const splitDecodeMaxBar = Math.max(
    splitDecodeResult.time.compute_ms,
    splitDecodeResult.time.memory_ms,
    splitDecodeResult.time.network_ms,
    1e-9,
  );
  const isSinglePhaseView = viewMode === "prefill" || viewMode === "decode";
  const isSplitView = viewMode === "split";
  const isEndToEndView = viewMode === "end_to_end";
  const bottleneckColor = (
    result.bottleneck === "network" ? "#f97316"
      : result.bottleneck === "compute" ? "#ef4444"
        : "#60a5fa"
  );

  const panel = {
    background: "#0f172a",
    borderRadius: 6,
    padding: 8,
    marginBottom: 8,
    border: "1px solid #1e293b",
  };
  const label = {
    fontSize: 8,
    color: "#64748b",
    textTransform: "uppercase",
    letterSpacing: 0.6,
    marginBottom: 3,
    display: "block",
  };
  const input = {
    background: "#1e293b",
    color: "#e2e8f0",
    border: "1px solid #334155",
    borderRadius: 3,
    padding: "4px 6px",
    fontSize: 10,
    width: "100%",
    fontFamily: "monospace",
    outline: "none",
  };
  const grid2 = { display: "grid", gap: 4, gridTemplateColumns: "1fr 1fr" };
  const modeButton = (active) => ({
    background: active ? "#0ea5e9" : "#1e293b",
    color: active ? "#082f49" : "#cbd5e1",
    border: "1px solid #334155",
    borderRadius: 4,
    padding: "3px 6px",
    fontSize: 8,
    fontWeight: 600,
    cursor: "pointer",
    fontFamily: "monospace",
  });

  return (
    <div style={{ fontFamily: "monospace" }}>
      <div style={panel}>
        <div style={{ display: "flex", gap: 4, marginBottom: 6, flexWrap: "wrap" }}>
          <button type="button" style={modeButton(viewMode === "prefill")} onClick={() => setViewMode("prefill")}>Prefill Page</button>
          <button type="button" style={modeButton(viewMode === "decode")} onClick={() => setViewMode("decode")}>Decode Page</button>
          <button type="button" style={modeButton(viewMode === "split")} onClick={() => setViewMode("split")}>Prefill + Decode Split</button>
          <button type="button" style={modeButton(viewMode === "end_to_end")} onClick={() => setViewMode("end_to_end")}>End-to-End Page</button>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6, gap: 8 }}>
          <div style={{ fontSize: 9, color: "#e2e8f0", fontWeight: 600 }}>Simple Co-Design Sizing</div>
          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            <label style={{ fontSize: 8, color: "#94a3b8", display: "flex", gap: 4, alignItems: "center" }}>
              <input type="checkbox" checked={enableE2E} onChange={(e) => setEnableE2E(e.target.checked)} />
              E2E (prefill→decode)
            </label>
            <label style={{ fontSize: 8, color: "#94a3b8", display: "flex", gap: 4, alignItems: "center" }}>
              <input type="checkbox" checked={showAdvanced} onChange={(e) => setShowAdvanced(e.target.checked)} />
              Advanced
            </label>
          </div>
        </div>

        <div style={grid2}>
          <label>
            <span style={label}>Model</span>
            <select value={modelChoice} onChange={(e) => setModelChoice(e.target.value)} style={input}>
              {modelNames.map((name) => <option key={name} value={name}>{name}</option>)}
              <option value={CUSTOM_MODEL}>{CUSTOM_MODEL}</option>
            </select>
          </label>
          <label>
            <span style={label}>Hardware</span>
            <select value={hardwareChoice} onChange={(e) => setHardwareChoice(e.target.value)} style={input}>
              {hardwareNames.map((name) => <option key={name} value={name}>{name}</option>)}
            </select>
          </label>
          <label>
            <span style={label}>Phase</span>
            <select value={phase} onChange={(e) => setPhase(e.target.value)} style={input} disabled={!isSinglePhaseView}>
              <option value="decode">decode</option>
              <option value="prefill">prefill</option>
            </select>
          </label>
          <label>
            <span style={label}>Precision profile</span>
            <select value={configChoice} onChange={(e) => setConfigChoice(e.target.value)} style={input}>
              {configNames.map((name) => <option key={name} value={name}>{name}</option>)}
            </select>
          </label>
          <label>
            <span style={label}>Batch</span>
            <input type="number" min={1} value={batch} onChange={(e) => setBatch(intValue(e.target.value, batch))} style={input} />
          </label>
          <label>
            <span style={label}>{phase === "decode" ? "Decode context tokens" : "Seq Len"}</span>
            <input
              type="number"
              min={1}
              value={phase === "decode" ? decodeContextTokens : seqLen}
              onChange={(e) => (
                phase === "decode"
                  ? setDecodeContextTokens(intValue(e.target.value, decodeContextTokens))
                  : setSeqLen(intValue(e.target.value, seqLen))
              )}
              style={input}
            />
          </label>
          <label>
            <span style={label}>TP shards</span>
            <input type="number" min={1} value={tp} onChange={(e) => setTp(intValue(e.target.value, tp))} style={input} />
          </label>
          <label>
            <span style={label}>PP stages</span>
            <input type="number" min={1} value={pp} onChange={(e) => setPp(intValue(e.target.value, pp))} style={input} />
          </label>
          <label>
            <span style={label}>TP BW (GB/s)</span>
            <input type="number" min={0.1} value={tpLinkBw} onChange={(e) => setTpLinkBw(numberValue(e.target.value, tpLinkBw, 0.1))} style={input} />
          </label>
          <label>
            <span style={label}>PP BW (GB/s)</span>
            <input type="number" min={0.1} value={ppLinkBw} onChange={(e) => setPpLinkBw(numberValue(e.target.value, ppLinkBw, 0.1))} style={input} />
          </label>
        </div>

        <div style={{ ...grid2, marginTop: 6 }}>
          <label>
            <span style={label}>Pod size option</span>
            <select value={podSize} onChange={(e) => setPodSize(intValue(e.target.value, podSize))} style={input}>
              {[4, 8, 16, 32].map((n) => <option key={n} value={n}>{n === 8 ? "8 ASIC pod (focus)" : `${n} ASIC pod`}</option>)}
            </select>
          </label>
          <label style={{ display: "flex", alignItems: "flex-end" }}>
            <span style={{ ...label, width: "100%" }}>
              <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <input type="checkbox" checked={constrainToPod} onChange={(e) => setConstrainToPod(e.target.checked)} />
                Constrain TP×PP to pod size
              </span>
            </span>
          </label>
        </div>
        <div style={{ ...grid2, marginTop: 4 }}>
          <label>
            <span style={label}>Pod target mode</span>
            <select value={podTargetMode} onChange={(e) => setPodTargetMode(e.target.value)} style={input}>
              <option value="at_most">Use up to pod size</option>
              <option value="exact">Use exactly pod size</option>
            </select>
          </label>
          <div style={{ fontSize: 8, color: "#94a3b8", display: "flex", alignItems: "center" }}>
            {constrainToPod
              ? `Recommendations constrained to ${podTargetMode === "exact" ? "exactly" : "at most"} ${podSize} ASICs`
              : `Manual ASIC cap active: ${maxAsicsEffective}`}
          </div>
        </div>

        {showAdvanced && (
          <div style={{ ...panel, marginTop: 6, marginBottom: 0, padding: 6 }}>
            <div style={grid2}>
              <label>
                <span style={label}>Memory model</span>
                <select value={memoryMode} onChange={(e) => setMemoryMode(e.target.value)} style={input}>
                  <option value="single">single-tier BW</option>
                  <option value="two_tier">two-tier (on/off-chip)</option>
                </select>
              </label>
              <label>
                <span style={label}>Peak TFLOPS ({computeKey})</span>
                <input
                  type="number"
                  min={0.1}
                  value={hardwareFlops[computeKey] ?? 0}
                  onChange={(e) => setHardwareFlops((prev) => ({ ...prev, [computeKey]: numberValue(e.target.value, prev[computeKey] ?? 1, 0.1) }))}
                  style={input}
                />
              </label>

              {memoryMode === "single" && (
                <label>
                  <span style={label}>Memory BW (GB/s)</span>
                  <input type="number" min={0.1} value={memBw} onChange={(e) => setMemBw(numberValue(e.target.value, memBw, 0.1))} style={input} />
                </label>
              )}

              {memoryMode === "two_tier" && (
                <>
                  <label>
                    <span style={label}>On-chip BW (GB/s)</span>
                    <input type="number" min={0.1} value={onchipBw} onChange={(e) => setOnchipBw(numberValue(e.target.value, onchipBw, 0.1))} style={input} />
                  </label>
                  <label>
                    <span style={label}>Off-chip BW (GB/s)</span>
                    <input type="number" min={0.1} value={offchipBw} onChange={(e) => setOffchipBw(numberValue(e.target.value, offchipBw, 0.1))} style={input} />
                  </label>
                  <label>
                    <span style={label}>On-chip hit rate</span>
                    <input type="number" min={0} max={1} step={0.01} value={onchipHitRate} onChange={(e) => setOnchipHitRate(numberValue(e.target.value, onchipHitRate, 0))} style={input} />
                  </label>
                </>
              )}

              <label>
                <span style={label}>Manual max ASICs</span>
                <input type="number" min={1} value={maxAsics} onChange={(e) => setMaxAsics(intValue(e.target.value, maxAsics))} style={input} />
              </label>
              <label>
                <span style={label}>Overlap (0-1)</span>
                <input type="number" min={0} max={1} step={0.05} value={overlap} onChange={(e) => setOverlap(numberValue(e.target.value, overlap, 0))} style={input} />
              </label>
              <label>
                <span style={label}>TP latency (us)</span>
                <input type="number" min={0} value={tpLinkLatUs} onChange={(e) => setTpLinkLatUs(numberValue(e.target.value, tpLinkLatUs, 0))} style={input} />
              </label>
              <label>
                <span style={label}>PP latency (us)</span>
                <input type="number" min={0} value={ppLinkLatUs} onChange={(e) => setPpLinkLatUs(numberValue(e.target.value, ppLinkLatUs, 0))} style={input} />
              </label>
            </div>

            {modelChoice === CUSTOM_MODEL && (
              <div style={{ ...panel, marginTop: 6, marginBottom: 0, padding: 6 }}>
                <div style={grid2}>
                  {["L", "H", "nh", "nkv", "dh", "dff", "V"].map((key) => (
                    <label key={key}>
                      <span style={label}>{key}</span>
                      <input
                        type="number"
                        min={1}
                        value={customModel[key]}
                        onChange={(e) => setCustomModel((prev) => ({ ...prev, [key]: intValue(e.target.value, prev[key]) }))}
                        style={input}
                      />
                    </label>
                  ))}
                </div>
                <label style={{ fontSize: 9, color: "#94a3b8", display: "flex", gap: 4, alignItems: "center", marginTop: 6 }}>
                  <input type="checkbox" checked={Boolean(customModel.gate)} onChange={(e) => setCustomModel((prev) => ({ ...prev, gate: e.target.checked }))} />
                  SwiGLU gate
                </label>
              </div>
            )}
          </div>
        )}
      </div>

      {isSinglePhaseView && (
        <>
          <div style={panel}>
            <div style={{ fontSize: 9, color: "#475569", textTransform: "uppercase", letterSpacing: 0.7, marginBottom: 6 }}>
              Pod Visualizer (cool node/fabric view)
            </div>
            <PodVisualizer podSize={podSize} tp={request.parallel.tp} pp={request.parallel.pp} bottleneck={result.bottleneck} />
          </div>

          <div style={panel}>
            <div style={{ fontSize: 9, color: "#475569", textTransform: "uppercase", letterSpacing: 0.7, marginBottom: 6 }}>
              Sharded Infra View
            </div>
            <StageFlow stageStats={stageStats} tp={request.parallel.tp} />
          </div>

          <div style={panel}>
            <div style={{ fontSize: 10, color: "#e2e8f0", marginBottom: 6 }}>
              Dominant bottleneck: <span style={{ color: bottleneckColor, fontWeight: 600 }}>{result.bottleneck.toUpperCase()}</span>
            </div>
            <TimeBar label="Compute roofline time" value={result.time.compute_ms} maxValue={maxBar} color="#ef4444" />
            <TimeBar label="Memory roofline time" value={result.time.memory_ms} maxValue={maxBar} color="#3b82f6" />
            <TimeBar label="Network movement time" value={result.time.network_ms} maxValue={maxBar} color="#f59e0b" />
            <div style={{ fontSize: 9, color: "#94a3b8", marginTop: 4 }}>
              End-to-end: <span style={{ color: "#22c55e" }}>{fmtMs(result.time.end_to_end_ms)}</span> · Throughput: {result.time.tokens_per_s.toFixed(1)} tok/s
            </div>
            <div style={{ fontSize: 8, color: "#64748b", marginTop: 2 }}>
              Active context: {request.workload.seq_len} tokens ({request.workload.phase})
            </div>
          </div>

          <div style={panel}>
            <div style={{ fontSize: 9, color: "#475569", textTransform: "uppercase", letterSpacing: 0.7, marginBottom: 4 }}>
              Data Movement + Guidance
            </div>
            <div style={{ fontSize: 9, color: "#94a3b8", lineHeight: 1.7 }}>
              <div>TP all-reduces: {result.collective.tp_allreduce_count} · {fmtBytes(result.collective.tp_allreduce_bytes)}</div>
              <div>PP sends: {result.collective.pp_send_count} · {fmtBytes(result.collective.pp_send_bytes)}</div>
              {result.required_to_debottleneck.network_bw_gbs && (
                <div style={{ color: "#fbbf24" }}>Need ~{result.required_to_debottleneck.network_bw_gbs.toFixed(1)} GB/s link to debottleneck network</div>
              )}
              {result.required_to_debottleneck.mem_bw_gbs && (
                <div style={{ color: "#fbbf24" }}>Need ~{result.required_to_debottleneck.mem_bw_gbs.toFixed(1)} GB/s effective memory BW to debottleneck memory</div>
              )}
              {result.required_to_debottleneck.offchip_bw_gbs && (
                <div style={{ color: "#fbbf24" }}>Need ~{result.required_to_debottleneck.offchip_bw_gbs.toFixed(1)} GB/s off-chip BW (with current hit rate)</div>
              )}
              {result.required_to_debottleneck.peak_tflops && (
                <div style={{ color: "#fbbf24" }}>Need ~{result.required_to_debottleneck.peak_tflops.toFixed(1)} TFLOPS to debottleneck compute</div>
              )}
            </div>
          </div>

          <div style={panel}>
            <div style={{ fontSize: 9, color: "#475569", textTransform: "uppercase", letterSpacing: 0.7, marginBottom: 4 }}>
              Best TP/PP Picks (within current pod/ASIC constraint)
            </div>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 8 }}>
              <thead>
                <tr style={{ color: "#64748b" }}>
                  <th align="left">tp</th>
                  <th align="left">pp</th>
                  <th align="left">ASICs</th>
                  <th align="right">Latency</th>
                </tr>
              </thead>
              <tbody>
                {recommendations.length === 0 ? (
                  <tr style={{ color: "#94a3b8" }}>
                    <td colSpan={4}>No valid TP/PP split under current pod target mode.</td>
                  </tr>
                ) : (
                  recommendations.map((row, idx) => (
                    <tr key={`${row.tp}-${row.pp}-${idx}`} style={{ color: idx === 0 ? "#22c55e" : "#94a3b8" }}>
                      <td>{row.tp}</td>
                      <td>{row.pp}</td>
                      <td>{row.asics}</td>
                      <td align="right">{fmtMs(row.latency_ms)}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
            {recommendations[0] && (
              <div style={{ marginTop: 5, fontSize: 8, color: "#fbbf24" }}>{recommendations[0].note}</div>
            )}
          </div>

          <div style={panel}>
            <label style={{ fontSize: 9, color: "#94a3b8", display: "flex", gap: 4, alignItems: "center", marginBottom: 4 }}>
              <input type="checkbox" checked={showDualPhase} onChange={(e) => setShowDualPhase(e.target.checked)} />
              Compare prefill/decode (single phase mode)
            </label>
            <div style={{ fontSize: 8, color: "#94a3b8", lineHeight: 1.6 }}>
              <div>{phase}: {fmtMs(result.time.end_to_end_ms)} · {result.bottleneck}</div>
              {showDualPhase && <div>{oppositePhase}: {fmtMs(oppositeResult.time.end_to_end_ms)} · {oppositeResult.bottleneck}</div>}
            </div>

            <label style={{ fontSize: 9, color: "#94a3b8", display: "flex", gap: 4, alignItems: "center", marginTop: 6, marginBottom: 4 }}>
              <input type="checkbox" checked={showLayerDetails} onChange={(e) => setShowLayerDetails(e.target.checked)} />
              Show per-layer movement
            </label>
            {showLayerDetails && (
              <div style={{ maxHeight: 180, overflowY: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 8 }}>
                  <thead>
                    <tr style={{ color: "#64748b" }}>
                      <th align="left">L</th>
                      <th align="left">Stg</th>
                      <th align="right">W</th>
                      <th align="right">TP</th>
                      <th align="right">PP</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.layer_io.map((row) => (
                      <tr key={`${row.layer}-${row.stage}`} style={{ color: "#94a3b8" }}>
                        <td>{row.layer}</td>
                        <td>{row.stage}</td>
                        <td align="right">{fmtBytes(row.weight_bytes)}</td>
                        <td align="right">{fmtBytes(row.tp_sync_bytes)}</td>
                        <td align="right">{fmtBytes(row.pp_boundary_send_bytes)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </>
      )}

      {isSplitView && (
        <>
          <div style={panel}>
            <div style={{ fontSize: 9, color: "#475569", textTransform: "uppercase", letterSpacing: 0.7, marginBottom: 6 }}>
              Split View (Prefill left · Decode right)
            </div>
            <div style={{ ...grid2, marginBottom: 8 }}>
              <label>
                <span style={label}>Prefill context tokens</span>
                <input type="number" min={1} value={promptTokens} onChange={(e) => setPromptTokens(intValue(e.target.value, promptTokens))} style={input} />
              </label>
              <label>
                <span style={label}>Decode context tokens</span>
                <input type="number" min={1} value={decodeContextTokens} onChange={(e) => setDecodeContextTokens(intValue(e.target.value, decodeContextTokens))} style={input} />
              </label>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
              <div style={{ border: "1px solid #334155", borderRadius: 6, padding: 8, background: "#0b1220" }}>
                <div style={{ fontSize: 9, color: "#22c55e", marginBottom: 4 }}>Prefill</div>
                <TimeBar label="Compute" value={splitPrefillResult.time.compute_ms} maxValue={splitPrefillMaxBar} color="#ef4444" />
                <TimeBar label="Memory" value={splitPrefillResult.time.memory_ms} maxValue={splitPrefillMaxBar} color="#3b82f6" />
                <TimeBar label="Network" value={splitPrefillResult.time.network_ms} maxValue={splitPrefillMaxBar} color="#f59e0b" />
                <div style={{ fontSize: 8, color: "#94a3b8", marginTop: 4 }}>
                  {fmtMs(splitPrefillResult.time.end_to_end_ms)} · {splitPrefillResult.bottleneck}
                </div>
                <PodVisualizer podSize={podSize} tp={request.parallel.tp} pp={request.parallel.pp} bottleneck={splitPrefillResult.bottleneck} />
              </div>
              <div style={{ border: "1px solid #334155", borderRadius: 6, padding: 8, background: "#0b1220" }}>
                <div style={{ fontSize: 9, color: "#f59e0b", marginBottom: 4 }}>Decode</div>
                <TimeBar label="Compute" value={splitDecodeResult.time.compute_ms} maxValue={splitDecodeMaxBar} color="#ef4444" />
                <TimeBar label="Memory" value={splitDecodeResult.time.memory_ms} maxValue={splitDecodeMaxBar} color="#3b82f6" />
                <TimeBar label="Network" value={splitDecodeResult.time.network_ms} maxValue={splitDecodeMaxBar} color="#f59e0b" />
                <div style={{ fontSize: 8, color: "#94a3b8", marginTop: 4 }}>
                  {fmtMs(splitDecodeResult.time.end_to_end_ms)} · {splitDecodeResult.bottleneck}
                </div>
                <PodVisualizer podSize={podSize} tp={request.parallel.tp} pp={request.parallel.pp} bottleneck={splitDecodeResult.bottleneck} />
              </div>
            </div>
          </div>

          <div style={panel}>
            <div style={{ fontSize: 9, color: "#475569", textTransform: "uppercase", letterSpacing: 0.7, marginBottom: 6 }}>
              Multi-node Data Flow (bottom)
            </div>
            <div style={{ fontSize: 8, color: "#94a3b8", marginBottom: 4 }}>Prefill stage flow</div>
            <StageFlow stageStats={splitPrefillStageStats} tp={request.parallel.tp} />
            <div style={{ fontSize: 8, color: "#94a3b8", marginTop: 8, marginBottom: 4 }}>Decode stage flow</div>
            <StageFlow stageStats={splitDecodeStageStats} tp={request.parallel.tp} />
          </div>
        </>
      )}

      {isEndToEndView && endToEnd && (
        <>
          <div style={panel}>
            <div style={{ fontSize: 9, color: "#475569", textTransform: "uppercase", letterSpacing: 0.7, marginBottom: 6 }}>
              End-to-End Phase Mapping (independent prefill/decode hardware)
            </div>
            <div style={grid2}>
              <label>
                <span style={label}>Prefill ASIC/GPU</span>
                <select value={e2ePrefillHardware} onChange={(e) => setE2ePrefillHardware(e.target.value)} style={input}>
                  {hardwareNames.map((name) => <option key={name} value={name}>{name}</option>)}
                </select>
              </label>
              <label>
                <span style={label}>Decode ASIC/GPU</span>
                <select value={e2eDecodeHardware} onChange={(e) => setE2eDecodeHardware(e.target.value)} style={input}>
                  {hardwareNames.map((name) => <option key={name} value={name}>{name}</option>)}
                </select>
              </label>
              <label>
                <span style={label}>Prompt tokens (prefill)</span>
                <input type="number" min={1} value={promptTokens} onChange={(e) => setPromptTokens(intValue(e.target.value, promptTokens))} style={input} />
              </label>
              <label>
                <span style={label}>Generated decode tokens</span>
                <input type="number" min={1} value={decodeTokens} onChange={(e) => setDecodeTokens(intValue(e.target.value, decodeTokens))} style={input} />
              </label>
              <label style={{ display: "flex", alignItems: "flex-end" }}>
                <span style={{ ...label, width: "100%" }}>
                  <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <input
                      type="checkbox"
                      checked={decodeUsesPrefillContext}
                      onChange={(e) => setDecodeUsesPrefillContext(e.target.checked)}
                    />
                    Decode uses prefill context
                  </span>
                </span>
              </label>
              {!decodeUsesPrefillContext ? (
                <label>
                  <span style={label}>Prior-phase context tokens</span>
                  <input
                    type="number"
                    min={1}
                    value={decodeContextTokens}
                    onChange={(e) => setDecodeContextTokens(intValue(e.target.value, decodeContextTokens))}
                    style={input}
                  />
                </label>
              ) : (
                <div style={{ fontSize: 8, color: "#94a3b8", display: "flex", alignItems: "center" }}>
                  Decode starts at prompt context length
                </div>
              )}
              <label>
                <span style={label}>Interconnect BW prefill→decode (GB/s)</span>
                <input type="number" min={0.1} value={handoffBwGBs} onChange={(e) => setHandoffBwGBs(numberValue(e.target.value, handoffBwGBs, 0.1))} style={input} />
              </label>
              <label>
                <span style={label}>Interconnect latency prefill→decode (us)</span>
                <input type="number" min={0} value={handoffLatencyUs} onChange={(e) => setHandoffLatencyUs(numberValue(e.target.value, handoffLatencyUs, 0))} style={input} />
              </label>
            </div>
          </div>

          <div style={panel}>
            <div style={{ fontSize: 10, color: "#e2e8f0", marginBottom: 6 }}>
              End-to-end latency: <span style={{ color: "#22c55e", fontWeight: 600 }}>{fmtMs(endToEnd.total_ms)}</span> · E2E tok/s {endToEnd.e2e_tok_s.toFixed(1)}
            </div>
            <TimeBar label={`Prefill (${endToEnd.prefill_hw})`} value={endToEnd.prefill_ms} maxValue={Math.max(endToEnd.prefill_ms, endToEnd.decode_total_ms, endToEnd.handoff_ms, 1e-6)} color="#22c55e" />
            <TimeBar label="Interconnect handoff" value={endToEnd.handoff_ms} maxValue={Math.max(endToEnd.prefill_ms, endToEnd.decode_total_ms, endToEnd.handoff_ms, 1e-6)} color="#38bdf8" />
            <TimeBar label={`Decode total (${endToEnd.decode_hw})`} value={endToEnd.decode_total_ms} maxValue={Math.max(endToEnd.prefill_ms, endToEnd.decode_total_ms, endToEnd.handoff_ms, 1e-6)} color="#f59e0b" />
            <div style={{ fontSize: 8, color: "#94a3b8", marginTop: 4, lineHeight: 1.6 }}>
              <div>Decode context start: {endToEnd.decode_context_start} tokens</div>
              <div>KV/context transfer: {fmtBytes(endToEnd.kv_transfer_bytes)}</div>
              <div>Prefill bottleneck: {endToEnd.prefill_bottleneck} · Decode bottleneck: {endToEnd.decode_bottleneck}</div>
            </div>
            <div style={{ marginTop: 6, fontSize: 8, color: "#94a3b8", lineHeight: 1.7 }}>
              <div>Prefill TP+PP network time: {fmtMs(endToEnd.prefill_result.metadata.network_breakdown_ms.tp_time_ms + endToEnd.prefill_result.metadata.network_breakdown_ms.pp_time_ms)}</div>
              <div>Decode TP+PP network time: {fmtMs(endToEnd.decode_result.metadata.network_breakdown_ms.tp_time_ms + endToEnd.decode_result.metadata.network_breakdown_ms.pp_time_ms)}</div>
              <div>Prefill inter-node latency (TP+PP): {fmtMs((endToEnd.prefill_result.metadata.network_breakdown_ms.tp_latency_ms || 0) + (endToEnd.prefill_result.metadata.network_breakdown_ms.pp_latency_ms || 0))}</div>
              <div>Decode inter-node latency (TP+PP): {fmtMs((endToEnd.decode_result.metadata.network_breakdown_ms.tp_latency_ms || 0) + (endToEnd.decode_result.metadata.network_breakdown_ms.pp_latency_ms || 0))}</div>
            </div>
          </div>

          <div style={panel}>
            <div style={{ fontSize: 9, color: "#475569", textTransform: "uppercase", letterSpacing: 0.7, marginBottom: 6 }}>
              Pod / Node Mapping by Phase
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
              <div>
                <div style={{ fontSize: 8, color: "#94a3b8", marginBottom: 4 }}>Prefill hardware: {endToEnd.prefill_hw}</div>
                <PodVisualizer podSize={podSize} tp={request.parallel.tp} pp={request.parallel.pp} bottleneck={endToEnd.prefill_bottleneck} />
              </div>
              <div>
                <div style={{ fontSize: 8, color: "#94a3b8", marginBottom: 4 }}>Decode hardware: {endToEnd.decode_hw}</div>
                <PodVisualizer podSize={podSize} tp={request.parallel.tp} pp={request.parallel.pp} bottleneck={endToEnd.decode_bottleneck} />
              </div>
            </div>
          </div>

          <div style={panel}>
            <div style={{ fontSize: 9, color: "#475569", textTransform: "uppercase", letterSpacing: 0.7, marginBottom: 6 }}>
              Multi-node Data Flow (bottom)
            </div>
            <div style={{ fontSize: 8, color: "#94a3b8", marginBottom: 4 }}>Prefill nodes</div>
            <StageFlow stageStats={endToEnd.prefill_stage_stats} tp={request.parallel.tp} />
            <div style={{ fontSize: 8, color: "#38bdf8", margin: "8px 0" }}>
              Prefill→Decode interconnect: {fmtBytes(endToEnd.kv_transfer_bytes)} over {numberValue(handoffBwGBs, 900, 0.1).toFixed(1)} GB/s + {numberValue(handoffLatencyUs, 5, 0).toFixed(2)} us
            </div>
            <div style={{ fontSize: 8, color: "#94a3b8", marginBottom: 4 }}>Decode nodes</div>
            <StageFlow stageStats={endToEnd.decode_stage_stats} tp={request.parallel.tp} />
          </div>
        </>
      )}
    </div>
  );
}
