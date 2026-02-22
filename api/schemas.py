"""Pydantic models for the roofline API."""

from pydantic import BaseModel
from typing import Optional, List, Dict


class GEMMSpec(BaseModel):
    """GEMM kernel specification from user."""
    M: int = 4096
    N: int = 4096
    K: int = 4096
    precision: str = "FP16"
    tile_m: int = 128
    tile_n: int = 128
    tile_k: int = 32


class HardwareSpecInput(BaseModel):
    """Custom hardware specification."""
    name: str
    bandwidth_gb_s: float
    flops_by_precision: Dict[str, float]
    memory_gb: Optional[float] = None


class RooflinePoint(BaseModel):
    """A single point on the roofline chart."""
    ai: float
    tflops: float
    time_us: float
    label: str
    source: str  # "simulated" or "measured"
    precision: str
    shape: str
    bottleneck: Optional[str] = None
    bandwidth_gb_s: Optional[float] = None


class SimplifiedBenchmarkPoint(BaseModel):
    """Simplified format - auto-calculates AI, TFLOPS from shape + time."""
    shape: str                          # "4096x4096" or "4096x4096x4096"
    precision: str                      # "FP16", "FP8_E4M3", etc.
    time_us: float                      # measured execution time
    tflops: Optional[float] = None      # computed if omitted
    ai: Optional[float] = None          # computed if omitted
    bandwidth_gb_s: Optional[float] = None
    label: Optional[str] = None
    source: str = "measured"


class FlexibleImportRequest(BaseModel):
    """Accepts full RooflinePoint array or simplified format."""
    points: Optional[List[RooflinePoint]] = None
    simplified: Optional[List[SimplifiedBenchmarkPoint]] = None


class ImportBenchmarkResponse(BaseModel):
    """Response from import endpoint."""
    accepted: int
    rejected: int
    points: List[RooflinePoint]
    errors: List[str]


class RooflineLine(BaseModel):
    """A roofline ceiling line."""
    precision: str
    peak_tflops: float
    critical_ai: float


class TilingResult(BaseModel):
    """Tiling analysis result."""
    tile_m: int
    tile_n: int
    tile_k: int
    shared_mem_bytes: int
    tiles_total: int
    waves: float
    wave_efficiency: float
    sm_occupancy_pct: float
    l2_hit_rate_estimate: float
    efficiency_score: float


class RecommendationResult(BaseModel):
    """Auto-quantizer recommendation."""
    precision: str
    method: str
    reason: str
    predicted_speedup: float
    memory_bound: bool
    memory_savings_pct: float


class AnalyzeResponse(BaseModel):
    """Response from /api/analyze endpoint."""
    hardware: str
    simulated: List[RooflinePoint]
    measured: List[RooflinePoint]
    roofline_lines: List[RooflineLine]
    recommendation: RecommendationResult
    tiling: Optional[TilingResult] = None
    nvml: Optional[dict] = None


class SweepResponse(BaseModel):
    """Response from /api/sweep endpoint."""
    hardware: str
    points: List[RooflinePoint]
    roofline_lines: List[RooflineLine]


class NVMLStatusResponse(BaseModel):
    """Live GPU status."""
    device_name: str = ""
    gpu_clock_mhz: int = 0
    mem_clock_mhz: int = 0
    power_draw_w: float = 0.0
    power_limit_w: float = 0.0
    temperature_c: int = 0
    mem_used_mb: int = 0
    mem_total_mb: int = 0
    gpu_utilization_pct: int = 0
    compute_capability: List[int] = [0, 0]


class HardwareListItem(BaseModel):
    """Hardware registry entry."""
    key: str
    name: str
    bandwidth_gb_s: float
    precisions: List[str]


class WorkloadModelSpec(BaseModel):
    """Model dimensions for inference sizing."""
    L: int
    H: int
    nh: int
    nkv: int
    dh: int
    dff: int
    V: int
    gate: bool = True


class WorkloadPrecisionSpec(BaseModel):
    """Precision assignment for workload sizing."""
    w: str
    a: str
    kv: str
    computeAs: str


class WorkloadSpec(BaseModel):
    """Workload descriptor (prefill/decode)."""
    phase: str = "decode"
    batch: int = 1
    seq_len: int = 4096
    prefill_tokens: Optional[int] = None
    decode_tokens: Optional[int] = None
    model: WorkloadModelSpec
    precision: WorkloadPrecisionSpec


class SizingHardwareSpec(BaseModel):
    """Hardware descriptor for sizing model."""
    name: str
    peak_tflops: Dict[str, float]
    mem_bw_gbs: float
    memory_model: Optional[Dict[str, float | str]] = None


class SizingParallelSpec(BaseModel):
    """Parallelization controls."""
    tp: int = 1
    pp: int = 1
    max_asics: int = 16


class SizingNetworkSpec(BaseModel):
    """Network controls for TP/PP communication."""
    tp_link_bw_gbs: float = 900.0
    tp_link_latency_us: float = 3.0
    pp_link_bw_gbs: float = 900.0
    pp_link_latency_us: float = 3.0
    overlap_fraction: float = 0.0


class SizingRecommendation(BaseModel):
    """Recommended TP/PP config."""
    tp: int
    pp: int
    asics: int
    latency_ms: float
    bottleneck: str
    note: str


class LayerIOItem(BaseModel):
    """Per-layer I/O + communication accounting."""
    layer: int
    stage: int
    input_bytes: float
    output_bytes: float
    weight_bytes: float
    tp_sync_bytes: float
    pp_boundary_send_bytes: float


class SizingRequest(BaseModel):
    """Workload sizing request."""
    workload: WorkloadSpec
    hardware: SizingHardwareSpec
    parallel: SizingParallelSpec
    network: SizingNetworkSpec
    target_latency_ms: Optional[float] = None


class SizingResponse(BaseModel):
    """Workload sizing response."""
    totals: Dict[str, float]
    collective: Dict[str, float]
    time: Dict[str, float]
    bottleneck: str
    layer_io: List[LayerIOItem]
    recommendations: List[SizingRecommendation]
    required_to_debottleneck: Optional[Dict[str, float]] = None


class SizingSweepRequest(BaseModel):
    """Sweep request over TP/PP candidates."""
    workload: WorkloadSpec
    hardware: SizingHardwareSpec
    parallel: SizingParallelSpec
    network: SizingNetworkSpec
    tp_candidates: Optional[List[int]] = None
    pp_candidates: Optional[List[int]] = None


class SizingSweepResponse(BaseModel):
    """Sweep response containing base and candidate configurations."""
    base: SizingResponse
    candidates: List[SizingRecommendation]
