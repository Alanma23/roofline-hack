"""Pydantic models for the roofline API."""

from pydantic import BaseModel
from typing import Optional, List, Dict, Union


class GEMMSpec(BaseModel):
    """GEMM kernel specification from user."""
    M: int = 4096
    N: int = 4096
    K: int = 4096
    precision: str = "FP16"
    tile_m: int = 128
    tile_n: int = 128
    tile_k: int = 32
    use_efficiency: bool = True  # Apply empirical efficiency factors (realistic vs ideal)


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
    max_asics: int = 128  # Support up to 128 GPUs


class SizingNetworkSpec(BaseModel):
    """Network controls for TP/PP communication."""
    tp_link_bw_gbs: float = 900.0
    tp_link_latency_us: float = 3.0
    pp_link_bw_gbs: float = 900.0
    pp_link_latency_us: float = 3.0
    overlap_fraction: float = 0.0


class MultiNodeNetworkSpec(BaseModel):
    """Multi-node network configuration with intra/inter-node topology."""
    # Intra-node (NVLink/ICI within single node)
    intra_node_bw_gbs: float = 900.0       # NVLink bandwidth per link
    intra_node_latency_us: float = 3.0     # NVLink latency
    gpus_per_node: int = 8                 # GPUs in single node

    # Inter-node (Ethernet/InfiniBand/DCN between nodes)
    inter_node_bw_gbs: float = 400.0       # Total inter-node bandwidth
    inter_node_latency_us: float = 10.0    # Inter-node latency

    # Overlap
    overlap_fraction: float = 0.0          # Compute/comm overlap (0-1)


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
    network: Union[SizingNetworkSpec, MultiNodeNetworkSpec]  # Accept both
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
    network: Union[SizingNetworkSpec, MultiNodeNetworkSpec]  # Accept both
    tp_candidates: Optional[List[int]] = None
    pp_candidates: Optional[List[int]] = None


class SizingSweepResponse(BaseModel):
    """Sweep response containing base and candidate configurations."""
    base: SizingResponse
    candidates: List[SizingRecommendation]


# ═══════════════════════════════════════════════
#  RUN TRACKING SCHEMAS
# ═══════════════════════════════════════════════

class RunMetadata(BaseModel):
    """Metadata for a saved run."""
    run_id: str                          # UUID
    timestamp: str                       # ISO 8601
    name: str                            # User label
    hardware_key: str                    # "b10", "b200", etc.
    workload_type: str                   # "transformer", "moe", "custom"
    config: Dict                         # Full config snapshot
    results: Dict                        # Performance results
    tags: List[str] = []                 # ["prefill", "8B", "FP8"]
    notes: str = ""


class RunListItem(BaseModel):
    """Summary item for run list."""
    run_id: str
    timestamp: str
    name: str
    hardware_key: str
    workload_type: str
    tags: List[str]
    preview: Dict  # {tflops, latency_ms, bottleneck}


class SaveRunRequest(BaseModel):
    """Request to save a run."""
    metadata: RunMetadata


class CompareRunsRequest(BaseModel):
    """Request to compare multiple runs."""
    run_ids: List[str]


class ComparisonMetric(BaseModel):
    """Single metric comparison across runs."""
    metric: str
    values: Dict[str, float]  # {run_id: value}
    deltas: Dict[str, float]  # {run_id: % delta from baseline}


class CompareRunsResponse(BaseModel):
    """Response from run comparison."""
    runs: List[RunMetadata]
    comparison_table: List[ComparisonMetric]


class ExportRunsRequest(BaseModel):
    """Request to export runs as JSON."""
    run_ids: List[str]


class ExportRunsResponse(BaseModel):
    """Response with exported run data."""
    runs: List[RunMetadata]
    export_format: str = "json_v1"


class ImportRunsRequest(BaseModel):
    """Request to import runs from JSON."""
    runs: List[RunMetadata]


class ImportRunsResponse(BaseModel):
    """Response from import."""
    imported: int
    errors: List[str]


# ═══════════════════════════════════════════════
#  MOE (MIXTURE OF EXPERTS) SCHEMAS
# ═══════════════════════════════════════════════

class MoEConfig(BaseModel):
    """MoE architecture configuration."""
    num_experts: int = 8
    experts_per_token: int = 2  # Top-K
    expert_ffn_dim: int = 14336
    capacity_factor: float = 1.25
    load_balance_loss_weight: float = 0.01
    expert_parallel: int = 1  # EP degree


class MoEWorkloadSpec(BaseModel):
    """MoE workload specification."""
    model: WorkloadModelSpec
    moe: MoEConfig
    precision: WorkloadPrecisionSpec
    phase: str = "decode"
    batch: int = 1
    seq_len: int = 4096
    network_bw_gbs: float = 900.0  # NVLink/ICI bandwidth
    network_latency_us: float = 3.0
    load_imbalance: float = 0.15  # Expected load imbalance


class AdvancedLayerSpec(BaseModel):
    """Layer specification for advanced roofline analysis."""
    name: str
    M: int  # Output rows
    N: int  # Output columns
    K: int  # Inner dimension
    input_precision_bits: int = 8
    accumulator_precision_bits: int = 32


class AdvancedRooflineResult(BaseModel):
    """Advanced roofline analysis result with utilization, memory hierarchy, and mixed precision."""
    # Performance
    actual_tops: float
    latency_ms: float
    latency_us: float

    # Roofline components
    peak_compute_tops: float
    effective_compute_tops: float
    mem_bound_tops: float
    arithmetic_intensity: float

    # Utilization (LEVEL 2)
    utilization: float
    padded_m: int
    padded_n: int
    useful_ops: int
    total_hw_ops: int

    # Memory (LEVEL 1 & 3)
    mem_tier: str  # "SRAM (On-Chip)" or "DRAM (Off-Chip)"
    total_data_bytes: float
    input_bytes: float
    output_bytes: float

    # Bottleneck
    bottleneck: str  # "COMPUTE", "UTILIZATION", or "MEMORY"

    # Hardware info
    hardware: str
    input_precision_bits: int
    accumulator_precision_bits: int


class BatchSweepRequest(BaseModel):
    """Request to sweep batch sizes for utilization analysis."""
    layer: AdvancedLayerSpec
    batch_sizes: List[int]


class BatchSweepResult(BaseModel):
    """Result from batch size sweep."""
    batch_size: int
    result: AdvancedRooflineResult


# ═══════════════════════════════════════════════
#  OPTIMIZER SCHEMAS
# ═══════════════════════════════════════════════

class OptimizationTargetInput(BaseModel):
    """Optimization target constraints."""
    latency_ms: Optional[float] = None
    throughput_tok_s: Optional[float] = None
    max_nodes: int = 8
    max_asics: int = 64


class RankedConfigResult(BaseModel):
    """A ranked hardware+parallelism configuration from optimizer search."""
    tp: int
    pp: int
    ep: int
    nodes: int
    precision: str
    predicted_latency_ms: float
    predicted_throughput_tok_s: float
    bottleneck: str
    pareto_optimal: bool
    score: float
    ep_uses_internode: bool = False
    network_time_ms: float = 0.0
    ep_time_ms: float = 0.0
    note: str = ""


class OptimizerSearchRequest(BaseModel):
    """Request to search for optimal hardware configurations."""
    workload: WorkloadSpec
    hardware: SizingHardwareSpec
    target: OptimizationTargetInput
    network: Union[SizingNetworkSpec, MultiNodeNetworkSpec] = None
    tp_candidates: Optional[List[int]] = None
    pp_candidates: Optional[List[int]] = None
    ep_candidates: Optional[List[int]] = None
    precision_candidates: Optional[List[str]] = None


class OptimizerSearchResponse(BaseModel):
    """Response from optimizer search."""
    configs: List[RankedConfigResult]
    pareto_count: int
    total_searched: int


class SuggestNextRequest(BaseModel):
    """Request for next-step suggestion based on run history."""
    run_history: List[Dict]


class SuggestNextResult(BaseModel):
    """Next optimization step recommendation."""
    action: str
    reason: str
    suggested_change: str


class MoEAnalysisResult(BaseModel):
    """MoE layer performance analysis result."""
    # Router
    router_flops: float
    router_time_us: float
    router_tflops: float
    routing_overhead_pct: float

    # Experts
    expert_flops: float
    expert_time_us: float
    expert_tflops: float

    # Communication
    all_to_all_bytes: float
    all_to_all_time_us: float
    communication_overhead_pct: float

    # Totals
    total_flops: float
    total_time_us: float
    total_time_ms: float

    # Load balancing
    load_imbalance: float
    capacity_factor: float
    avg_tokens_per_expert: float
    expert_utilization: List[float]

    # Sparsity
    sparsity: float
    active_experts_pct: float

    # Analysis
    bottleneck: str
    recommendations: List[str]

    # Debug info (optional)
    debug: Optional[Dict] = None
