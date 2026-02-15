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
