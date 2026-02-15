"""
FastAPI backend for GEMM Roofline Analysis.

Endpoints:
  POST /api/analyze     — Analyze a single GEMM (simulated + measured + recommendation)
  POST /api/sweep       — Sweep across shapes/precisions
  GET  /api/nvml/status — Live GPU status via NVML
  GET  /api/hardware    — List all hardware specs
  POST /api/hardware    — Register custom hardware/ASIC
  POST /api/tiling      — Tiling analysis for a GEMM shape
  POST /api/recommend   — Auto-quantizer recommendation

Run: uvicorn api.server:app --reload --port 8000
"""

import sys
import asyncio
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.roofline.calculator_shell import RooflineCalculator, bytes_per_element
from src.roofline.hardware_registry import (
    get_hardware, list_hardware, register_hardware, create_custom_asic,
    HARDWARE_REGISTRY, BLACKWELL_B10,
)
from src.roofline.auto_quantize import recommend_quantization
from src.roofline.tiling_model import analyze_tiling, sweep_tilings
from api.schemas import (
    GEMMSpec, HardwareSpecInput, RooflinePoint, RooflineLine,
    TilingResult, RecommendationResult, AnalyzeResponse, SweepResponse,
    NVMLStatusResponse, HardwareListItem,
)

app = FastAPI(title="Blackwell GEMM Roofline Analyzer", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global NVML monitor (lazy-init)
_nvml_monitor = None


def _get_nvml():
    global _nvml_monitor
    if _nvml_monitor is None:
        try:
            from src.nvml.monitor import NVMLMonitor
            _nvml_monitor = NVMLMonitor()
            _nvml_monitor.init()
        except Exception:
            return None
    return _nvml_monitor


def _make_roofline_lines(hw) -> List[RooflineLine]:
    """Generate roofline ceiling lines for a hardware spec."""
    lines = []
    for prec, tflops in hw.peak_flops_tflops.items():
        if tflops > 0:
            lines.append(RooflineLine(
                precision=prec,
                peak_tflops=tflops,
                critical_ai=hw.critical_ai(prec),
            ))
    return lines


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ═══════════════════════════════════════════════
#  ENDPOINTS
# ═══════════════════════════════════════════════

@app.get("/")
def root():
    """Redirect to API docs."""
    return RedirectResponse(url="/docs", status_code=302)


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze_gemm(spec: GEMMSpec, hardware_key: str = "b10", run_all_precisions: bool = False):
    """
    Analyze a single GEMM: simulated roofline + measured (if GPU) + recommendation + tiling.
    If run_all_precisions=True, runs benchmarks for FP16, FP8, NVFP4, INT8, INT4.
    """
    try:
        hw = get_hardware(hardware_key)
    except KeyError:
        raise HTTPException(404, f"Hardware '{hardware_key}' not found")

    calc = RooflineCalculator(hw)
    M, N, K = spec.M, spec.N, spec.K

    # Simulated prediction
    try:
        if M <= 1:
            pred = calc.predict_gemv(N, K, spec.precision)
        else:
            pred = calc.predict_gemm(M, N, K, spec.precision)
    except ValueError as e:
        raise HTTPException(400, str(e))

    sim_point = RooflinePoint(
        ai=pred["ai"],
        tflops=pred["predicted_tflops"],
        time_us=pred["predicted_time_us"],
        label=f"GEMM {M}x{N}x{K} [{spec.precision}]",
        source="simulated",
        precision=spec.precision,
        shape=f"{M}x{N}x{K}",
        bottleneck=pred["bottleneck"],
    )

    # Measured (if CUDA available)
    meas_points = []
    nvml_data = None
    if _has_cuda():
        try:
            from benchmarks.kernel_shell import GEMMKernel, GEMVKernel

            if run_all_precisions:
                # Run sequential benchmarks for all precisions
                precisions_to_test = ["FP16", "FP8_E4M3", "NVFP4", "INT8", "INT4"]
                for prec in precisions_to_test:
                    try:
                        if M <= 1:
                            kernel = GEMVKernel(N, K, prec)
                        else:
                            kernel = GEMMKernel(M, N, K, prec)
                        meas = kernel.benchmark(num_iters=30)
                        meas_points.append(RooflinePoint(
                            ai=meas["ai"],
                            tflops=meas["measured_tflops"],
                            time_us=meas["measured_time_us"],
                            label=f"{M}x{N}x{K} [{prec}]",
                            source="measured",
                            precision=prec,
                            shape=f"{M}x{N}x{K}",
                            bandwidth_gb_s=meas["measured_bandwidth_gb_s"],
                        ))
                        if nvml_data is None:
                            nvml_data = meas.get("nvml")
                    except Exception:
                        # Skip precision if benchmark fails
                        pass
            else:
                # Single precision benchmark (existing behavior)
                if M <= 1:
                    kernel = GEMVKernel(N, K, spec.precision)
                else:
                    kernel = GEMMKernel(M, N, K, spec.precision)
                meas = kernel.benchmark(num_iters=30)
                meas_points.append(RooflinePoint(
                    ai=meas["ai"],
                    tflops=meas["measured_tflops"],
                    time_us=meas["measured_time_us"],
                    label=f"GEMM {M}x{N}x{K} [{spec.precision}] (measured)",
                    source="measured",
                    precision=spec.precision,
                    shape=f"{M}x{N}x{K}",
                    bandwidth_gb_s=meas["measured_bandwidth_gb_s"],
                ))
                nvml_data = meas.get("nvml")
        except Exception:
            pass

    # Recommendation
    rec = recommend_quantization(hardware=hw, M=M, N=N, K=K)
    rec_result = RecommendationResult(
        precision=rec.precision,
        method=rec.method,
        reason=rec.reason,
        predicted_speedup=rec.predicted_speedup,
        memory_bound=rec.memory_bound,
        memory_savings_pct=rec.memory_savings_pct,
    )

    # Tiling analysis
    tiling = analyze_tiling(M, N, K, spec.tile_m, spec.tile_n, spec.tile_k, spec.precision)
    tiling_result = TilingResult(**tiling.to_dict())

    return AnalyzeResponse(
        hardware=hw.name,
        simulated=[sim_point],
        measured=meas_points,
        roofline_lines=_make_roofline_lines(hw),
        recommendation=rec_result,
        tiling=tiling_result,
        nvml=nvml_data,
    )


@app.post("/api/sweep", response_model=SweepResponse)
def sweep_gemm(
    precisions: Optional[List[str]] = None,
    hardware_key: str = "b10",
    run_measured: bool = False,
):
    """Sweep standard GEMM shapes across precisions."""
    try:
        hw = get_hardware(hardware_key)
    except KeyError:
        raise HTTPException(404, f"Hardware '{hardware_key}' not found")

    from benchmarks.gemm_sweep import run_gemm_sweep, DEFAULT_SHAPES
    results = run_gemm_sweep(
        hardware=hw,
        precisions=precisions,
        run_measured=run_measured,
    )

    points = []
    for r in results:
        points.append(RooflinePoint(
            ai=r.sim_ai,
            tflops=r.sim_tflops,
            time_us=r.sim_time_us,
            label=f"{r.M}x{r.N}x{r.K} [{r.precision}]",
            source="simulated",
            precision=r.precision,
            shape=f"{r.M}x{r.N}x{r.K}",
            bottleneck=r.sim_bottleneck,
        ))
        if r.meas_tflops is not None:
            points.append(RooflinePoint(
                ai=r.meas_ai,
                tflops=r.meas_tflops,
                time_us=r.meas_time_us,
                label=f"{r.M}x{r.N}x{r.K} [{r.precision}] (measured)",
                source="measured",
                precision=r.precision,
                shape=f"{r.M}x{r.N}x{r.K}",
                bandwidth_gb_s=r.meas_bandwidth_gb_s,
            ))

    return SweepResponse(
        hardware=hw.name,
        points=points,
        roofline_lines=_make_roofline_lines(hw),
    )


@app.get("/api/nvml/status", response_model=NVMLStatusResponse)
def nvml_status():
    """Live GPU status via NVML."""
    mon = _get_nvml()
    if mon is None:
        raise HTTPException(503, "NVML not available")
    s = mon.sample()
    return NVMLStatusResponse(
        device_name=s.device_name,
        gpu_clock_mhz=s.gpu_clock_mhz,
        mem_clock_mhz=s.mem_clock_mhz,
        power_draw_w=s.power_draw_w,
        power_limit_w=s.power_limit_w,
        temperature_c=s.temperature_c,
        mem_used_mb=s.mem_used_mb,
        mem_total_mb=s.mem_total_mb,
        gpu_utilization_pct=s.gpu_utilization_pct,
        compute_capability=list(s.compute_capability),
    )


@app.get("/api/nvml/stream")
async def nvml_stream():
    """SSE stream of NVML samples for real-time dashboard."""
    import json

    mon = _get_nvml()
    if mon is None:
        raise HTTPException(503, "NVML not available")

    async def generate():
        while True:
            s = mon.sample()
            data = json.dumps(s.to_dict())
            yield f"data: {data}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/hardware", response_model=List[HardwareListItem])
def get_all_hardware():
    """List all registered hardware specs."""
    items = []
    for key in list_hardware():
        hw = get_hardware(key)
        precs = [k for k, v in hw.peak_flops_tflops.items() if v > 0]
        items.append(HardwareListItem(
            key=key,
            name=hw.name,
            bandwidth_gb_s=hw.peak_bandwidth_gb_s,
            precisions=precs,
        ))
    return items


@app.post("/api/hardware")
def add_custom_hardware(spec: HardwareSpecInput):
    """Register a custom ASIC/GPU."""
    hw = create_custom_asic(
        name=spec.name,
        bandwidth_gb_s=spec.bandwidth_gb_s,
        flops_by_precision=spec.flops_by_precision,
    )
    return {"status": "registered", "name": hw.name, "key": spec.name.lower().replace(" ", "_")}


@app.post("/api/recommend", response_model=RecommendationResult)
def get_recommendation(spec: GEMMSpec, hardware_key: str = "b10"):
    """Auto-quantizer recommendation for a GEMM workload."""
    try:
        hw = get_hardware(hardware_key)
    except KeyError:
        raise HTTPException(404, f"Hardware '{hardware_key}' not found")

    rec = recommend_quantization(hardware=hw, M=spec.M, N=spec.N, K=spec.K)
    return RecommendationResult(
        precision=rec.precision,
        method=rec.method,
        reason=rec.reason,
        predicted_speedup=rec.predicted_speedup,
        memory_bound=rec.memory_bound,
        memory_savings_pct=rec.memory_savings_pct,
    )


@app.post("/api/tiling", response_model=List[TilingResult])
def tiling_analysis(
    M: int = 4096, N: int = 4096, K: int = 4096,
    precision: str = "FP16", hardware_key: str = "b10",
):
    """Sweep tile sizes for a GEMM shape."""
    results = sweep_tilings(M, N, K, precision)
    return [TilingResult(**r.to_dict()) for r in results]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
