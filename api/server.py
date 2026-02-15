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

Public API (for anyone on the site, no GPU/NVML required):
  GET  /api/public/hardware  — List hardware specs
  POST /api/public/analyze   — Analyze GEMM (simulation only)
  POST /api/public/recommend — Quantization recommendation

Run:
  uvicorn api.server:app --reload --port 8000

Single-server deployment (app + API for anyone on the site):
  cd frontend && npm run build && cd ..
  uvicorn api.server:app --host 0.0.0.0 --port 8000
  → Visit http://localhost:8000 for the app; /api/* for API; /docs for OpenAPI.
"""

import sys
import asyncio
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIST = ROOT / "frontend" / "dist"
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
    ImportBenchmarkResponse, FlexibleImportRequest, SimplifiedBenchmarkPoint,
)

app = FastAPI(title="Blackwell GEMM Roofline Analyzer", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Public API router — for anyone on the site (simulation-only, no GPU/NVML)
public_router = APIRouter(prefix="/api/public", tags=["public"])

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
    """Serve the app if built, else redirect to API docs."""
    if FRONTEND_DIST.exists() and (FRONTEND_DIST / "index.html").exists():
        return FileResponse(FRONTEND_DIST / "index.html")
    return RedirectResponse(url="/docs", status_code=302)


def _mount_frontend():
    """Mount built frontend so anyone visiting gets app + API from one server."""
    if not FRONTEND_DIST.exists():
        return
    assets = FRONTEND_DIST / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

_mount_frontend()


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
                    except Exception as e:
                        # Skip precision if benchmark fails
                        import traceback
                        print(f"[BENCHMARK ERROR] {prec}: {e}")
                        traceback.print_exc()
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
        except Exception as e:
            import traceback
            print(f"[BENCHMARK ERROR] {spec.precision}: {e}")
            traceback.print_exc()

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


# Quick shapes for faster sweep (decode + prefill + square)
QUICK_SWEEP_SHAPES = [(1, 4096, 4096), (2048, 4096, 4096), (4096, 4096, 4096)]


@app.post("/api/sweep", response_model=SweepResponse)
def sweep_gemm(
    precisions: Optional[List[str]] = None,
    hardware_key: str = "b10",
    run_measured: bool = False,
    quick: bool = False,
):
    """Sweep GEMM shapes across precisions. Use quick=True for fewer shapes."""
    try:
        hw = get_hardware(hardware_key)
    except KeyError:
        raise HTTPException(404, f"Hardware '{hardware_key}' not found")

    from benchmarks.gemm_sweep import run_gemm_sweep, DEFAULT_SHAPES
    shapes = QUICK_SWEEP_SHAPES if quick else None
    results = run_gemm_sweep(
        hardware=hw,
        shapes=shapes,
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


BENCHMARK_SHAPES_DEFAULT = "1,4096,4096;2048,4096,4096;4096,4096,4096"
BENCHMARK_PRECISIONS_DEFAULT = "FP16,FP8_E4M3,NVFP4,INT8"

# Saturation sweep shapes (GEMM + GEMV) — push GPU to max
SATURATION_SHAPES = [
    (4096, 4096, 4096), (8192, 4096, 4096), (8192, 8192, 4096),
    (8192, 8192, 8192), (16384, 4096, 4096), (16384, 8192, 4096),
    (1, 4096, 4096), (1, 8192, 8192), (1, 16384, 4096), (1, 4096, 14336),
]
SATURATION_PRECISIONS = ["FP16", "FP8_E4M3", "TF32"]


@app.get("/api/benchmark/stream")
async def benchmark_stream(
    hardware_key: str = "b10",
    shapes: str = BENCHMARK_SHAPES_DEFAULT,
    precisions: str = BENCHMARK_PRECISIONS_DEFAULT,
    saturate: bool = False,
):
    """
    SSE stream: run benchmarks across shapes × precisions, yield each result as it completes.
    Saturates GPU with each format for live throughput measurement.
    """
    import json
    try:
        hw = get_hardware(hardware_key)
    except KeyError:
        raise HTTPException(404, f"Hardware '{hardware_key}' not found")

    if saturate:
        shape_tuples = [s for s in SATURATION_SHAPES]
        prec_list = list(SATURATION_PRECISIONS)
        num_iters, warmup = 100, 50
    else:
        shape_tuples = []
        for part in shapes.split(";"):
            nums = [int(x.strip()) for x in part.split(",") if x.strip()]
            if len(nums) == 3:
                shape_tuples.append(tuple(nums))
        prec_list = [p.strip() for p in precisions.split(",") if p.strip()]
        if not shape_tuples:
            shape_tuples = [(1, 4096, 4096), (2048, 4096, 4096), (4096, 4096, 4096)]
        if not prec_list:
            prec_list = ["FP16", "FP8_E4M3", "NVFP4", "INT8"]
        num_iters, warmup = 50, 25

    async def generate():
        try:
            from benchmarks.kernel_shell import GEMMKernel, GEMVKernel, check_cuda
        except ImportError:
            yield f"data: {json.dumps({'error': 'Kernel module not available'})}\n\n"
            return
        if not check_cuda():
            yield f"data: {json.dumps({'error': 'CUDA not available'})}\n\n"
            return

        total = len(shape_tuples) * len(prec_list)
        done = 0

        for M, N, K in shape_tuples:
            for prec in prec_list:
                try:
                    def run_one():
                        if M <= 1:
                            k = GEMVKernel(N, K, prec)
                        else:
                            k = GEMMKernel(M, N, K, prec)
                        return k.benchmark(num_iters=num_iters, warmup=warmup)

                    meas = await asyncio.to_thread(run_one)
                    done += 1
                    evt = {
                        "shape": f"{M}x{N}x{K}",
                        "precision": prec,
                        "time_us": meas["measured_time_us"],
                        "tflops": meas["measured_tflops"],
                        "bandwidth_gb_s": meas.get("measured_bandwidth_gb_s"),
                        "ai": meas["ai"],
                        "progress": f"{done}/{total}",
                    }
                    yield f"data: {json.dumps(evt)}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e), 'shape': f'{M}x{N}x{K}', 'precision': prec})}\n\n"
        yield f"data: {json.dumps({'done': True, 'total': total})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


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


@app.post("/api/import-benchmarks", response_model=ImportBenchmarkResponse)
def import_benchmarks(req: FlexibleImportRequest):
    """Import externally-collected benchmark data (no GPU required)."""
    accepted_points = []
    errors = []

    # Process full-format points
    if req.points:
        for i, pt in enumerate(req.points):
            try:
                if pt.ai <= 0 or pt.tflops <= 0 or pt.time_us <= 0:
                    errors.append(f"Point {i}: ai/tflops/time_us must be > 0")
                    continue
                pt_dict = pt.dict()
                pt_dict["source"] = "measured"
                accepted_points.append(RooflinePoint(**pt_dict))
            except Exception as e:
                errors.append(f"Point {i}: {str(e)}")

    # Process simplified-format points
    if req.simplified:
        for i, sp in enumerate(req.simplified):
            try:
                shape_parts = [int(x) for x in sp.shape.split("x")]
                if len(shape_parts) == 2:
                    N, K = shape_parts
                    flops = 2 * N * K
                    bpe = bytes_per_element(sp.precision)
                    bytes_total = K * bpe + N * K * bpe + N * 2
                elif len(shape_parts) == 3:
                    M, N, K = shape_parts
                    flops = 2 * M * N * K
                    bpe = bytes_per_element(sp.precision)
                    bytes_total = (M * K + K * N) * bpe + M * N * 2
                else:
                    errors.append(f"Point {i}: shape must be NxK or MxNxK")
                    continue

                ai = sp.ai if sp.ai else flops / bytes_total
                tflops = sp.tflops if sp.tflops else flops / (sp.time_us * 1e-6) / 1e12
                bw = sp.bandwidth_gb_s if sp.bandwidth_gb_s else bytes_total / (sp.time_us * 1e-6) / 1e9
                label = sp.label or f"{sp.shape} [{sp.precision}]"

                accepted_points.append(RooflinePoint(
                    ai=ai, tflops=tflops, time_us=sp.time_us,
                    label=label, source="measured", precision=sp.precision,
                    shape=sp.shape, bandwidth_gb_s=bw,
                ))
            except Exception as e:
                errors.append(f"Point {i}: {str(e)}")

    return ImportBenchmarkResponse(
        accepted=len(accepted_points),
        rejected=len(errors),
        points=accepted_points,
        errors=errors,
    )


@app.post("/api/tiling", response_model=List[TilingResult])
def tiling_analysis(
    M: int = 4096, N: int = 4096, K: int = 4096,
    precision: str = "FP16", hardware_key: str = "b10",
):
    """Sweep tile sizes for a GEMM shape."""
    results = sweep_tilings(M, N, K, precision)
    return [TilingResult(**r.to_dict()) for r in results]


# ═══════════════════════════════════════════════
#  PUBLIC API — for anyone on the site (no GPU required)
# ═══════════════════════════════════════════════

@public_router.get("/hardware", response_model=List[HardwareListItem])
def public_get_hardware():
    """List all registered hardware specs (public)."""
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


@public_router.post("/analyze", response_model=AnalyzeResponse)
def public_analyze_gemm(spec: GEMMSpec, hardware_key: str = "b10"):
    """
    Analyze a single GEMM: simulation + recommendation + tiling only.
    No GPU/measurement — safe for anyone on the site.
    """
    try:
        hw = get_hardware(hardware_key)
    except KeyError:
        raise HTTPException(404, f"Hardware '{hardware_key}' not found")

    calc = RooflineCalculator(hw)
    M, N, K = spec.M, spec.N, spec.K

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

    rec = recommend_quantization(hardware=hw, M=M, N=N, K=K)
    rec_result = RecommendationResult(
        precision=rec.precision,
        method=rec.method,
        reason=rec.reason,
        predicted_speedup=rec.predicted_speedup,
        memory_bound=rec.memory_bound,
        memory_savings_pct=rec.memory_savings_pct,
    )

    tiling = analyze_tiling(M, N, K, spec.tile_m, spec.tile_n, spec.tile_k, spec.precision)
    tiling_result = TilingResult(**tiling.to_dict())

    return AnalyzeResponse(
        hardware=hw.name,
        simulated=[sim_point],
        measured=[],
        roofline_lines=_make_roofline_lines(hw),
        recommendation=rec_result,
        tiling=tiling_result,
        nvml=None,
    )


@public_router.post("/recommend", response_model=RecommendationResult)
def public_recommend(spec: GEMMSpec, hardware_key: str = "b10"):
    """Auto-quantizer recommendation for a GEMM (public)."""
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


@public_router.post("/import-benchmarks", response_model=ImportBenchmarkResponse)
def public_import_benchmarks(req: FlexibleImportRequest):
    """Import externally-collected benchmark data (public, no GPU required)."""
    return import_benchmarks(req)


app.include_router(public_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
