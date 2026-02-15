# Roofline Performance Toolkit for GB10

A toolkit for predicting and measuring GPU performance using the roofline model, with automatic quantization recommendations for transformer inference on NVIDIA GB10 Grace Blackwell.

## What is this?

This toolkit helps you understand and optimize transformer inference performance by:

1. **Predicting** performance from hardware specs using roofline theory
2. **Measuring** actual performance with CUDA kernels
3. **Recommending** optimal precision formats (FP16, FP8, INT8, INT4, NVFP4)
4. **Applying** quantization automatically via TorchAO

## Why does this matter?

Transformer inference is **memory-bound** on modern GPUs. The bottleneck isn't compute — it's moving data from memory.

Example: A single token decode (GEMV 1×4096×4096) has arithmetic intensity of ~2 FLOP/byte. GB10 can compute 124 TFLOPS but only has 287 GB/s bandwidth. This means the tensor cores sit idle waiting for data.

**The solution:** Lower precision formats (FP8, INT4, NVFP4) move less data, yielding proportional speedup.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Predict performance (no GPU needed)
python src/roofline/calculator_shell.py

# Benchmark on real hardware (requires CUDA GPU)
python benchmarks/kernel_shell.py

# Compare theory vs reality
python compare_shell.py

# Get automatic quantization recommendation
python -c "from src.roofline.auto_quantize import recommend_quantization; \
print(recommend_quantization(M=1, N=4096, K=4096))"
```

## Hardware: NVIDIA GB10 (ASUS Ascent GX10)

**Specs:**
- **Name:** NVIDIA GB10 Grace Blackwell (GX10)
- **Memory:** 128GB LPDDR5X @ 9400MT/s (256-bit bus)
- **Bandwidth:** 287 GB/s
- **FP32:** 31 TFLOPS
- **FP16/BF16:** 62 TFLOPS
- **FP8:** 124 TFLOPS
- **NVFP4:** 1000 TFLOPS (native Blackwell FP4 support!)
- **INT8:** 124 TFLOPS
- **INT4:** 248 TFLOPS
- **Power:** 140W TDP

**Critical AI:** 216 FLOP/byte (FP16), 432 FLOP/byte (FP8), 3484 FLOP/byte (NVFP4)

Most transformer operations have AI < 10, making them heavily memory-bound.

## Roofline Theory (TL;DR)

```
time = max(Bytes/Bandwidth, FLOPs/PeakFLOPs)
```

For memory-bound operations (most transformer inference):
- **2× less bytes → 2× faster**
- FP16 → FP8: 2× faster
- FP16 → INT4: 4× faster
- FP16 → NVFP4: 4× faster with 1000 TFLOPS compute support

## Precision Formats

| Precision | Bytes/Element | GB10 Support | Use Case |
|-----------|---------------|--------------|----------|
| FP32 | 4.0 | ✓ | Baseline, not recommended |
| FP16 | 2.0 | ✓ | Standard inference |
| BF16 | 2.0 | ✓ | Training, stable gradients |
| FP8 E4M3 | 1.0 | ✓ (Blackwell native) | 2× speedup, minimal quality loss |
| FP8 E5M2 | 1.0 | ✓ (Blackwell native) | 2× speedup, different dynamic range |
| INT8 | 1.0 | ✓ | 2× speedup, requires quantization |
| INT4 | 0.5 | ✓ | 4× speedup, more quality loss |
| NVFP4 | ~0.5 | ✓✓ (1000 TFLOPS!) | 4× speedup, Blackwell optimized |
| MXFP4 | ~0.5 | ✓ | Block-scaled FP4 |

## Architecture

```
src/roofline/
├── calculator_shell.py    # Core roofline formulas
├── hardware_registry.py   # GB10, B200, H100, A100 specs
├── auto_quantize.py       # Automatic precision recommender
└── tiling_model.py        # GEMM tiling analysis

benchmarks/
├── kernel_shell.py        # GEMV/GEMM benchmarking
├── gemm_sweep.py          # Systematic shape sweeps
└── transformer_bench.py   # Full transformer benchmarks

quantization/
├── pipeline.py            # End-to-end quantization pipeline
└── torchao_configs.py     # TorchAO INT4/INT8 configs

api/
├── server.py              # FastAPI REST backend
└── schemas.py             # Request/response models

frontend/
└── roofline-calc-v2.jsx   # Interactive React visualizer
```

## Example Usage

### Predict performance for a GEMM

```python
from src.roofline import RooflineCalculator, get_hardware

# Load GB10 specs
hw = get_hardware("b10")
calc = RooflineCalculator(hw)

# Predict GEMM performance (batch=1, M=1, N=4096, K=4096)
result = calc.predict_gemm(M=1, N=4096, K=4096, precision="FP16")
print(f"Predicted time: {result.time_us:.1f} µs")
print(f"Bottleneck: {result.bottleneck}")  # "memory" or "compute"
print(f"Arithmetic Intensity: {result.ai:.2f} FLOP/byte")
```

### Get quantization recommendation

```python
from src.roofline.auto_quantize import recommend_quantization

# Decode scenario (M=1, memory-bound)
rec = recommend_quantization(M=1, N=4096, K=4096, hardware_key="b10")
print(f"Recommended: {rec.precision} ({rec.method})")
print(f"Reason: {rec.reason}")
print(f"Expected speedup: {rec.predicted_speedup:.2f}x vs FP16")

# Typical output:
# Recommended: NVFP4 (native_fp4)
# Reason: Blackwell native FP4, 4.0× speedup, 1000 TFLOPS
# Expected speedup: 4.00x vs FP16
```

### Benchmark on real hardware

```python
from benchmarks.kernel_shell import GEMMKernel

kernel = GEMMKernel(M=1, N=4096, K=4096, precision="FP16")
result = kernel.benchmark(warmup=10, iters=100)

print(f"Measured time: {result.measured_time_us:.1f} µs")
print(f"Achieved: {result.measured_tflops:.2f} TFLOPS")
print(f"Bandwidth: {result.measured_bandwidth_gb_s:.1f} GB/s")
```

### Run REST API server

```bash
uvicorn api.server:app --reload --port 8000
```

Endpoints:
- `POST /api/analyze` - Analyze single GEMM (theory + measurement)
- `POST /api/sweep` - Sweep across shapes and precisions
- `POST /api/recommend` - Get quantization recommendation
- `GET /api/nvml/status` - Live GPU monitoring
- `GET /api/nvml/stream` - Real-time SSE stream

## Expected Results (GB10)

| Precision | GEMV 1×4096×4096 | Speedup vs FP16 | Notes |
|-----------|------------------|-----------------|-------|
| FP16 | ~287 µs | 1.0× | Baseline, memory-bound |
| FP8 E4M3 | ~143 µs | 2.0× | Half the bytes |
| INT8 | ~143 µs | 2.0× | Half the bytes |
| INT4 | ~72 µs | 4.0× | Quarter the bytes |
| NVFP4 | ~72 µs | 4.0× | Quarter the bytes + 1000 TFLOPS |

These are **theoretical predictions**. Actual speedups depend on:
- Kernel efficiency
- Memory access patterns
- Quantization quality (accuracy vs speed tradeoff)

## Supported Hardware

Pre-configured:
- `b10` - NVIDIA GB10 Grace Blackwell (GX10) — **primary target**
- `b200` - NVIDIA B200 (datacenter, 8000 GB/s)
- `h100` - NVIDIA H100 SXM (3350 GB/s)
- `a100` - NVIDIA A100 SXM (2039 GB/s)

Custom hardware:
```python
from src.roofline import create_custom_asic

create_custom_asic(
    name="My GPU",
    bandwidth_gb_s=500.0,
    flops_by_precision={"FP16": 100.0, "FP8": 200.0}
)
```

## TorchAO Integration

Automatic quantization via PyTorch Architecture Optimization:

```python
from quantization.pipeline import run_pipeline

result = run_pipeline(
    model_name="TinyLlama/TinyLlama-1.1B",
    hardware_key="b10",
    M=1,  # batch size
    N=2048,  # hidden dim
    K=2048
)

print(f"Recommended: {result.recommendation.precision}")
print(f"Tokens/sec: {result.tokens_per_sec:.1f}")
```

Supported methods:
- `native_fp8` - Blackwell native FP8 (via `torch._scaled_mm`)
- `native_fp4` - Blackwell native NVFP4 (via tensor cores)
- `int8_weight_only` - TorchAO INT8 PTQ
- `int4_weight_only` - TorchAO INT4 PTQ
- `awq` - Activation-Aware Weight Quantization (not yet implemented)

## Documentation

- **THEORY.md** - Deep dive into roofline math and precision formats
- **GUIDE.md** - Implementation guide with formulas and examples

## GPU Monitoring (NVML)

Real-time GPU status:
```python
from src.nvml import NVMLMonitor

monitor = NVMLMonitor()
status = monitor.sample()
print(f"Power: {status.power_w:.1f}W")
print(f"Temp: {status.temp_c}°C")
print(f"GPU Clock: {status.gpu_clock_mhz} MHz")
```

Background power tracking during benchmarks:
```python
from src.nvml import PowerTracker

with PowerTracker(interval_ms=10) as tracker:
    # Run benchmark
    kernel.benchmark()

summary = tracker.get_summary()
print(f"Avg power: {summary.power_avg_w:.1f}W")
print(f"Energy: {summary.energy_j:.2f}J")
```

## Validation Strategy

1. **Theory validation:** Compare roofline predictions to measured kernel times (target: <15% error)
2. **Quantization validation:** Measure actual speedup vs predicted (FP16 → FP8 → INT4 → NVFP4)
3. **Power validation:** Correlate performance with power draw and temperature

## Limitations

- GB10 specs are based on published specifications (31 TFLOPS FP32, 1000 TFLOPS FP4)
- Roofline model assumes perfect memory access patterns
- Actual speedups depend on quantization quality (accuracy tradeoff)
- Tiling model is analytical, not empirically validated on GB10

## References

- [ASUS Ascent GX10 Review - ServeTheHome](https://www.servethehome.com/asus-ascent-gx10-review-a-new-nvidia-gb10-solution/)
- [NVIDIA GB10 Architecture - Hot Chips 2025](https://www.servethehome.com/nvidia-outlines-gb10-soc-architecture-at-hot-chips-2025/)
- [Roofline Model (Williams et al.)](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf)
- [TorchAO Documentation](https://github.com/pytorch/ao)

## License

MIT License - TreeHacks 2026 NVIDIA Track
