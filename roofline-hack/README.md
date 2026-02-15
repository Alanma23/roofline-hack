# Roofline Analysis + Auto-Quantization Toolkit

TreeHacks 2026 — NVIDIA Track / Edge AI

## Problem

Transformer inference is memory-bound on most hardware. A decode step (GEMV 1x4096x4096) has arithmetic intensity ~1–2 FLOP/byte. Jetson Orin Nano has critical AI ~45; H100 has ~590. In both cases, AI << critical AI, so the kernel is bandwidth-limited: tensor cores idle while waiting for DRAM. Lower precision (INT8, INT4) reduces bytes moved and yields proportional speedup. The question is which format and method to use for a given workload and hardware.

This toolkit predicts performance from roofline theory, measures it on real GPUs, and recommends quantization automatically.

## Roofline Model

Execution time is the maximum of memory time and compute time:

```
FLOPs  = 2 * M * N * K
Bytes  = (M*K + K*N) * bytes_per_element(precision) + M*N * output_bytes
AI     = FLOPs / Bytes   [arithmetic intensity, FLOP/byte]

time_memory  = Bytes / peak_bandwidth
time_compute = FLOPs / peak_flops(precision)
time         = max(time_memory, time_compute)

critical_AI = peak_flops(precision) / peak_bandwidth
```

If AI < critical_AI: memory-bound. Lower precision reduces bytes and time proportionally.
If AI > critical_AI: compute-bound. Lower precision helps via higher tensor core throughput.

Decode (M=1) is always memory-bound. Prefill (M=2048+) can be compute-bound on high-BW GPUs.

## Supported Hardware

| Key | Name | Bandwidth | FP16 TFLOPS | INT8 TFLOPS |
|-----|------|-----------|-------------|-------------|
| jetson_orin_nano | Jetson Orin Nano 8GB | 60 GB/s | 2.7 | 5.5 |
| b10 | Blackwell B10 | 1000 GB/s | 200 | 400 |
| b200 | B200 | 8000 GB/s | 4500 | 9000 |
| h100 | H100 SXM | 3350 GB/s | 1979 | 3958 |
| a100 | A100 SXM | 2039 GB/s | 312 | 624 |

Jetson Orin (Ampere) has no native FP8/FP4; use INT8/INT4. Blackwell B10/B200 support FP8 and NVFP4 natively.

## Precision Formats

Modeled formats: FP32, TF32, FP16, BF16, FP8 E4M3/E5M2, INT8, MXFP8, MXFP6, MXFP4, NVFP4, INT4, NF4, INT2. Block formats (MXFP4, NVFP4, NF4) include scale overhead in bytes-per-element.

## Architecture

```
calculator_shell.py    predict_gemv, predict_gemm, predict_attention, predict_ffn
hardware_registry.py   HardwareSpec for Jetson, B10, B200, H100, A100; create_custom_asic()
auto_quantize.py       recommend_quantization() — ranks precisions by speedup, returns best
tiling_model.py        GEMM tiling: shared memory, wave quantization, SM occupancy, L2 hit rate
kernel_shell.py        GEMV/GEMM benchmarks, CUDA event timing
gemm_sweep.py          Shape x precision sweep, simulated + measured
lowlevel.py            Jetson: nvpmodel, sysfs GPU/EMC freq, power mode
scheduler.py           Power mode + precision from roofline (latency/throughput/power targets)
api/server.py          FastAPI: /api/analyze, /api/recommend, /api/nvml/status, /api/hardware
```

## Jetson-Specific: Low-Level Systems

`benchmarks/jetson/lowlevel.py` exposes:

- Power mode via `nvpmodel -q` (7W vs 15W)
- GPU frequency from sysfs (`/sys/devices/platform/17000000.gpu/devfreq/...`)
- EMC frequency, CPU frequency, memory stats
- `scale_roofline_for_power()` — scales BW/FLOPS by current GPU freq
- `set_power_mode(id)` — `sudo nvpmodel -m 0` (15W) or `-m 1` (7W)

`scheduler.py` recommends power mode, precision, and batch size for latency, throughput, or power targets.

## Running It

### Install

```bash
pip install -r requirements.txt
```

### Simulated roofline (no GPU)

```bash
python3 -m src.roofline.calculator_shell
python3 -m src.roofline.auto_quantize
python3 -m src.roofline.hardware_registry
```

### With GPU

```bash
# Kernel benchmarks (FP16, INT8)
python3 benchmarks/kernel_shell.py

# Theory vs measurement
python3 compare_shell.py

# Full sweep (use hardware key: jetson_orin_nano, b10, b200, h100, a100)
python3 benchmarks/gemm_sweep.py --hardware jetson_orin_nano
python3 benchmarks/gemm_sweep.py --hardware b10 --no-measure
```

### Jetson Orin

```bash
# Validation (FP16 vs INT8)
python3 benchmarks/jetson/validate_jetson.py

# Low-level status
python3 benchmarks/jetson/lowlevel.py

# Scheduler recommendations
python3 benchmarks/jetson/scheduler.py

# Power sweep (15W vs 7W, requires sudo)
python3 benchmarks/jetson/power_bench.py
```

### API server

```bash
uvicorn api.server:app --reload --port 8000
```

```bash
curl -X POST "localhost:8000/api/analyze?hardware_key=jetson_orin_nano" \
  -H "Content-Type: application/json" \
  -d '{"M":1,"N":4096,"K":4096,"precision":"FP16"}'

curl -X POST "localhost:8000/api/recommend?hardware_key=jetson_orin_nano" \
  -H "Content-Type: application/json" \
  -d '{"M":1,"N":4096,"K":4096}'

curl localhost:8000/api/hardware
curl localhost:8000/api/nvml/status
```

### Quantization pipeline (transformers)

```bash
python3 -m quantization.pipeline
```

### Transformer benchmark (TinyLlama FP16/INT8/INT4)

```bash
python3 benchmarks/transformer_bench.py
```

## File Structure

```
roofline-hack/
  src/roofline/
    calculator_shell.py   Roofline math, bytes_per_element, JETSON_ORIN_NANO
    hardware_registry.py  B10, B200, H100, A100, Jetson, create_custom_asic
    auto_quantize.py      recommend_quantization
    tiling_model.py       GEMM tiling analysis
  src/nvml/
    monitor.py            pynvml GPU telemetry
    power_tracker.py      Background sampling during kernel execution
  benchmarks/
    kernel_shell.py       GEMV/GEMM, CUDA events
    gemm_sweep.py         Shape x precision sweep
    transformer_bench.py  Full model inference
    jetson/
      validate_jetson.py  FP16/INT8 validation
      lowlevel.py         nvpmodel, sysfs, scale_roofline_for_power
      power_bench.py      15W vs 7W sweep
      scheduler.py        recommend, apply_decision
  quantization/
    torchao_configs.py   get_quantize_config, apply_quantization
    pipeline.py          run_pipeline (load, recommend, quantize, bench)
  api/
    server.py            FastAPI backend
    schemas.py           Pydantic models
  frontend/
    roofline-calc-v2.jsx React + D3 roofline plot
  compare_shell.py       Theory vs measurement
```

## Expected Results (Jetson Orin Nano)

| Precision | GEMV 4096x4096 Time | Speedup vs FP16 | Bottleneck |
|-----------|---------------------|-----------------|------------|
| FP16 | ~280 us | 1.0x | memory |
| INT8 | ~145 us | ~2.0x | memory |
| INT4 | ~78 us (predicted) | ~3.6x | memory |

INT8 validated at ~1.9x. INT4 predicted; native W4A16 dequant path.

## Limitations

- Jetson Orin has no native FP8/FP4; INT4 uses dequant-to-FP16.
- B10 specs (1000 GB/s, 200T FP16) are placeholders.
- Tiling model is analytical, not empirical.
- Frontend is raw JSX without a build system.
