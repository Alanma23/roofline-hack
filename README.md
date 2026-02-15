# Roofline Analysis + Auto-Quantization Toolkit

**TreeHacks 2026 — NVIDIA Track**

A toolkit that predicts transformer inference performance from roofline theory, measures it on real GPUs, and **automatically recommends and applies quantization** (PTQ via **torchao**). Primary target: Blackwell GB10/B10 and datacenter GPUs.

---

## Table of Contents

1. [Problem & Solution](#problem--solution)
2. [Theory: Roofline Model](#theory-roofline-model)
3. [Precision Formats](#precision-formats)
4. [Auto-Quantization Flow](#auto-quantization-flow)
5. [TorchAO Integration](#torchao-integration)
6. [Deployment to Real Hardware](#deployment-to-real-hardware)
7. [Packages & Infrastructure](#packages--infrastructure)
8. [Use Cases](#use-cases)
9. [Deployment to Real Hardware](#deployment-to-real-hardware)
10. [File Structure](#file-structure)
11. [Running Everything](#running-everything)

---

## Problem & Solution

**Problem:** Transformer inference is **memory-bound** on most hardware. A decode step (GEMV 1×4096×4096) has arithmetic intensity ~1–2 FLOP/byte. H100 has critical AI ~590; Blackwell GB10/B10 has critical AI ~200. AI ≪ critical AI → tensor cores idle while waiting for DRAM. Lower precision (INT8, INT4, FP8, FP4) reduces bytes moved and yields proportional speedup. The question: **which format and method** for a given workload and hardware?

**Solution:** This toolkit:
1. **Predicts** performance from roofline theory (simulated)
2. **Measures** on real GPUs (CUDA kernels, NVML power)
3. **Recommends** optimal precision via `recommend_quantization()`
4. **Applies** quantization via **torchao** (PyTorch Architecture Optimization)
5. **Validates** on Blackwell B10 and datacenter GPUs

---

## Theory: Roofline Model

Execution time is the maximum of memory time and compute time:

```
FLOPs  = 2 × M × N × K
Bytes  = (M×K + K×N) × bytes_per_element(precision) + M×N × output_bytes
AI     = FLOPs / Bytes   [arithmetic intensity, FLOP/byte]

time_memory  = Bytes / peak_bandwidth
time_compute = FLOPs / peak_flops(precision)
time         = max(time_memory, time_compute)

critical_AI = peak_flops(precision) / peak_bandwidth
```

- **AI < critical_AI** → memory-bound. Lower precision reduces bytes and time proportionally.
- **AI > critical_AI** → compute-bound. Lower precision helps via higher tensor-core throughput.

**Decode (M=1)** is always memory-bound. **Prefill (M=2048+)** can be compute-bound on high-BW GPUs.

See `docs/THEORY_MATH.md` for per-operator FLOP/byte derivations (attention, FFN, KV cache).

---

## Precision Formats

| Format | Bytes/elem | Hardware | Notes |
|--------|------------|----------|-------|
| FP32 | 4.0 | Universal | Baseline |
| FP16 | 2.0 | H100, B10 | Standard inference |
| INT8 | 1.0 | H100, B10 | 2× speedup typical |
| INT4 | 0.5 | B10 | 4× speedup predicted |
| FP8 E4M3/E5M2 | 1.0 | H100+, B10 | Blackwell native |
| NVFP4, MXFP4 | ~0.5 | B10, B200 | Blackwell FP4 tensor cores |

Blackwell B10/B200 support FP8 and NVFP4 natively.

See `docs/THEORY_FORMATS.md` for full catalog (NF4, MXFP, block formats).

---

## Auto-Quantization Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  recommend_quantization(hardware, M, N, K, phase, memory_limit)   │
│  - Predicts time for each precision (FP16, INT8, INT4, FP8...)   │
│  - Ranks by speedup vs FP16                                      │
│  - memory_limit_gb: prefer smallest format when constrained      │
│  - Returns: precision, method (PTQ/AWQ/native_fp8), reason      │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  map_precision_to_torchao(precision)                             │
│  NVFP4/NF4/MXFP4 → INT4   FP8 → INT8   INT4/INT8 → passthrough   │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  apply_quantization(model, precision, group_size=128)            │
│  - quantize_(model, int4_weight_only() | int8_weight_only())     │
│  - In-place modification of Linear layers                       │
└─────────────────────────────────────────────────────────────────┘
```

**Pipeline:** `quantization/pipeline.py` → `run_pipeline()` chains: load model → recommend → quantize → benchmark.

---

## TorchAO Integration

**TorchAO** = PyTorch Architecture Optimization (https://github.com/pytorch/ao). **Not** "torchAL".

| Role | Module | Functions |
|------|--------|-----------|
| Config | `quantization/torchao_configs.py` | `get_quantize_config()`, `map_precision_to_torchao()` |
| Apply | `quantization/torchao_configs.py` | `apply_quantization()` |
| Pipeline | `quantization/pipeline.py` | `run_pipeline()` |

TorchAO provides `quantize_()`, `int4_weight_only()`, `int8_weight_only()`. Lazy import: if torchao is not installed, `apply_quantization` returns `False` and the pipeline continues without quantization.

See `docs/TORCHAO.md` for deep dive.

---


## Packages & Infrastructure

| Package | Purpose |
|---------|---------|
| `torch` | GEMM/GEMV kernels, model inference |
| `torchao` | Weight-only quantization (INT4, INT8) |
| `transformers` | Model loading (TinyLlama, Phi-2) |
| `fastapi`, `uvicorn` | REST API |
| `pynvml` | GPU telemetry (power, utilization) |

**Optional:** `benchmarks/requirements.txt` for extra benchmarks.

---

## Use Cases

| Scenario | Hardware | Phase | Typical Recommendation |
|----------|----------|-------|------------------------|
| Desktop decode | Blackwell B10 | decode | INT4, NVFP4, or FP8 |
| Batch prefill | B10, H100 | prefill | FP16 or FP8 (compute-bound) |
| Datacenter inference | H100, B200 | decode/prefill | FP8 or NVFP4 |

---

## Deployment to Real Hardware

### Blackwell B10 (GB10)

1. **Measure specs:** Run `benchmarks/gemm_sweep.py --hardware b10` to validate B10 bandwidth/FLOPS
2. **Update `hardware_registry.py`:** Replace placeholder B10 values with measured data
3. **Compare:** `python3 compare_shell.py` — theory vs measurement

### API Server

```bash
uvicorn api.server:app --reload --port 8000
```

Endpoints: `/api/analyze`, `/api/recommend`, `/api/nvml/status`, `/api/hardware`, `/api/sweep`, `/api/tiling`.

---

## File Structure

```
roofline-hack/
├── src/roofline/
│   ├── calculator_shell.py   # Roofline math, bytes_per_element
│   ├── hardware_registry.py # B10, B200, H100, A100, create_custom_asic
│   ├── auto_quantize.py     # recommend_quantization
│   └── tiling_model.py      # GEMM tiling analysis
├── src/nvml/
│   ├── monitor.py           # pynvml GPU telemetry
│   └── power_tracker.py     # Background sampling during kernels
├── benchmarks/
│   ├── kernel_shell.py      # GEMV/GEMM, CUDA events, FP8/INT8/INT4
│   ├── gemm_sweep.py        # Shape × precision sweep
│   └── transformer_bench.py # Full model FP16/INT8/INT4
├── quantization/
│   ├── torchao_configs.py   # get_quantize_config, apply_quantization, map_precision_to_torchao
│   └── pipeline.py          # run_pipeline (load, recommend, quantize, bench)
├── api/
│   ├── server.py            # FastAPI backend
│   └── schemas.py           # Pydantic models
├── frontend/
│   └── roofline-calc-v2.jsx # React + D3 roofline plot
├── docs/
│   ├── THEORY_MATH.md       # Per-operator FLOP/byte derivations
│   ├── THEORY_FORMATS.md    # Precision format catalog
│   └── TORCHAO.md           # TorchAO deep dive
├── compare_shell.py         # Theory vs measurement
├── requirements.txt
└── README.md
```

---

## Running Everything

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
python3 benchmarks/kernel_shell.py
python3 compare_shell.py
python3 benchmarks/gemm_sweep.py --hardware b10 --no-measure
```

### Quantization pipeline

```bash
python3 -m quantization.pipeline
```

### Transformer benchmark

```bash
python3 benchmarks/transformer_bench.py
```

### API

```bash
uvicorn api.server:app --reload --port 8000
curl -X POST "localhost:8000/api/recommend?hardware_key=b10" \
  -H "Content-Type: application/json" -d '{"M":1,"N":4096,"K":4096}'
```

---

## Supported Hardware

| Key | Name | Bandwidth | FP16 TFLOPS | INT8 TFLOPS |
|-----|------|-----------|-------------|-------------|
| b10 | Blackwell B10 | 1000 GB/s | 200 | 400 |
| b200 | B200 | 8000 GB/s | 4500 | 9000 |
| h100 | H100 SXM | 3350 GB/s | 1979 | 3958 |
| a100 | A100 SXM | 2039 GB/s | 312 | 624 |

B10 specs are placeholders; measure and update for your GB10.

---

## Expected Results (Blackwell B10)

| Precision | GEMV 4096×4096 Time | Speedup vs FP16 | Bottleneck |
|-----------|---------------------|-----------------|------------|
| FP16 | ~34 μs | 1.0× | memory |
| FP8 | ~17 μs | ~2.0× | memory |
| NVFP4 | ~9.5 μs | ~3.5× | memory |
| INT4 | ~8.4 μs | ~4.0× | memory |

Predictions based on placeholder B10 specs (1000 GB/s, 200 TFLOPS FP16).

---

## Limitations

- B10 specs are placeholders; measure on your GB10.
- Tiling model is analytical, not empirical.
- TorchAO FP8 weight-only not available; FP8 recommendations map to INT8.
