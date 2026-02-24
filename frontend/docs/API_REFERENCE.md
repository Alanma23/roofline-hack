# API Reference

Complete reference for all backend endpoints.

## Base URL
```
http://localhost:8000
```

## OpenAPI Documentation
Interactive docs available at: `http://localhost:8000/docs`

---

## Table of Contents
1. [Basic Roofline](#basic-roofline)
2. [Run Tracking](#run-tracking)
3. [MoE Analysis](#moe-analysis)
4. [Advanced Roofline](#advanced-roofline)
5. [Hardware Management](#hardware-management)
6. [Utilities](#utilities)

---

## Basic Roofline

### POST /api/analyze
Analyze a single GEMM operation with optional GPU benchmarking.

**Query Parameters:**
- `hardware_key` (string): Hardware to use (`"b10"`, `"b200"`, `"h100"`, `"a100"`)
- `run_all_precisions` (boolean): Run benchmarks for all precisions (default: `false`)

**Request Body:**
```json
{
  "M": 4096,
  "N": 4096,
  "K": 4096,
  "precision": "FP16",
  "use_efficiency": true,
  "tile_m": 128,
  "tile_n": 128,
  "tile_k": 32
}
```

**Response:**
```json
{
  "hardware": "NVIDIA GB10 Grace Blackwell (GX10)",
  "simulated": [{
    "ai": 682.67,
    "tflops": 9.61,
    "time_us": 14339.2,
    "label": "GEMM 4096x4096x4096 [FP16]",
    "bottleneck": "memory",
    "precision": "FP16",
    "efficiency_factor": 0.155,
    "peak_tflops": 62.0,
    "effective_tflops": 9.61
  }],
  "measured": [],
  "roofline_lines": [...],
  "recommendation": {...},
  "tiling": {...}
}
```

### POST /api/sweep
Sweep GEMM shapes across multiple precisions.

**Query Parameters:**
- `precisions` (list[string]): Precisions to test
- `hardware_key` (string): Hardware key
- `run_measured` (boolean): Run GPU benchmarks
- `quick` (boolean): Use subset of shapes

**Response:**
```json
{
  "hardware": "...",
  "points": [...],
  "roofline_lines": [...]
}
```

---

## Run Tracking

### POST /api/runs/save
Save a run configuration and results.

**Request Body:**
```json
{
  "metadata": {
    "run_id": "uuid",
    "timestamp": "2024-01-01T00:00:00Z",
    "name": "GB10 Llama-3 FP8 decode",
    "hardware_key": "b10",
    "workload_type": "transformer",
    "config": {...},
    "results": {...},
    "tags": ["decode", "Llama-3", "FP8_E4M3"],
    "notes": ""
  }
}
```

**Response:**
```json
{
  "status": "saved",
  "run_id": "uuid"
}
```

### GET /api/runs/list
List all saved runs with optional filtering.

**Query Parameters:**
- `hardware_key` (string, optional): Filter by hardware
- `workload_type` (string, optional): Filter by workload type

**Response:**
```json
[
  {
    "run_id": "uuid",
    "timestamp": "2024-01-01T00:00:00Z",
    "name": "...",
    "hardware_key": "b10",
    "workload_type": "transformer",
    "tags": [...],
    "preview": {
      "tflops": "9.61",
      "latency_ms": "14.34",
      "bottleneck": "memory"
    }
  },
  ...
]
```

### GET /api/runs/{run_id}
Load full details for a specific run.

**Response:** Full `RunMetadata` object

### DELETE /api/runs/{run_id}
Delete a saved run.

**Response:**
```json
{
  "status": "deleted",
  "run_id": "uuid"
}
```

### POST /api/runs/compare
Compare multiple runs side-by-side.

**Request Body:**
```json
{
  "run_ids": ["uuid1", "uuid2", "uuid3"]
}
```

**Response:**
```json
{
  "runs": [...],
  "comparison_table": [
    {
      "metric": "tflops",
      "values": {"uuid1": 9.61, "uuid2": 15.2, ...},
      "deltas": {"uuid1": 0.0, "uuid2": 58.2, ...}
    },
    ...
  ]
}
```

### POST /api/runs/export
Export runs as JSON.

**Request Body:**
```json
{
  "run_ids": ["uuid1", "uuid2"]
}
```

**Response:**
```json
{
  "runs": [...],
  "export_format": "json_v1"
}
```

### POST /api/runs/import
Import runs from JSON.

**Request Body:**
```json
{
  "runs": [...]
}
```

**Response:**
```json
{
  "imported": 5,
  "errors": []
}
```

---

## MoE Analysis

### POST /api/moe/analyze
Analyze MoE layer with expert routing and parallelism.

**Query Parameters:**
- `hardware_key` (string): Hardware to use
- `use_efficiency` (boolean): Apply efficiency factors

**Request Body:**
```json
{
  "model": {
    "L": 61,
    "H": 7168,
    "nh": 56,
    "nkv": 8,
    "dh": 128,
    "dff": 18432,
    "V": 129280,
    "gate": true
  },
  "moe": {
    "num_experts": 256,
    "experts_per_token": 8,
    "expert_ffn_dim": 1536,
    "capacity_factor": 1.3,
    "expert_parallel": 8,
    "load_balance_loss_weight": 0.01
  },
  "precision": {
    "w": "FP8_E4M3",
    "a": "FP8_E4M3",
    "kv": "FP8_E4M3",
    "computeAs": "FP8_E4M3"
  },
  "phase": "decode",
  "batch": 1,
  "seq_len": 4096,
  "network_bw_gbs": 900.0,
  "network_latency_us": 3.0,
  "load_imbalance": 0.12
}
```

**Response:**
```json
{
  "router_flops": 1.8e12,
  "router_time_us": 5.2,
  "router_tflops": 346.2,
  "routing_overhead_pct": 2.1,
  "expert_flops": 4.5e14,
  "expert_time_us": 152.3,
  "expert_tflops": 2952.1,
  "all_to_all_bytes": 234567890,
  "all_to_all_time_us": 18.7,
  "communication_overhead_pct": 10.6,
  "total_flops": 4.52e14,
  "total_time_us": 176.2,
  "total_time_ms": 0.176,
  "load_imbalance": 0.12,
  "capacity_factor": 1.3,
  "avg_tokens_per_expert": 512,
  "expert_utilization": [85.2, 92.1, 78.5, ...],
  "sparsity": 0.969,
  "active_experts_pct": 3.1,
  "bottleneck": "communication",
  "recommendations": [
    "Communication overhead 10.6%. Consider reducing EP degree.",
    ...
  ]
}
```

---

## Advanced Roofline

### POST /api/advanced/analyze
Analyze layer with 3-level hardware realism.

**Query Parameters:**
- `hardware_key` (string): `"b10"` or `"tpu_v3"`

**Request Body:**
```json
{
  "name": "Small_Inference",
  "M": 10,
  "N": 1024,
  "K": 1024,
  "input_precision_bits": 8,
  "accumulator_precision_bits": 32
}
```

**Response:**
```json
{
  "actual_tops": 10.24,
  "latency_ms": 0.002,
  "latency_us": 2.0,
  "peak_compute_tops": 131.07,
  "effective_compute_tops": 10.24,
  "mem_bound_tops": 38.14,
  "arithmetic_intensity": 19.07,
  "utilization": 0.078,
  "padded_m": 128,
  "padded_n": 1024,
  "useful_ops": 20971520,
  "total_hw_ops": 268435456,
  "mem_tier": "SRAM (On-Chip)",
  "total_data_bytes": 1099776,
  "input_bytes": 1060864,
  "output_bytes": 40960,
  "bottleneck": "UTILIZATION",
  "hardware": "TPU_v3_Like",
  "input_precision_bits": 8,
  "accumulator_precision_bits": 32
}
```

### POST /api/advanced/batch-sweep
Sweep batch sizes for utilization analysis.

**Query Parameters:**
- `hardware_key` (string): Hardware to use

**Request Body:**
```json
{
  "layer": {
    "name": "LLM_Decode",
    "M": 0,
    "N": 4096,
    "K": 4096,
    "input_precision_bits": 8,
    "accumulator_precision_bits": 32
  },
  "batch_sizes": [1, 8, 16, 32, 64, 128, 256]
}
```

**Response:**
```json
[
  {
    "batch_size": 1,
    "result": {...}
  },
  {
    "batch_size": 8,
    "result": {...}
  },
  ...
]
```

---

## Hardware Management

### GET /api/hardware
List all registered hardware specs.

**Response:**
```json
[
  {
    "key": "b10",
    "name": "NVIDIA GB10 Grace Blackwell (GX10)",
    "bandwidth_gb_s": 287.0,
    "precisions": ["FP64", "FP32", "TF32", "BF16", "FP16", "FP8_E4M3", ...]
  },
  ...
]
```

### POST /api/hardware
Register custom ASIC/GPU.

**Request Body:**
```json
{
  "name": "Custom Chip",
  "bandwidth_gb_s": 500.0,
  "flops_by_precision": {
    "FP16": 100.0,
    "INT8": 200.0
  },
  "memory_gb": 32
}
```

**Response:**
```json
{
  "status": "registered",
  "name": "Custom Chip",
  "key": "custom_chip"
}
```

---

## Utilities

### POST /api/recommend
Get quantization recommendation for a workload.

**Query Parameters:**
- `hardware_key` (string): Hardware to use

**Request Body:**
```json
{
  "M": 4096,
  "N": 4096,
  "K": 4096,
  "precision": "FP16"
}
```

**Response:**
```json
{
  "precision": "NVFP4",
  "method": "W4A16",
  "reason": "Memory-bound workload benefits from weight compression",
  "predicted_speedup": 1.8,
  "memory_bound": true,
  "memory_savings_pct": 75.0
}
```

### POST /api/import-benchmarks
Import externally-collected benchmark data.

**Request Body:**
```json
{
  "simplified": [
    {
      "shape": "4096x4096x4096",
      "precision": "FP16",
      "time_us": 14500.0
    },
    ...
  ]
}
```

**Response:**
```json
{
  "accepted": 5,
  "rejected": 0,
  "points": [...],
  "errors": []
}
```

### GET /api/nvml/status
Get live GPU status (requires NVIDIA GPU).

**Response:**
```json
{
  "device_name": "NVIDIA GB10",
  "gpu_clock_mhz": 1200,
  "mem_clock_mhz": 9400,
  "power_draw_w": 85.2,
  "power_limit_w": 100.0,
  "temperature_c": 65,
  "mem_used_mb": 4096,
  "mem_total_mb": 131072,
  "gpu_utilization_pct": 45,
  "compute_capability": [9, 0]
}
```

---

## Error Responses

All endpoints return standard HTTP error codes:

**400 Bad Request**
```json
{
  "detail": "Invalid precision format: XYZ"
}
```

**404 Not Found**
```json
{
  "detail": "Hardware 'xyz' not found"
}
```

**503 Service Unavailable**
```json
{
  "detail": "NVML not available"
}
```

---

## Rate Limits

No rate limits in current version. For production deployment, consider adding rate limiting middleware.

## Versioning

Current API version: `1.0.0`

No breaking changes expected. New endpoints will be additive.

---

**For interactive API exploration, visit: http://localhost:8000/docs**
