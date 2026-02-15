# Implementation Plan: Simulation → Validation Pipeline

## COMPACT EXECUTION PLAN

### Context
- **Hardware:** NVIDIA GB10 (48 SMs, 287 GB/s BW, 119.6 GB memory)
- **Current Issue:** Simulation predictions don't match actual GPU benchmarks
- **Goal:** Align simulation with reality, enable side-by-side comparison for demo

---

## Phase 1: Fix Core Simulation (45 minutes)

### Task 1.1: Update GB10 Hardware Spec (10 min)
**File:** `src/roofline/hardware_registry.py:36-56`

**Changes:**
```python
BLACKWELL_B10 = HardwareSpec(
    name="NVIDIA GB10 Grace Blackwell (GX10)",
    peak_bandwidth_gb_s=287.0,
    peak_flops_tflops={
        "FP64": 15.5,
        "FP32": 31.0,
        "TF32": 62.0,
        "BF16": 62.0,
        "FP16": 62.0,
        "FP8_E4M3": 164.0,  # ← UPDATE from 124 to 164 (empirical)
        "FP8_E5M2": 164.0,  # ← UPDATE
        "NVFP4": 1000.0,
        "MXFP4": 1000.0,
        "MXFP8": 164.0,     # ← UPDATE
        "INT8": 124.0,
        "INT4": 248.0,
    },
    # NEW: Kernel efficiency factors (fraction of peak achieved)
    kernel_efficiency={
        "gemv": {
            "FP16": 0.575, "BF16": 0.585, "TF32": 0.43,
            "FP8_E4M3": 1.68, "INT8": 0.14,  # INT8 GEMV has no native path
        },
        "gemm": {
            "FP16": 0.155, "BF16": 0.155, "TF32": 0.602,
            "FP8_E4M3": 1.0, "INT8": 0.435,  # FP8 now matches updated 164 TFLOPS spec
        },
    },
)
```

### Task 1.2: Extend HardwareSpec Dataclass (5 min)
**File:** `src/roofline/calculator_shell.py:86-104`

**Changes:**
```python
@dataclass
class HardwareSpec:
    name: str
    peak_bandwidth_gb_s: float
    peak_flops_tflops: Dict[str, float]
    kernel_efficiency: Dict[str, Dict[str, float]] = None  # NEW

    def __post_init__(self):
        if self.kernel_efficiency is None:
            self.kernel_efficiency = {}
```

### Task 1.3: Apply Efficiency Factors in Calculator (20 min)
**File:** `src/roofline/calculator_shell.py:110-231`

**Changes:**
```python
def predict_gemv(self, N: int, K: int, precision: str, use_efficiency: bool = True) -> Dict:
    # ... existing FLOP/byte calculation ...

    peak_tflops = self._get_peak_tflops(precision)

    # NEW: Apply kernel efficiency
    if use_efficiency and self.hardware.kernel_efficiency:
        eff = self.hardware.kernel_efficiency.get("gemv", {}).get(precision, 1.0)
        effective_tflops = peak_tflops * eff
    else:
        effective_tflops = peak_tflops

    t_math = flop_count / (effective_tflops * 1e12)
    # ... rest unchanged ...

    return {
        # ... existing fields ...
        "efficiency_factor": eff if use_efficiency else 1.0,  # NEW
        "effective_tflops": effective_tflops,  # NEW
    }
```

Repeat for `predict_gemm()`, `predict_attention()`, `predict_ffn()`.

### Task 1.4: Refactor Duplicate Code (10 min)
**File:** `src/roofline/calculator_shell.py`

**Extract helper:**
```python
def _compute_roofline(self, flop_count: int, bytes_used: float,
                      precision: str, kernel_type: str) -> Dict:
    """Unified roofline calculation with efficiency factors."""
    peak_tflops = self._get_peak_tflops(precision)

    # Apply efficiency
    eff = 1.0
    if self.hardware.kernel_efficiency:
        eff = self.hardware.kernel_efficiency.get(kernel_type, {}).get(precision, 1.0)
    effective_tflops = peak_tflops * eff

    t_math = flop_count / (effective_tflops * 1e12)
    t_comm = bytes_used / (self.hardware.peak_bandwidth_gb_s * 1e9)
    pred_time_s = max(t_math, t_comm)

    return {
        "predicted_time_us": pred_time_s * 1e6,
        "predicted_tflops": (flop_count / pred_time_s) / 1e12 if pred_time_s > 0 else 0,
        "effective_tflops": effective_tflops,
        "ai": flop_count / bytes_used if bytes_used > 0 else 0,
        "critical_ai": self.hardware.critical_ai(precision),
        "bottleneck": "memory" if t_comm >= t_math else "compute",
        "efficiency_factor": eff,
        "flops": flop_count,
        "bytes": int(bytes_used),
    }
```

---

## Phase 2: Validation API (30 minutes)

### Task 2.1: Add Validation Schema (10 min)
**File:** `api/schemas.py`

**Add:**
```python
class BenchmarkResult(BaseModel):
    """Single benchmark measurement from GPU."""
    kernel_type: str  # "gemv" or "gemm"
    shape: List[int]  # [N, K] for GEMV or [M, N, K] for GEMM
    precision: str
    measured_time_us: float
    measured_tflops: float
    measured_bandwidth_gb_s: float
    ai: float

class ValidationRequest(BaseModel):
    """Request to validate simulation against measurements."""
    hardware_key: str  # "b10", "b200", etc.
    benchmarks: List[BenchmarkResult]

class ValidationResponse(BaseModel):
    """Comparison of predicted vs measured."""
    results: List[dict]  # {predicted, measured, error_pct, ...}
    summary: dict  # {mean_error, max_error, ...}
```

### Task 2.2: Add Validation Endpoint (20 min)
**File:** `api/server.py`

**Add:**
```python
@app.post("/api/validate")
def validate_simulation(req: ValidationRequest):
    """Compare simulation predictions against actual benchmark results."""
    from src.roofline.hardware_registry import get_hardware
    from src.roofline.calculator_shell import RooflineCalculator

    hw = get_hardware(req.hardware_key)
    calc = RooflineCalculator(hw)

    results = []
    errors = []

    for bench in req.benchmarks:
        # Predict
        if bench.kernel_type == "gemv":
            pred = calc.predict_gemv(bench.shape[0], bench.shape[1], bench.precision)
        elif bench.kernel_type == "gemm":
            pred = calc.predict_gemm(*bench.shape, bench.precision)
        else:
            continue

        # Compare
        error_pct = abs(pred["predicted_time_us"] - bench.measured_time_us) / bench.measured_time_us * 100
        errors.append(error_pct)

        results.append({
            "kernel_type": bench.kernel_type,
            "shape": bench.shape,
            "precision": bench.precision,
            "predicted_time_us": pred["predicted_time_us"],
            "measured_time_us": bench.measured_time_us,
            "error_pct": round(error_pct, 1),
            "predicted_tflops": pred["predicted_tflops"],
            "measured_tflops": bench.measured_tflops,
            "bottleneck": pred["bottleneck"],
        })

    return {
        "results": results,
        "summary": {
            "mean_error_pct": round(sum(errors) / len(errors), 1) if errors else 0,
            "max_error_pct": round(max(errors), 1) if errors else 0,
            "num_benchmarks": len(results),
        },
    }
```

---

## Phase 3: Frontend Validation Panel (60 minutes)

### Task 3.1: Create Validation Component (45 min)
**File:** `frontend/src/components/ValidationPanel.jsx` (NEW)

**Features:**
- Textarea to paste JSON benchmark results
- "Validate" button → calls `/api/validate`
- Table showing Predicted | Measured | Error %
- Color coding: green (<10%), yellow (10-25%), red (>25%)

### Task 3.2: Overlay Measured Points on Roofline Plot (30 min)
**File:** `frontend/roofline-calc-v2.jsx`

**Changes:**
- Add `measuredPoints` state
- Render second scatter series with different marker (triangle vs circle)
- Legend: "Predicted (○)" vs "Measured (△)"

---

## Phase 4: Export Benchmark Script (15 minutes)

### Task 4.1: Create Export Generator (15 min)
**File:** `benchmarks/export_benchmark.py` (NEW)

**Generate:**
```python
def generate_benchmark_script(hardware_key: str, shapes: List[tuple]) -> str:
    """Generate Python script to run benchmarks and output JSON."""
    template = '''
#!/usr/bin/env python3
"""Auto-generated benchmark script for {hardware}"""
import json
from benchmarks.kernel_shell import GEMVKernel, GEMMKernel

results = []

# GEMV benchmarks
for shape in {gemv_shapes}:
    for prec in ["FP16", "BF16", "TF32", "FP8_E4M3", "INT8"]:
        try:
            kernel = GEMVKernel(*shape, prec)
            r = kernel.benchmark()
            results.append({{
                "kernel_type": "gemv",
                "shape": list(shape),
                "precision": prec,
                "measured_time_us": r["measured_time_us"],
                "measured_tflops": r["measured_tflops"],
                "measured_bandwidth_gb_s": r["measured_bandwidth_gb_s"],
                "ai": r["ai"],
            }})
        except Exception as e:
            print(f"SKIP {{prec}}: {{e}}")

# GEMM benchmarks
for shape in {gemm_shapes}:
    for prec in ["FP16", "BF16", "TF32", "FP8_E4M3", "INT8"]:
        try:
            kernel = GEMMKernel(*shape, prec)
            r = kernel.benchmark()
            results.append({{
                "kernel_type": "gemm",
                "shape": list(shape),
                "precision": prec,
                "measured_time_us": r["measured_time_us"],
                "measured_tflops": r["measured_tflops"],
                "measured_bandwidth_gb_s": r["measured_bandwidth_gb_s"],
                "ai": r["ai"],
            }})
        except Exception as e:
            print(f"SKIP {{prec}}: {{e}}")

# Output JSON
print(json.dumps({{"hardware_key": "{hardware}", "benchmarks": results}}, indent=2))
'''
    return template.format(
        hardware=hardware_key,
        gemv_shapes=[(4096, 4096), (8192, 8192)],
        gemm_shapes=[(4096, 4096, 4096), (8192, 8192, 8192)],
    )
```

---

## Phase 5: Documentation (20 minutes)

### Task 5.1: Update README (20 min)
**File:** `README.md`

**Add section:**
```markdown
## Validation Workflow

### 1. Generate Benchmark Script
```bash
# From UI or CLI
python -m benchmarks.export_benchmark --hardware b10 > benchmark_gb10.py
```

### 2. Run on GPU
```bash
python benchmark_gb10.py > results.json
```

### 3. Validate in UI
- Open http://localhost:5173
- Click "Validation" tab
- Paste `results.json` contents
- Click "Validate"
- See predicted vs measured comparison

### 4. Interpret Results
- **Green** (<10% error): Excellent match
- **Yellow** (10-25% error): Good, within modeling uncertainty
- **Red** (>25% error): Investigate (wrong kernel, unsupported precision, etc.)
```

---

## EXECUTION ORDER

### Critical Path (Must Do Now)
1. ✅ Task 1.1: Update GB10 spec (10 min)
2. ✅ Task 1.2: Extend HardwareSpec (5 min)
3. ✅ Task 1.3: Apply efficiency factors (20 min)
4. ✅ Task 1.4: Refactor duplicates (10 min)
5. ✅ Test predictions match actual benchmarks

**Checkpoint:** Verify simulation now within ±10% of actual GB10 results

### Demo Preparation (Parallel Tracks)
**Track A (Backend):**
6. Task 2.1: Add schemas (10 min)
7. Task 2.2: Add /api/validate (20 min)

**Track B (Frontend):**
8. Task 3.1: Validation panel (45 min)
9. Task 3.2: Roofline overlay (30 min)

**Track C (Tooling):**
10. Task 4.1: Export script (15 min)
11. Task 5.1: Documentation (20 min)

**Total Time:** ~3 hours (can parallelize to 90 minutes with 2 people)

---

## SUCCESS CRITERIA

### Phase 1 Complete When:
- [ ] GEMV FP16 prediction: 203 μs ± 20 μs (actual: 203.4 μs)
- [ ] GEMV FP8 prediction: 35 μs ± 5 μs (actual: 34.8 μs)
- [ ] GEMM FP16 prediction: 14300 μs ± 1500 μs (actual: 14339 μs)
- [ ] GEMM FP8 prediction: 834 μs ± 100 μs (actual: 834 μs)
- [ ] GEMM TF32 prediction: 3680 μs ± 400 μs (actual: 3681 μs)

### Demo Ready When:
- [ ] User can paste benchmark JSON and see validation table
- [ ] Roofline plot shows both predicted and measured points
- [ ] Export script generates runnable benchmark code
- [ ] Documentation has complete validation workflow example

---

## NOTES

### What NOT to Do
- ❌ Don't rewrite core roofline math (it's correct)
- ❌ Don't add ML/curve fitting (over-engineering)
- ❌ Don't remove ideal roofline (good for education)

### What TO Do
- ✅ Add efficiency factors as optional multiplier
- ✅ Preserve "ideal" vs "realistic" modes
- ✅ Make validation visual and intuitive
- ✅ Keep code changes minimal and focused

