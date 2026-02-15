# Next Steps: Theory → Kernel → Validation

**Your path from understanding roofline to proving it works on Blackwell GB10/B10.**

---

## 🎯 What You'll Do

1. **Understand theory** - Run roofline calculator (predictions)
2. **Write a kernel** - Implement simple GEMV benchmark
3. **Measure reality** - Benchmark on GB10/B10
4. **Compare** - Does theory match reality?

---

## 📝 Step 1: Understand Roofline Theory (5 min)

Run the theoretical calculator to see predictions:

```bash
cd ~/roofline-hack
python3 src/roofline/calculator_shell.py
```

**Expected output:**
```
======================================================================
Roofline Predictions: 4096×4096 GEMV on Blackwell B10
Peak BW: 1000.0 GB/s
======================================================================

[FP16]
  AI: 1.00 FLOP/byte (critical: 200.00)
  Predicted time: 34.0 μs
  Predicted throughput: 0.494 TFLOPS
  Bottleneck: memory

[FP8]
  AI: 2.00 FLOP/byte (critical: 400.00)
  Predicted time: 17.0 μs
  Predicted throughput: 0.988 TFLOPS
  Bottleneck: memory

[NVFP4]
  AI: 3.53 FLOP/byte (critical: 800.00)
  Predicted time: 9.5 μs
  Predicted throughput: 1.77 TFLOPS
  Bottleneck: memory

Predicted Speedups vs FP16:
  FP16: 1.00×
  FP8: 2.0×  ← Theory says 2× faster
  NVFP4: 3.5×  ← Theory says 3.5× faster
  INT4: 4.0×  ← Theory says 4× faster
```

**What this tells you:**
- All precisions are memory-bound (AI << Critical AI = 200)
- Lower precision = less bandwidth = proportional speedup
- FP8 should be ~2× faster than FP16
- NVFP4 should be ~3.5× faster (Blackwell native)

---

## 🔧 Step 2: Write & Run Kernel (B10, 10 min)

Benchmark on your B10:

```bash
# Install dependencies
pip3 install torch pandas --user

# Run kernel benchmark
python3 benchmarks/kernel_shell.py
```

**Expected output:**
```
======================================================================
Benchmarking 4096×4096 GEMV on B10
======================================================================

Running FP16...
  Time: ~34 μs
  TFLOPS: ~0.5
  Bandwidth: ~500 GB/s
  AI: 1.00 FLOP/byte

Running FP8...
  Time: ~17 μs
  TFLOPS: ~1.0
  Bandwidth: ~500 GB/s
  AI: 2.00 FLOP/byte

FP8 Speedup: 2.0× (theoretical: 2.0×)
```

---

## 📊 Step 3: Compare Theory vs Reality

Run the comparison script:

```bash
python3 compare_shell.py
```

**Expected output:**
```
THEORY VS REALITY: Roofline Validation
================================================================================
Hardware: B10
Problem Size: 4096×4096 GEMV

Precision    Predicted (μs)   Measured (μs)    Error      Bottleneck
--------------------------------------------------------------------------------
FP16         34.0             TBD              TBD        memory
FP8          17.0             TBD              TBD        memory
NVFP4        9.5              TBD              TBD        memory

Target: <10% error = excellent prediction
```

**What this validates:**
- ✅ Roofline model accuracy on Blackwell
- ✅ Memory bandwidth is the bottleneck
- ✅ Lower precision delivers proportional speedup

---

## 🚀 Step 4: Explore FP4 and Advanced Formats

Blackwell B10 has native FP4 tensor cores:

**Test NVFP4:**
```python
# NVFP4 should give ~3.5× speedup vs FP16
# Uses native Blackwell FP4 tensor cores
```

**Your task:**
1. Measure NVFP4 performance
2. Compare to FP8 and FP16
3. Check accuracy impact on real models

---

## 📖 Understanding the Code

### 1. Roofline Calculator (`src/roofline/calculator_shell.py`)

```python
# Predicts performance based on hardware specs
ai = flops / bytes  # Arithmetic intensity
time_memory = bytes / bandwidth  # Memory-bound time
time_compute = flops / peak_flops  # Compute-bound time
predicted_time = max(time_memory, time_compute)  # Bottleneck
```

### 2. Benchmark Kernel (`benchmarks/kernel_shell.py`)

```python
# Measures actual GPU performance
# Uses CUDA events for precise timing
# Supports FP16, FP8, FP4, INT8, INT4
```

### 3. Comparison (`compare_shell.py`)

```python
# Compares theory vs measurement
# Calculates error percentage
# Validates roofline predictions
```

---

## 🎓 What You're Learning

### Theoretical Understanding:
- ✅ Arithmetic Intensity formula: `AI = FLOPs / Bytes`
- ✅ Roofline model: `time = max(bytes/BW, flops/peak)`
- ✅ Memory-bound regime: AI << Critical AI → BW limited
- ✅ Blackwell FP8/FP4 native support

### Practical Skills:
- ✅ Implement GPU benchmarks
- ✅ Measure with CUDA events
- ✅ Calculate FLOPS and bandwidth
- ✅ Validate theoretical models

---

## 🔬 Advanced Experiments

### 1. Different Problem Sizes
```bash
python3 compare_shell.py --M 8192 --N 4096 --K 4096
```

### 2. Prefill vs Decode
```bash
# Decode: M=1 (memory-bound)
# Prefill: M=2048+ (may be compute-bound)
```

### 3. FP4 Deep Dive
```bash
# Compare NVFP4 vs MXFP4
# Test W4A16 vs W4A4 configurations
```

---

## 🎯 Success Criteria

**You've succeeded when:**
- ✅ Roofline predictions within 10% of measurements
- ✅ FP8 delivers ~2× speedup (validated)
- ✅ NVFP4 delivers ~3.5× speedup (validated)
- ✅ You understand why (memory bandwidth bottleneck)

---

## 📚 Key Files

```
roofline-hack/
├── src/roofline/
│   ├── calculator_shell.py    # Theory: Roofline predictions
│   ├── hardware_registry.py   # B10, B200, H100, A100 specs
│   └── auto_quantize.py       # Auto-quantization recommendation
├── benchmarks/
│   ├── kernel_shell.py        # Practice: GEMV/GEMM kernels
│   └── gemm_sweep.py          # Sweep across shapes and precisions
└── compare_shell.py           # Validation: Compare theory vs reality
```

---

## 🎉 What's Next?

1. **Validate on real B10** - Measure actual performance
2. **Compare FP8 vs NVFP4** - Which is better for your workload?
3. **Full model benchmark** - Run Llama with auto-quantization
4. **Document findings** - Share your B10 validation results
