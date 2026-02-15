# Roofline Analysis - Implementation Guide

**Goal:** Build roofline calculator + benchmark kernels, validate predictions on Blackwell GB10/B10

---

## 🎯 What You'll Build

1. **Calculator** (`src/roofline/calculator_shell.py`) - Predict performance from theory
2. **Benchmarks** (`benchmarks/kernel_shell.py`) - Measure actual performance  
3. **Validation** (`compare_shell.py`) - Compare theory vs reality

---

## 📐 Core Roofline Formula

```
Performance = min(Peak_FLOPS, Arithmetic_Intensity × Bandwidth)

Where:
- Arithmetic_Intensity (AI) = FLOPs / Bytes
- Critical_AI = Peak_FLOPS / Bandwidth
- If AI < Critical_AI → memory-bound
- If AI > Critical_AI → compute-bound
```

---

## 🔢 Operator Math (from frontend)

### GEMV: y[N] = W[N,K] @ x[K]

```
FLOPs = 2*N*K

Bytes = K*bytes_per_elem(input)      // x[K]
      + N*K*bytes_per_elem(weights)  // W[N,K]  
      + N*bytes_per_elem(output)     // y[N]

AI = FLOPs / Bytes
```

### GEMM: C[M,N] = A[M,K] @ B[K,N]

```
FLOPs = 2*M*N*K

Bytes = M*K*bytes_per_elem(A)
      + K*N*bytes_per_elem(B)
      + M*N*bytes_per_elem(C)

AI = FLOPs / Bytes
```

### Attention: softmax(Q @ K^T) @ V

```
# Two matmuls
QK_flops = 2*B*nh*Sq*Skv*dh
QKV_flops = 2*B*nh*Sq*Skv*dh
Total_flops = QK_flops + QKV_flops

# Simplified bytes (Q, K, V, output)
Bytes = 4 * B*nh*S*dh * bytes_per_elem

AI = Total_flops / Bytes
```

---

## 🎨 Precision Formats (from frontend)

```python
# Bytes per element including quantization overhead
FP32:  4.0 bytes
FP16:  2.0 bytes
INT8:  1.0 bytes
INT4:  0.5 bytes

# Block formats have overhead
MXFP4:  4 bits + (8-bit scale / 32 elements) = 4.25 bits
NVFP4:  4 bits + (8-bit scale / 16) + (32-bit tensor scale / 1024) = 4.53 bits
NF4:    4 bits + (16-bit scale / 64) = 4.25 bits
```

---

## 🚀 Implementation Steps

### 1. Implement Calculator (30-45 min)

File: `src/roofline/calculator_shell.py`

**TODO for you:**
- Define hardware specs (B10: 1000 GB/s, 200 TFLOPS FP16)
- Implement `predict_gemv()` using formulas above
- Calculate: FLOPs, Bytes, AI, time_memory, time_compute
- Determine bottleneck (which time is larger?)

### 2. Implement Benchmark Kernel (30-45 min)

File: `benchmarks/kernel_shell.py`

**TODO for you:**
- Create data: `torch.randn()` for FP16, `torch.randint()` for INT8
- Implement kernel: `torch.matmul(W, x)`
- Add timing: CUDA events (`torch.cuda.Event()`)
- Calculate metrics: time, TFLOPS, bandwidth

### 3. Compare & Validate (15-30 min)

File: `compare_shell.py`

**TODO for you:**
- Run calculator to get prediction
- Run benchmark to get measurement
- Calculate error: `|predicted - measured| / predicted * 100%`
- Analyze: Does theory match reality?

---

## 📊 Expected Results (Blackwell B10)

| Operator | Precision | Predicted | Measured | Error |
|----------|-----------|-----------|----------|-------|
| GEMV 4K  | FP16      | ~34 μs    | TBD      | TBD   |
| GEMV 4K  | FP8       | ~17 μs    | TBD      | TBD   |
| GEMV 4K  | NVFP4     | ~9.5 μs   | TBD      | TBD   |

**Speedup:** FP8 should be ~2.0× faster than FP16, NVFP4 ~3.5× faster

---

## 🔍 Understanding Your Results

**Why memory-bound?**
- GEMV AI = 1.0 (FP16) or 2.0 (FP8)
- Critical AI = 200 (B10 FP16)
- AI << Critical AI → bandwidth limits performance

**Why does quantization help?**
- Less data to move → less time waiting on memory
- FP16→FP8: 2× less data → 2× faster
- FP16→NVFP4: ~3.5× less data → ~3.5× faster
- FP16→INT4: 4× less data → 4× faster

---

## 📚 Reference

- **Theory formulas:** `docs/THEORY_MATH.md`
- **Precision catalog:** `docs/THEORY_FORMATS.md`
- **Frontend reference:** `frontend/roofline-calc-v2.jsx` (lines 148-208)

---

**Total time:** ~90 minutes to implement + validate roofline model on real hardware!
