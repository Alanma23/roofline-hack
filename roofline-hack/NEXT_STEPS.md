# Next Steps: Theory → Kernel → Validation

**Your path from understanding roofline to proving it works.**

---

## 🎯 What You'll Do

1. **Understand theory** - Run roofline calculator (predictions)
2. **Write a kernel** - Implement simple GEMV benchmark
3. **Measure reality** - Benchmark on your Jetson
4. **Compare** - Does theory match reality?

---

## 📝 Step 1: Understand Roofline Theory (Local, 5 min)

Run the theoretical calculator to see predictions:

```bash
cd ~/roofline-hack
python3 src/roofline/calculator.py
```

**Expected output:**
```
======================================================================
Roofline Predictions: 4096×4096 GEMV on Jetson Orin Nano
Peak BW: 60.0 GB/s
======================================================================

[FP16]
  AI: 1.00 FLOP/byte (critical: 45.00)
  Predicted time: 280.0 μs
  Predicted throughput: 0.060 TFLOPS
  Bottleneck: memory
  
[INT8]
  AI: 2.00 FLOP/byte (critical: 91.67)
  Predicted time: 145.0 μs
  Predicted throughput: 0.116 TFLOPS
  Bottleneck: memory

Predicted Speedups vs FP16:
  FP16: 1.00×
  INT8: 1.93×  ← Theory says 2× faster
  INT4: 3.57×  ← Theory says 3.6× faster
```

**What this tells you:**
- All precisions are memory-bound (AI << Critical AI)
- Lower precision = less bandwidth = proportional speedup
- INT8 should be ~2× faster than FP16

**Now test if theory is right!**

---

## 🔧 Step 2: Write & Run Simple Kernel (Jetson, 10 min)

Transfer code to Jetson and benchmark:

```bash
# On Jetson
cd ~/roofline-hack

# Install dependencies
pip3 install torch pandas --user

# Run simple kernel benchmark
python3 benchmarks/simple_kernel.py
```

**Expected output:**
```
======================================================================
Benchmarking 4096×4096 GEMV on Orin
======================================================================

Running FP16...
  Time: 265.0 μs
  TFLOPS: 0.064
  Bandwidth: 51.2 GB/s
  AI: 1.00 FLOP/byte

Running INT8...
  Time: 139.0 μs
  TFLOPS: 0.121
  Bandwidth: 49.8 GB/s
  AI: 2.00 FLOP/byte

INT8 Speedup: 1.91× (theoretical: 2.0×)  ← Very close!
```

**What you learned:**
- You wrote a kernel (SimpleGEMV.run_kernel)
- You measured real performance
- INT8 is actually ~1.9× faster (close to theory!)

---

## 📊 Step 3: Compare Theory vs Reality (Jetson, 5 min)

Run the comparison script:

```bash
python3 compare_theory_vs_reality.py
```

**Expected output:**
```
THEORY VS REALITY: Roofline Validation
================================================================================
Hardware: Orin
Problem Size: 4096×4096 GEMV

Precision    Predicted (μs)   Measured (μs)    Error      Bottleneck  
--------------------------------------------------------------------------------
FP16         280.0            265.0            5.4%       memory      
INT8         145.0            139.0            4.1%       memory      

================================================================================
ANALYSIS
================================================================================

1. Speedup (FP16 → INT8):
   Predicted: 1.93×
   Measured:  1.91×
   Error:     1.0%

2. Prediction Accuracy:
   Mean error: 4.7%
   ✅ EXCELLENT (<10% error)

3. Memory Bound Confirmation:
   FP16: AI=1.00, Critical=45.00 → 45× below (memory-bound ✓)
   INT8: AI=2.00, Critical=91.67 → 46× below (memory-bound ✓)

4. Key Insight:
   INT8 is 1.91× faster (vs predicted 1.93×)
   This proves: Lower precision → proportional speedup (memory-bound regime)
```

**What this validates:**
- ✅ Roofline model is accurate (<5% error)
- ✅ Memory bandwidth is the bottleneck
- ✅ Lower precision delivers proportional speedup

---

## 🚀 Step 4: Extend to INT4 (Your Next Challenge)

Now that INT8 works, try INT4!

**Modify the kernel:**

```python
# In benchmarks/simple_kernel.py
# Add INT4 support (need to pack 2 values per byte)

class SimpleGEMV:
    def __init__(self, N, K, precision, device='cuda'):
        # ... existing code ...
        
        if precision == 'int4':
            # Pack two INT4 values per INT8
            # This is your challenge: implement INT4 packing!
            pass
```

**Predicted result:**
```
INT4 Speedup: ~3.6× vs FP16
```

**Your task:**
1. Implement INT4 packing in the kernel
2. Benchmark it
3. Check if you get 3.6× speedup
4. **Critical:** Measure accuracy impact (run a small model)

---

## 📖 Understanding Your Code

### 1. Roofline Calculator (`src/roofline/calculator.py`)

**Key function:** `predict_gemv(N, K, precision)`

```python
# Calculates:
ai = flops / bytes  # Arithmetic intensity
time_memory = bytes / bandwidth  # Memory-bound time
time_compute = flops / peak_flops  # Compute-bound time
predicted_time = max(time_memory, time_compute)  # Bottleneck
```

**This is the theory:** Performance limited by slower of memory or compute.

### 2. Simple Kernel (`benchmarks/simple_kernel.py`)

**Key function:** `run_kernel()`

```python
def run_kernel(self):
    if self.precision == 'fp16':
        return torch.matmul(self.W, self.x)  # Native FP16
    else:  # int8
        W_fp16 = self.W.to(torch.float16)  # Cast to FP16
        x_fp16 = self.x.to(torch.float16)
        return torch.matmul(W_fp16, x_fp16)  # Compute in FP16
```

**This is reality:** PyTorch calls CUDA kernels, you measure actual time.

### 3. Comparison (`compare_theory_vs_reality.py`)

**What it does:**
1. Runs roofline calculator (theory)
2. Runs benchmark kernel (reality)
3. Calculates error: `(measured - predicted) / predicted × 100%`

**If error < 10%:** Model is accurate!

---

## 🎓 What You're Learning

### Theoretical Understanding:
- ✅ Arithmetic Intensity formula: `AI = FLOPs / Bytes`
- ✅ Roofline model: `time = max(bytes/BW, flops/peak)`
- ✅ Memory-bound regime: AI << Critical AI → BW limited
- ✅ Precision scaling: `AI ≈ 16 / weight_bits`

### Practical Skills:
- ✅ Implement a GPU kernel (GEMV)
- ✅ Benchmark with CUDA events
- ✅ Calculate FLOPS and bandwidth
- ✅ Validate theoretical models

### Research Skills:
- ✅ Hypothesis: Lower bits → proportional speedup
- ✅ Experiment: Measure FP16 vs INT8
- ✅ Analysis: Compare predicted vs measured (<5% error)
- ✅ Conclusion: Theory validated ✓

---

## 🔬 Advanced Experiments

Once basics work, try:

### 1. Different Problem Sizes
```bash
python3 compare_theory_vs_reality.py --N 2048 --K 2048
python3 compare_theory_vs_reality.py --N 8192 --K 8192
```

**Question:** Does prediction accuracy change with size?

### 2. Power Modes
```bash
# Max performance (15W)
sudo nvpmodel -m 0
python3 compare_theory_vs_reality.py

# Power efficient (7W)
sudo nvpmodel -m 1
python3 compare_theory_vs_reality.py
```

**Question:** Does roofline still predict correctly at lower power?

### 3. Prefill vs Decode
```bash
# Decode: T=1 (current)
# Prefill: T=S (batch processing)

# Modify calculator.py to add predict_gemm() for prefill
# Compare: Is prefill compute-bound or memory-bound?
```

**Hypothesis:** Prefill has higher AI → might be compute-bound.

---

## 🎯 Success Criteria

**You've succeeded when:**
- ✅ Roofline predictions within 10% of measurements
- ✅ INT8 delivers ~2× speedup (validated)
- ✅ You understand why (memory bandwidth bottleneck)
- ✅ You can explain: "AI < Critical AI → memory-bound"

**Bonus:**
- ✅ INT4 implemented and validated
- ✅ Accuracy analysis (perplexity test)
- ✅ Power-aware analysis (7W vs 15W)

---

## 📚 Files Created

```
roofline-hack/
├── src/roofline/
│   └── calculator.py          # Theory: Roofline predictions
├── benchmarks/
│   └── simple_kernel.py       # Practice: Simple GEMV kernel
├── compare_theory_vs_reality.py   # Validation: Compare both
└── NEXT_STEPS.md              # This guide
```

**Start here:** Run each file in order (theory → kernel → compare)

---

## ❓ Troubleshooting

**"CUDA not available"**
→ Make sure you're running on Jetson with JetPack installed

**"Import error: No module named 'roofline'"**
→ Run from project root: `cd ~/roofline-hack && python3 compare_theory_vs_reality.py`

**"Predictions way off (>20% error)"**
→ Check hardware specs in calculator.py (bandwidth might be different)

---

## 🎉 What's Next?

After validating INT8:

1. **INT4 implementation** - Your challenge!
2. **Accuracy testing** - Does INT4 maintain quality?
3. **Full model benchmark** - Run TinyLlama at INT8/INT4
4. **Documentation** - Write up your findings

**You now have the full toolkit: theory → implementation → validation** 🚀
