# Project Structure

**Clean, focused: Theory + Validation**

---

## 📂 Files (Simplified)

```
roofline-hack/
│
├── README.md                         ⭐ Project overview
├── GUIDE.md                          📖 Complete implementation guide  
├── NEXT_STEPS.md                     📚 Theory reference
├── STRUCTURE.md                      📋 This file
│
├── src/roofline/
│   └── calculator_shell.py           🔢 YOUR CODE: Roofline formulas
│
├── benchmarks/
│   ├── kernel_shell.py               ⚡ YOUR CODE: Benchmark kernels
│   └── jetson/
│       ├── validate_jetson.py        ✅ Reference implementation
│       ├── setup_jetson.sh           🔧 Jetson setup
│       └── README.md
│
├── compare_shell.py                  🎯 YOUR CODE: Validate predictions
│
├── docs/                             📖 Theory (3 files)
│   ├── THEORY_FORMATS.md             → Precision formats
│   ├── THEORY_MATH.md                → Operator math
│   └── JETSON_VALIDATION.md          → Validation guide
│
└── frontend/
    └── roofline-calc-v2.jsx          🎨 Interactive visualizer (reference)
```

**Total:** 14 files (3 to implement, rest are reference/docs)

---

## 🎯 Your Implementation Tasks

### 1. Calculator (`src/roofline/calculator_shell.py`)

**What:** Roofline formulas to predict performance

**Implement:**
- `bytes_per_element(precision)` - FP16=2, INT8=1, INT4=0.5
- `HardwareSpec.critical_ai()` - Peak FLOPS / Bandwidth
- `RooflineCalculator.predict_gemv()` - FLOPs, Bytes, AI, time
- `RooflineCalculator.predict_gemm()` - (optional)

**Time:** 30-45 min

**Formulas provided in:** GUIDE.md, frontend code, docs/THEORY_MATH.md

---

### 2. Benchmarks (`benchmarks/kernel_shell.py`)

**What:** Actual kernel execution + timing

**Implement:**
- `get_torch_dtype()` - Map precision to torch dtype
- `GEMVKernel.__init__()` - Create random data
- `GEMVKernel.run()` - Execute matmul
- `GEMVKernel.benchmark()` - CUDA events timing

**Time:** 30-45 min

**Reference:** `benchmarks/jetson/validate_jetson.py` (working example)

---

### 3. Validation (`compare_shell.py`)

**What:** Compare theory vs reality

**Implement:**
- `compare_gemv()` - Run both, calculate error
- `precision_sweep()` - Test FP16/INT8, show speedups

**Time:** 15-30 min

**Goal:** Prove roofline model is accurate (<15% error)

---

## 📊 Data Flow

```
Theory (Calculator)
    ↓
Predict: "INT8 should be 145 μs (memory-bound)"
    ↓
Reality (Benchmark)
    ↓
Measure: "INT8 is actually 139 μs"
    ↓
Validation (Compare)
    ↓
Error: 4.1% → Model is accurate! ✓
```

---

## 🔬 Investigation Focus

**Question:** How far can we push quantization?

- **FP16 → INT8:** 2× speedup (proven on Jetson)
- **FP16 → INT4:** 4× speedup (predicted, needs custom kernel)
- **Tradeoff:** Speed vs accuracy

**This is ML systems co-design!**

---

## 📚 Documentation

| File | Purpose |
|------|---------|
| `GUIDE.md` | Implementation guide (formulas, steps, testing) |
| `NEXT_STEPS.md` | Theory reference (roofline model, derivations) |
| `docs/THEORY_FORMATS.md` | Precision catalog (15+ formats) |
| `docs/THEORY_MATH.md` | Operator math (GEMV, GEMM, attention) |
| `docs/JETSON_VALIDATION.md` | Jetson-specific guide |
| `frontend/roofline-calc-v2.jsx` | Full reference implementation |

---

## ⚡ Quick Commands

```bash
# Test calculator
python src/roofline/calculator_shell.py

# Test benchmarks (requires CUDA)
python benchmarks/kernel_shell.py

# Validate (compare predictions vs measurements)
python compare_shell.py

# Jetson setup
cd benchmarks/jetson
bash setup_jetson.sh
python validate_jetson.py
```

---

**Key principle:** Simple structure. Clear separation. You implement theory + benchmarks. Docs provide formulas. Frontend is reference.
