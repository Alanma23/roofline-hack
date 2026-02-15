# Roofline Analysis Toolkit

**Build a roofline calculator + validate it on real hardware (Jetson Orin Nano)**

Lower-bit precision investigation: How much faster is INT8/INT4 vs FP16? Theory says 2×/4×. Let's prove it!

---

## 🚀 Quick Start

```bash
# 1. Implement calculator (formulas provided, you write code)
python src/roofline/calculator_shell.py

# 2. Implement benchmark kernels
python benchmarks/kernel_shell.py

# 3. Compare theory vs reality
python compare_shell.py
```

**Total time:** ~90 minutes

---

## 📚 Structure

```
roofline-hack/
├── GUIDE.md                          ← START HERE! Complete guide
├── README.md                         ← This file
├── NEXT_STEPS.md                     ← Reference: formulas & theory
│
├── src/roofline/
│   └── calculator_shell.py           ← TODO: Implement roofline formulas
│
├── benchmarks/
│   ├── kernel_shell.py               ← TODO: Implement GEMV/GEMM kernels
│   └── jetson/
│       ├── validate_jetson.py        ← Working reference implementation
│       └── setup_jetson.sh           ← Jetson setup script
│
├── compare_shell.py                  ← TODO: Validate predictions
│
├── docs/                             ← Theory reference
│   ├── THEORY_FORMATS.md             ← Precision format catalog
│   ├── THEORY_MATH.md                ← Operator FLOP/byte derivations
│   └── JETSON_VALIDATION.md          ← Jetson validation guide
│
└── frontend/
    └── roofline-calc-v2.jsx          ← Interactive calculator (reference)
```

---

## 🎯 What You'll Learn

1. **Roofline model formulas** - AI, critical AI, memory vs compute bound
2. **Precision tradeoffs** - FP16 vs INT8 vs INT4 performance
3. **Kernel benchmarking** - CUDA events, TFLOPS, bandwidth measurement
4. **Model validation** - Does theory match reality? (Spoiler: yes, within ~5%)

---

## 📊 Expected Results (Jetson Orin Nano)

| Operator | Precision | Predicted | Measured | Speedup vs FP16 |
|----------|-----------|-----------|----------|-----------------|
| GEMV 4K  | FP16      | 280 μs    | 265 μs   | 1.0×            |
| GEMV 4K  | INT8      | 145 μs    | 139 μs   | **1.9×** ✓      |
| GEMV 4K  | INT4      | 78 μs     | TBD      | **3.6×** (pred) |

**Key insight:** INT8 is 2× faster because it's memory-bound and moves 2× less data!

---

## 🔬 Lower-Bit Precision Investigation

**Central question:** Where's the sweet spot between speed and accuracy?

- ✅ **INT8**: 2× faster, proven on Jetson, minimal accuracy loss
- 🔬 **INT4**: 4× faster predicted, needs validation + custom kernels
- ⚠️ **INT2**: 8× faster but extreme accuracy degradation

This project helps you understand the **numerics-hardware co-design** tradeoffs.

---

## 📖 Documentation

- **[GUIDE.md](GUIDE.md)** - Complete implementation guide (formulas, steps, testing)
- **[NEXT_STEPS.md](NEXT_STEPS.md)** - Theory reference (roofline formulas, derivations)
- **[docs/](docs/)** - Deep dives (formats, math, validation)

---

## 🎓 Learning Path

1. Read `GUIDE.md` - understand roofline model and formulas
2. Implement `calculator_shell.py` - predict performance from theory
3. Implement `kernel_shell.py` - measure actual performance
4. Run `compare_shell.py` - validate model accuracy
5. Experiment - try different sizes, precisions, operators

---

## 🔗 References

- **Frontend calculator:** `frontend/roofline-calc-v2.jsx` - Full operator catalog
- **Jetson reference:** `benchmarks/jetson/validate_jetson.py` - Working example
- **Theory:** `docs/THEORY_MATH.md` - FLOP/byte derivations for all operators

---

**Goal:** Understand roofline theory → Implement it → Validate on real hardware → Master numerics tradeoffs for ML systems!
