# Precision Format Catalog

**Understanding lower-bit precision formats and their hardware performance implications.**

## Why Lower Precision Matters

**Memory bandwidth bottleneck:** Modern ML inference is memory-bound, not compute-bound.

**The tradeoff:**
- ✅ **Lower bits = faster:** 16→8→4 bits = 2×→4× less data = proportional speedup
- ⚠️ **Lower bits = less accurate:** Quantization error increases with fewer bits
- 🎯 **Question:** Where's the sweet spot? (This project investigates INT8 vs INT4)

**This catalog** shows all format options from FP32 (highest quality, slowest) down to INT2 (lowest quality, fastest), with focus on formats actually deployed in hardware (FP16, INT8, INT4).

## Lower Precision Performance Impact

**Key insight:** In memory-bound regime, performance scales inversely with bits.

| Format | Bits | Bytes/elem | Relative BW | Predicted Speedup | Accuracy | Hardware Support |
|--------|------|------------|-------------|-------------------|----------|------------------|
| **FP32** | 32 | 4.0 | 1.0× | Baseline | Perfect | Universal |
| **FP16** | 16 | 2.0 | 2.0× | 2× | Minimal loss | H100/B10 |
| **FP8** | 8 | 1.0 | 4.0× | 4× | <1% loss | H100+, B10 native |
| **NVFP4** | ~4.5 | ~0.57 | ~7.0× | ~7× | ~1-2% loss | B10/B200 native |
| **INT4** | 4 | 0.5 | 8.0× | 8× | ~2-5% loss | B10 supported |
| **INT2** | 2 | 0.25 | 16.0× | 16× | Severe | ⚠️ Research only |

**Hardware Reality (Blackwell B10):**
- FP16: ~34 μs (baseline)
- FP8: ~17 μs (2.0× predicted)
- NVFP4: ~9.5 μs (3.5× predicted)
- INT4: ~8.4 μs (4.0× predicted)

→ **Blackwell native FP8/FP4 support for optimal performance.**

---

## Format Taxonomy

### Scalar Formats (1 exponent per element)

| Format | Bits | Layout | Range | Levels | Bytes/elem |
|--------|------|--------|-------|--------|------------|
| FP32 | 32 | E8M23 | ±3.4e38 | ~16M | 4.000 |
| TF32 | 19 | E8M10 | ±3.4e38 | ~1K | 2.375 |
| BF16 | 16 | E8M7 | ±3.4e38 | 127 | 2.000 |
| FP16 | 16 | E5M10 | ±65504 | ~1K | 2.000 |
| FP8 E4M3 | 8 | E4M3 | ±448 | 29 | 1.000 |
| FP8 E5M2 | 8 | E5M2 | ±57344 | 15 | 1.000 |
| INT8 | 8 | — | ±127 | 127 | 1.000 |
| INT4 | 4 | — | ±7 | 7 | 0.500 |
| INT2 | 2 | — | ±1 | 1 | 0.250 |

### Block Floating Point — OCP MX Formats (block=32, E8M0 scale)

| Format | Element | Block | Scale | Eff bits/elem | Overhead |
|--------|---------|-------|-------|----------------|----------|
| MXFP8 E4M3 | 8b E4M3 | 32 | E8M0 (8b) | 8.250 | 3.0% |
| MXFP8 E5M2 | 8b E5M2 | 32 | E8M0 (8b) | 8.250 | 3.0% |
| MXFP6 E3M2 | 6b E3M2 | 32 | E8M0 (8b) | 6.250 | 4.0% |
| MXFP6 E2M3 | 6b E2M3 | 32 | E8M0 (8b) | 6.250 | 4.0% |
| MXFP4 | 4b E2M1 | 32 | E8M0 (8b) | 4.250 | 5.9% |
| MXINT8 | 8b INT | 32 | E8M0 (8b) | 8.250 | 3.0% |

E8M0 scale type: unsigned, encodes powers of 2 from 2^(-127) to 2^(127). Only power-of-two scaling — no fractional adjustment.

### NVIDIA NVFP4 (block=16, E4M3 scale + FP32 tensor)

| Format | Element | Block | Scale | Eff bits/elem | Overhead |
|--------|---------|-------|-------|----------------|----------|
| NVFP4 | 4b E2M1 | 16 | E4M3 (8b) + FP32 tensor | 4.531 | 11.7% |

Key differences from MXFP4:
- **Smaller block (16 vs 32)**: 2× more scale factors → finer-grained adaptation
- **E4M3 scale (vs E8M0)**: non-power-of-two values → smoother scaling
- **Two-level scaling**: per-block E4M3 + per-tensor FP32 → enormous effective range
- **Higher byte overhead**: 4.531 vs 4.250 bits/elem (+6.6% more storage)
- **Better accuracy**: the extra overhead buys measurable quality (<1% degradation from FP8)

### Lookup Table — NF4 (bitsandbytes)

| Format | Element | Block | Scale | Eff bits/elem | Overhead |
|--------|---------|-------|-------|----------------|----------|
| NF4 | 4b index | 64 | FP16 absmax (16b) | 4.250 | 5.9% |

NF4 uses a non-uniform 16-entry lookup table optimized for normally-distributed weights. The codebook maps 4-bit indices to values in [-1, 1], then scales by a per-block FP16 absmax factor.

## The FP4 E2M1 Codebook

Shared by both MXFP4 and NVFP4. 8 representable positive values (+ negatives + zero = 16 total):

```
Value:    0    0.5    1.0    1.5    2.0    3.0    4.0    6.0
Step:     —    0.5    0.5    0.5    0.5    1.0    1.0    2.0
```

Step sizes are **non-uniform**: fine near zero (0.5 spacing), coarse near maximum (2.0 spacing). This means values between 4 and 6 can only be represented as exactly 4 or 6 — a 50% gap.

### The Four Over Six (4/6) Algorithm

Addresses the coarse quantization near FP4's maximum. For some blocks, NVFP4 scales to max=4 instead of max=6:
- Standard scaling (max=6): steps are 0.5, 0.5, 0.5, 0.5, 1, 1, 2
- 4/6 scaling (max=4): steps are 0.33, 0.33, 0.33, 0.33, 0.67, 0.67, 1.33

The algorithm picks whichever scaling yields lower MSE per block. Trades dynamic range for resolution at boundaries.

## NVFP4 vs MXFP4: Roofline Impact

The byte overhead difference (4.531 vs 4.250 bits/elem) has two effects on the roofline:

1. **Arithmetic intensity**: MXFP4 has ~6.6% higher AI for weight-loading operations (less overhead per element)
2. **Accuracy**: NVFP4's finer-grained scaling yields <1% accuracy loss vs FP8, while MXFP4 may require more careful calibration

For a decode GEMV with K=4096:
- MXFP4 weights: AI = 2×4096 / (4096×0.53125 + 2 + 2) ≈ 3.76 FLOP/byte
- NVFP4 weights: AI = 2×4096 / (4096×0.56625 + 2 + 2) ≈ 3.53 FLOP/byte

This ~6% AI difference translates to ~6% throughput difference in the memory-bound regime — meaningful at scale, but accuracy often favors NVFP4.

## Hardware Support Matrix

| Format | A100 | H100 | B10 | B200 | B300 |
|--------|------|------|-----|------|------|
| FP32 | ✓ CUDA | ✓ CUDA | ✓ CUDA | ✓ CUDA | ✓ CUDA |
| FP16 | ✓ TC | ✓ TC | ✓ TC | ✓ TC | ✓ TC |
| BF16 | ✓ TC | ✓ TC | ✓ TC | ✓ TC | ✓ TC |
| FP8 | ✗ | ✓ TC | ✓ TC | ✓ TC | ✓ TC |
| MXFP4 | ✗ | ✗ | ✓ TC | ✓ TC | ✓ TC |
| NVFP4 | ✗ | ✗ | ✓ TC | ✓ TC | ✓ TC |
| INT8 | ✓ TC | ✓ TC | ✓ TC | ✓ TC | ✓ TC |

"TC" = native tensor core support (compute at that precision's peak).
Formats without TC support can still be used for storage (W4A16 dequant pattern) — the calculator models this correctly via the `compute_as` field.
