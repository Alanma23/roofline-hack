# Roofline Theory & Precision Formats

Deep dive into the roofline performance model and precision format catalog for GB10.

## Table of Contents

1. [Roofline Model Fundamentals](#roofline-model-fundamentals)
2. [Precision Format Catalog](#precision-format-catalog)
3. [Per-Operator Math](#per-operator-math)
4. [Hardware Support Matrix](#hardware-support-matrix)

---

## Roofline Model Fundamentals

### The Core Formula

```
time = max(time_memory, time_compute)

where:
  time_memory  = Bytes / Bandwidth
  time_compute = FLOPs / Peak_FLOPS
```

**Arithmetic Intensity (AI)** determines which limit applies:

```
AI = FLOPs / Bytes

if AI < Critical_AI:  memory-bound (bandwidth limits)
if AI > Critical_AI:  compute-bound (FLOPS limits)

Critical_AI = Peak_FLOPS / Bandwidth
```

### GB10 Critical AI Values

For NVIDIA GB10 (287 GB/s bandwidth):

| Precision | Peak TFLOPS | Critical AI (FLOP/byte) |
|-----------|-------------|-------------------------|
| FP32 | 31 | 108 |
| FP16 | 62 | 216 |
| FP8 | 124 | 432 |
| INT8 | 124 | 432 |
| INT4 | 248 | 864 |
| NVFP4 | 1000 | 3484 |

**Key insight:** Most transformer operations have AI < 10, making them heavily memory-bound on GB10.

### GEMV (Matrix-Vector, Decode)

```
y[N] = W[N,K] @ x[K]

FLOPs = 2 × N × K
Bytes = K × bpe + N × K × bpe + N × 2
      = (K + N×K + N) × bpe     (assuming same precision)
      ≈ N × K × bpe              (for large N, K)

AI = 2 × N × K / (N × K × bpe) ≈ 2 / bpe

For FP16 (bpe=2):  AI ≈ 1 FLOP/byte  → memory-bound
For FP8 (bpe=1):   AI ≈ 2 FLOP/byte  → still memory-bound
For NVFP4 (bpe=0.5): AI ≈ 4 FLOP/byte → still memory-bound
```

**Speedup from lower precision = bytes reduction ratio** (in memory-bound regime).

### GEMM (Matrix-Matrix, Prefill)

```
C[M,N] = A[M,K] @ B[K,N]

FLOPs = 2 × M × N × K
Bytes = M × K × bpe + K × N × bpe + M × N × bpe

AI = 2 × M × N × K / ((M×K + K×N + M×N) × bpe)
   = 2 × M × N × K / (K×(M+N) + M×N) / bpe

For M = N = K (square):  AI = 2K / (3×bpe)
For K=4096, FP16:       AI = 1365 FLOP/byte → compute-bound!
```

Large batch prefill becomes compute-bound, so lower precision helps less.

---

## Precision Format Catalog

### Why Lower Precision Matters

**The tradeoff:**
- ✅ Lower bits = faster (proportional to bytes reduction)
- ⚠️ Lower bits = less accurate (quantization error increases)
- 🎯 Question: Where's the sweet spot?

### Format Comparison

| Format | Bits | Bytes/elem | Predicted Speedup | Accuracy Loss | GB10 Support |
|--------|------|------------|-------------------|---------------|--------------|
| FP32 | 32 | 4.0 | 1.0× (baseline) | None | ✓ |
| FP16 | 16 | 2.0 | 2.0× | Minimal | ✓ |
| BF16 | 16 | 2.0 | 2.0× | Minimal | ✓ |
| FP8 E4M3 | 8 | 1.0 | 4.0× | <1% | ✓✓ (native TC) |
| FP8 E5M2 | 8 | 1.0 | 4.0× | <1% | ✓✓ (native TC) |
| INT8 | 8 | 1.0 | 4.0× | 1-2% (w/ PTQ) | ✓ |
| INT4 | 4 | 0.5 | 8.0× | 2-5% | ✓ |
| NVFP4 | ~4.5 | ~0.56 | ~7.0× | 1-2% | ✓✓ (1000 TFLOPS!) |
| MXFP4 | ~4.25 | ~0.53 | ~7.5× | 2-3% | ✓ |

**✓✓** = native tensor core support with exceptional performance

### Scalar Floating Point Formats

| Format | Bits | Layout (E/M) | Range | Bytes/elem |
|--------|------|--------------|-------|------------|
| FP64 | 64 | E11M52 | ±1.8e308 | 8.000 |
| FP32 | 32 | E8M23 | ±3.4e38 | 4.000 |
| TF32 | 19* | E8M10 | ±3.4e38 | 2.375 |
| BF16 | 16 | E8M7 | ±3.4e38 | 2.000 |
| FP16 | 16 | E5M10 | ±65504 | 2.000 |
| FP8 E4M3 | 8 | E4M3 | ±448 | 1.000 |
| FP8 E5M2 | 8 | E5M2 | ±57344 | 1.000 |

*TF32 uses 32-bit container but only 19 significant bits

### Block Floating Point - OCP MX Formats

Block size = 32 elements, E8M0 scale (power-of-two only)

| Format | Element Bits | Scale Bits | Eff bits/elem | Overhead |
|--------|--------------|------------|---------------|----------|
| MXFP8 E4M3 | 8 | 8 (shared) | 8.250 | 3.0% |
| MXFP8 E5M2 | 8 | 8 (shared) | 8.250 | 3.0% |
| MXFP6 E3M2 | 6 | 8 (shared) | 6.250 | 4.0% |
| MXFP6 E2M3 | 6 | 8 (shared) | 6.250 | 4.0% |
| MXFP4 | 4 | 8 (shared) | 4.250 | 5.9% |
| MXINT8 | 8 | 8 (shared) | 8.250 | 3.0% |

### NVIDIA NVFP4 Format

Block size = 16 elements (2× more scales than MXFP4)

| Component | Bits | Description |
|-----------|------|-------------|
| Element | 4 | E2M1 (8 positive values + negatives + zero) |
| Per-block scale | 8 | E4M3 (non-power-of-two, smoother) |
| Per-tensor scale | 32 | FP32 (huge dynamic range) |
| **Effective bits/elem** | **4.531** | **(11.7% overhead)** |

**Key advantages over MXFP4:**
- Smaller blocks (16 vs 32) = finer-grained adaptation
- E4M3 scales (vs E8M0) = non-power-of-two values
- Two-level scaling = enormous effective range
- Better accuracy (<1% loss vs FP8) at cost of 6.6% more storage

### FP4 E2M1 Codebook

8 representable positive values (sign bit doubles to 16 total):

```
Value: 0.0  0.5  1.0  1.5  2.0  3.0  4.0  6.0
Step:  ---  0.5  0.5  0.5  0.5  1.0  1.0  2.0
```

**Non-uniform spacing:** Fine near zero (0.5), coarse near max (2.0).

The "4/6 algorithm" chooses between max=6 (standard) or max=4 scaling per block to minimize MSE.

### NF4 (bitsandbytes Lookup Table)

Block size = 64 elements, FP16 absmax scale

| Component | Bits | Description |
|-----------|------|-------------|
| Element | 4 | Index into 16-entry lookup table |
| Per-block scale | 16 | FP16 absmax |
| **Effective bits/elem** | **4.250** | **(5.9% overhead)** |

Non-uniform codebook optimized for normally-distributed weights.

### Integer Formats

| Format | Bits | Range | Bytes/elem | Notes |
|--------|------|-------|------------|-------|
| INT8 | 8 | ±127 | 1.000 | Uniform quantization, per-tensor or per-channel scales |
| INT4 | 4 | ±7 | 0.500 | Often used with FP16 dequant (W4A16) |
| INT2 | 2 | ±1 | 0.250 | Research only, severe accuracy loss |

---

## Per-Operator Math

Exact FLOP and byte counts for transformer operations.

### Notation

| Symbol | Meaning |
|--------|---------|
| B | Batch size |
| S | Total sequence length (KV cache size) |
| T | Tokens being processed (= S for prefill, = 1 for decode) |
| H | Hidden dimension (d_model) |
| n_h | Number of query heads |
| n_kv | Number of KV heads (< n_h for GQA) |
| d_h | Head dimension (= H / n_h) |
| d_kv | Total KV dimension (= n_kv × d_h) |
| d_ff | FFN intermediate dimension |
| w_B | Bytes per weight element |
| a_B | Bytes per activation element |
| kv_B | Bytes per KV cache element |

### Attention Linear Projections (Q, K, V, O)

Standard GEMM: C[M,N] = A[M,K] × W[K,N]

| Projection | M | N | K | FLOPs | Bytes |
|------------|---|---|---|-------|-------|
| Q | B×T | H | H | 2·B·T·H² | B·T·H·a_B + H²·w_B + B·T·H·a_B |
| K | B×T | d_kv | H | 2·B·T·H·d_kv | B·T·H·a_B + H·d_kv·w_B + B·T·d_kv·a_B |
| V | B×T | d_kv | H | 2·B·T·H·d_kv | Same as K |
| O | B×T | H | H | 2·B·T·H² | B·T·H·a_B + H²·w_B + B·T·H·a_B |

**Decode AI (T=1):**
```
AI ≈ 2·H² / (H²·w_B + H·a_B + H·a_B) ≈ 2 / w_B

For FP16 weights: AI ≈ 1 FLOP/byte → memory-bound
```

### Attention Score Computation (QK^T)

Batched matmul: Score[B,n_h,T,S] = Q[B,n_h,T,d_h] × K^T[B,n_kv,d_h,S]

**FLOPs:** 2 · B · n_h · T · S · d_h

**Bytes:**
- Read Q: B · n_h · T · d_h · a_B
- Read K from KV cache: B · n_kv · S · d_h · kv_B
- Write scores: B · n_h · T · S · a_B

**Prefill AI (T=S):** ≈ 2·d_h / (a_B + kv_B·(n_kv/n_h) + a_B)
- For d_h=128, FP16: AI ≈ 42.7 FLOP/byte → **compute-bound**

**Decode AI (T=1):** For large S, KV read dominates
- AI → 2·d_h / (d_h·kv_B·(n_kv/n_h) + a_B)
- This is the **KV cache wall** - bytes grow linearly with context length

### Score × V

Batched matmul: Out[B,n_h,T,d_h] = Score[B,n_h,T,S] × V[B,n_kv,S,d_h]

Similar structure to QK^T. V read from KV cache at kv_B precision.

### Softmax

**FLOPs:** ~5 · B · n_h · T · S
**Bytes:** 2 · B · n_h · T · S · a_B
**AI:** ≈ 2.5 → always memory-bound

### FFN Block (SwiGLU / LLaMA-style)

Three projections: gate, up, down

| Projection | M | N | K | FLOPs | Weight Bytes |
|------------|---|---|---|-------|--------------|
| Gate | B×T | d_ff | H | 2·B·T·H·d_ff | H·d_ff·w_B |
| Up | B×T | d_ff | H | 2·B·T·H·d_ff | H·d_ff·w_B |
| Down | B×T | H | d_ff | 2·B·T·H·d_ff | H·d_ff·w_B |

**SiLU + elementwise multiply:**
- FLOPs: 3 · B · T · d_ff
- Bytes: 3 · B · T · d_ff · a_B

**Total FFN FLOPs:** 6 · B · T · H · d_ff

For LLaMA (d_ff ≈ 8/3 · H): FFN ≈ 16 · B · T · H²

**FFN is ~2/3 of total layer compute** in LLaMA models.

### Elementwise Operations

| Op | FLOPs | Bytes | AI |
|----|-------|-------|----|
| RMSNorm | 5·B·T·H | 2·B·T·H·a_B | ~2.5 |
| Residual add | B·T·H | 3·B·T·H·a_B | ~0.33 |
| RoPE | 6·B·T·H | 2·B·T·H·a_B | ~3 |

All have AI < 5 → always memory-bound. Prime candidates for operator fusion.

### Full Model Totals

**Total FLOPs per forward pass:**
```
Prefill: L × (8·B·S·H² + 4·B·n_h·S²·d_h + 6·B·S·H·d_ff) + 2·B·S·H·V
Decode:  L × (8·B·H² + 4·B·n_h·S·d_h + 6·B·H·d_ff) + 2·B·H·V
```

**Total weight bytes:**
```
L × (4·H² + 2·H·d_kv + 3·H·d_ff) × w_B + H·V·w_B
```

**KV cache bytes per decode step:**
```
L × 2 × n_kv × S × d_h × kv_B
```

### KV Cache Crossover Point

Context length where KV cache reading overtakes weight loading:

```
S_cross = [L × (4H² + 2H·d_kv + 3H·d_ff) × w_B + H·V·w_B] / [L × 2 × n_kv × d_h × kv_B]
```

| Model | FP16 W + FP16 KV | FP8 W + FP8 KV | FP8 W + NVFP4 KV |
|-------|------------------|----------------|-------------------|
| Llama-3 8B | ~1600 tokens | ~4096 tokens | ~7300 tokens |
| Llama-2 70B | ~12800 tokens | ~32000 tokens | ~57000 tokens |

**For long-context generation (10K+ tokens), KV cache quantization becomes critical.**

---

## Hardware Support Matrix

### GB10 Grace Blackwell (Primary Target)

| Format | Peak TFLOPS | Native TC | Use Case |
|--------|-------------|-----------|----------|
| FP32 | 31 | ✓ | Baseline, not recommended |
| FP16/BF16 | 62 | ✓ | Standard inference |
| FP8 E4M3/E5M2 | 124 | ✓ | 2× speedup, minimal loss |
| INT8 | 124 | ✓ | 2× speedup, PTQ required |
| INT4 | 248 | ✓ | 4× speedup, more loss |
| NVFP4 | **1000** | ✓✓ | 4× speedup, Blackwell optimized! |
| MXFP4 | 1000 | ✓ | Similar to NVFP4 |

### Comparison: Other GPUs

| Format | A100 | H100 | B10 | B200 |
|--------|------|------|-----|------|
| FP32 | ✓ | ✓ | ✓ | ✓ |
| FP16/BF16 | ✓ TC | ✓ TC | ✓ TC | ✓ TC |
| FP8 | ✗ | ✓ TC | ✓ TC | ✓ TC |
| NVFP4 | ✗ | ✗ | ✓ TC | ✓ TC |
| INT8 | ✓ TC | ✓ TC | ✓ TC | ✓ TC |

**TC** = Tensor Core support

---

## Key Takeaways

1. **Memory-bound dominance:** Transformer inference has AI < 10 for most ops, far below GB10's critical AI (216+ for FP16)

2. **Precision speedup = bytes reduction** in memory-bound regime:
   - FP16 → FP8: 2× faster
   - FP16 → INT4/NVFP4: ~4× faster

3. **GB10's NVFP4 advantage:** 1000 TFLOPS at 4-bit precision with minimal accuracy loss

4. **KV cache becomes bottleneck** at long context (>4K tokens for 8B models)

5. **Validation is essential:** Roofline predicts ideal performance; real kernels may differ by 10-20%

## References

- [Roofline Model (Williams et al.)](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf)
- [OCP MX Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [NVIDIA FP8 Formats](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)
- [TorchAO Quantization](https://github.com/pytorch/ao)
