# Operator Math: Per-Operator FLOP and Byte Derivations

Every operator in a transformer decoder layer, with exact FLOP counts and byte
counts parameterized by precision format.

## Notation

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
| V | Vocabulary size |
| L | Number of layers |
| w_B | Bytes per weight element |
| a_B | Bytes per activation element |
| kv_B | Bytes per KV cache element |
| o_B | Bytes per output element (= a_B) |

## Attention Block

### Linear Projections (Q, K, V, O)

Standard GEMM: C[M,N] = A[M,K] × W[K,N]

| Projection | M | N | K | FLOPs | Bytes |
|------------|---|---|---|-------|-------|
| Q | B×T | H | H | 2·B·T·H² | B·T·H·a_B + H²·w_B + B·T·H·o_B |
| K | B×T | d_kv | H | 2·B·T·H·d_kv | B·T·H·a_B + H·d_kv·w_B + B·T·d_kv·o_B |
| V | B×T | d_kv | H | same as K | same as K |
| O | B×T | H | H | 2·B·T·H² | B·T·H·a_B + H²·w_B + B·T·H·o_B |

**Key insight for decode (T=1):**

```
AI_decode_proj = 2·H² / (H²·w_B + H·a_B + H·o_B)
               ≈ 2·H / (H·w_B + a_B + o_B)
               ≈ 2 / w_B                       (for large H)
```

This is independent of H! Decode projections always have AI ≈ 2/w_B.

### Attention Score (QK^T)

Batched matmul: Score[B,n_h,T,S] = Q[B,n_h,T,d_h] × K^T[B,n_kv,d_h,S]

**FLOPs**: 2 · B · n_h · T · S · d_h

**Bytes**:
- Read Q: B · n_h · T · d_h · a_B
- Read K (from KV cache): B · n_kv · S · d_h · **kv_B** ← KV cache precision!
- Write scores: B · n_h · T · S · a_B

**Prefill AI** (T = S): ≈ 2·d_h / (a_B + kv_B·(n_kv/n_h) + a_B)
For d_h=128, FP16 everything: AI ≈ 256/6 ≈ 42.7 → **compute-bound**

**Decode AI** (T = 1): ≈ 2·B·n_h·S·d_h / (B·n_h·d_h·a_B + B·n_kv·S·d_h·kv_B + B·n_h·S·a_B)
For large S, the KV read term dominates and AI → 2·d_h / (d_h·kv_B·(n_kv/n_h) + a_B)
With GQA (n_kv << n_h), this is actually quite favorable.
But the absolute bytes grow linearly with S → **this is the KV cache wall**.

### Score × V

Batched matmul: Out[B,n_h,T,d_h] = Score[B,n_h,T,S] × V[B,n_kv,S,d_h]

Same FLOPs and similar byte structure as QK^T. V is read from KV cache at kv_B.

### Softmax

**FLOPs**: ~5 · B · n_h · T · S (exp, sum, divide, plus numerical stability)
**Bytes**: 2 · B · n_h · T · S · a_B (read + write)
**AI**: ≈ 2.5 → always memory-bound

## FFN Block (SwiGLU / LLaMA-style)

Three projections: gate, up, down.

| Projection | M | N | K | FLOPs | Weight Bytes |
|------------|---|---|---|-------|--------------|
| Gate | B×T | d_ff | H | 2·B·T·H·d_ff | H·d_ff·w_B |
| Up | B×T | d_ff | H | 2·B·T·H·d_ff | H·d_ff·w_B |
| Down | B×T | H | d_ff | 2·B·T·H·d_ff | H·d_ff·w_B |

**SiLU + elementwise multiply** (between gate and up outputs):
- FLOPs: 3 · B · T · d_ff
- Bytes: 3 · B · T · d_ff · a_B (read two, write one)

**Total FFN FLOPs per layer**: 6 · B · T · H · d_ff (for SwiGLU)

For LLaMA-style where d_ff ≈ 8/3 · H:
FFN FLOPs = 6 · B · T · H · (8/3 · H) = 16 · B · T · H²

Attention FLOPs (projections only) = 4 · 2 · B · T · H² = 8 · B · T · H²

→ **FFN is ~2/3 of total layer compute** in LLaMA models.

## Elementwise Operations

| Op | FLOPs | Bytes | AI |
|----|-------|-------|----|
| RMSNorm | 5·B·T·H | 2·B·T·H·a_B | ~2.5 |
| Residual add | B·T·H | 3·B·T·H·a_B | ~0.33 |
| RoPE | 6·B·T·H | 2·B·T·H·a_B | ~3 |

All have AI < 5 → always memory-bound. Candidates for **operator fusion**.

## Full Model Totals

**Total FLOPs per forward pass** (approximate, for SwiGLU + GQA):
```
prefill: L × (8·B·S·H² + 4·B·n_h·S²·d_h + 6·B·S·H·d_ff) + 2·B·S·H·V
decode:  L × (8·B·H² + 4·B·n_h·S·d_h + 6·B·H·d_ff) + 2·B·H·V
```

**Total weight bytes** (all layers + logit projection):
```
L × (4·H² + 2·H·d_kv + 3·H·d_ff) × w_B + H·V·w_B
```

**KV cache bytes per decode step** (reads for attention):
```
L × 2 × n_kv × S × d_h × kv_B
```

## The KV Cache Crossover Point

Context length where KV cache reading overtakes weight loading:

```
S_cross = [L × (4H² + 2H·d_kv + 3H·d_ff) × w_B + H·V·w_B] / [L × 2 × n_kv × d_h × kv_B]
```

| Model | W4A16 + FP16 KV | FP8 W + FP8 KV | FP8 W + NVFP4 KV |
|-------|-----------------|----------------|-------------------|
| Llama-3 8B | ~1600 | ~4096 | ~7300 |
| Llama-2 70B | ~12800 | ~32000 | ~57000 |
| o1 est. | ~4800 | ~12000 | ~21000 |

For reasoning models generating 10K+ thinking tokens, KV almost always dominates → KV cache quantization (NVFP4 KV) is the highest-leverage optimization.
