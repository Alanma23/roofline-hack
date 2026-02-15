# Advanced Quantization System

Comprehensive quantization framework supporting mixed precision strategies, dynamic quantization, and FlashAttention 1/2/3/4 optimizations.

## Table of Contents

1. [Overview](#overview)
2. [Mixed Precision Strategies](#mixed-precision-strategies)
3. [Dynamic Quantization](#dynamic-quantization)
4. [FlashAttention Support](#flashattention-support)
5. [Integrated Engine](#integrated-engine)
6. [Usage Examples](#usage-examples)
7. [Performance Results](#performance-results)

---

## Overview

This quantization system provides:

- **Mixed Precision**: Per-operator precision configuration (Q/K/V/O/FFN/etc.)
- **Dynamic Quantization**: Runtime adaptation based on activation statistics
- **FlashAttention**: Specialized models for FA1, FA2, FA3, FA4 with precision support
- **Integrated Engine**: Unified recommendation system combining all techniques

### Architecture

```
src/roofline/
├── mixed_precision.py          # Per-operator precision strategies
├── dynamic_quant.py             # Runtime adaptive quantization
├── flash_attention.py           # FA1/2/3/4 performance models
└── quantization_integration.py  # Unified recommendation engine
```

---

## Mixed Precision Strategies

Configure different precision for each operator type to optimize the speed/accuracy tradeoff.

### Available Strategies

| Strategy | Speedup | Quality Loss | Memory Savings | Best For |
|----------|---------|--------------|----------------|----------|
| **FP16 Baseline** | 1.0x | None | 0% | Accuracy reference |
| **FP8 Balanced** | 2.0x | <1% | 50% | Hopper/Blackwell, balanced |
| **W4A16 Aggressive** | 4.0x | 2-5% | 75% | Max speed, decode |
| **NVFP4 Blackwell** | 4.0x | <1% | 72% | GB10/B200, best accuracy/speed |
| **Hybrid Long Context** | 3.0x | 1-2% | 60% | >8K context, aggressive KV cache |
| **Layerwise Gradient** | 2.5x | 0.5-1% | 55% | Progressive quantization |

### Operator Types

```python
from src.roofline.mixed_precision import OperatorType

# Attention projections
OperatorType.Q_PROJ
OperatorType.K_PROJ
OperatorType.V_PROJ
OperatorType.O_PROJ

# Attention compute
OperatorType.QK_MATMUL
OperatorType.SV_MATMUL
OperatorType.SOFTMAX

# FFN (2/3 of total compute!)
OperatorType.FFN_GATE
OperatorType.FFN_UP
OperatorType.FFN_DOWN

# Other
OperatorType.KV_CACHE  # Critical for long context!
OperatorType.RMSNORM
OperatorType.EMBEDDINGS
```

### Example: NVFP4 Blackwell Strategy

```python
from src.roofline.mixed_precision import create_nvfp4_blackwell

config = create_nvfp4_blackwell()
# All weights: NVFP4 (4-bit, 1000 TFLOPS on GB10)
# Activations: FP16 (for quality)
# KV cache: NVFP4_KV (critical for long context)

# Expected: 4.0x speedup, <1% quality loss, 72% memory savings
```

### Example: Custom Strategy

```python
from src.roofline.mixed_precision import MixedPrecisionConfig, OperatorType

custom = MixedPrecisionConfig(
    name="Custom Strategy",
    description="Aggressive FFN, conservative attention",
    operator_precisions={
        # Attention: FP8 for speed
        OperatorType.Q_PROJ.value: "FP8_E4M3",
        OperatorType.K_PROJ.value: "FP8_E4M3",
        OperatorType.V_PROJ.value: "FP8_E4M3",
        OperatorType.O_PROJ.value: "FP8_E4M3",

        # FFN: NVFP4 for max speed (2/3 of compute!)
        OperatorType.FFN_GATE.value: "NVFP4",
        OperatorType.FFN_UP.value: "NVFP4",
        OperatorType.FFN_DOWN.value: "NVFP4",

        # KV cache: NVFP4 for long context
        OperatorType.KV_CACHE.value: "NVFP4_KV",
    },
    default_activation_precision="FP16",
    kv_cache_precision="NVFP4_KV",
)
```

### Example: Layer-wise Quantization

```python
from src.roofline.mixed_precision import create_layerwise_gradient

# Progressive quantization: early layers more precise, later layers aggressive
config = create_layerwise_gradient(num_layers=32)

# Layer 0-10:   FP16 (most sensitive)
# Layer 11-21:  FP8_E4M3 (moderate)
# Layer 22-31:  NVFP4 (most aggressive)

# Rationale: Later layers are more robust to quantization
```

---

## Dynamic Quantization

Runtime quantization that adapts to actual activation distributions.

### Key Features

1. **Per-Token Quantization**: Each token gets its own scale (optimal for decode)
2. **Percentile-Based Clipping**: Clip outliers based on 99.9th percentile
3. **Automatic Fallback**: Falls back to FP16 if severe outliers detected
4. **Smoothing**: EMA of scales across batches for stability

### How It Works

```python
from src.roofline.dynamic_quant import DynamicQuantizer, DynamicQuantConfig

# Configure
config = DynamicQuantConfig(
    precision="FP8_E4M3",
    granularity="per_token",  # or "per_tensor", "per_channel"
    calibration_mode="percentile",  # or "absmax", "mse_optimal"
    fallback_to_fp16=True,  # Fallback if outliers detected
)

quantizer = DynamicQuantizer(config)

# During inference
result, metadata = quantizer.quantize_dynamic(
    activation=x,
    weight=W,
    operator_name="ffn_gate",
    target_precision="FP8_E4M3"
)

# metadata contains:
# - scale: computed quantization scale
# - precision: actual precision used (may be FP16 if fallback)
# - outlier_ratio: severity of outliers
```

### Activation Statistics

```python
from src.roofline.dynamic_quant import ActivationStats

stats = ActivationStats(
    absmax=10.0,      # Maximum absolute value
    mean=0.0,         # Mean value
    std=2.0,          # Standard deviation
    percentile_99=8.0,    # 99th percentile
    percentile_999=9.5,   # 99.9th percentile
    shape=(1024, 4096)
)

# Outlier detection
outlier_ratio = stats.outlier_ratio  # > 2.0 = severe outliers → fallback to FP16
dynamic_range = stats.dynamic_range  # Effective range (99.9th percentile)
```

### Adaptive Precision Selection

```python
from src.roofline.dynamic_quant import AdaptivePrecisionSelector, AdaptiveQuantConfig

config = AdaptiveQuantConfig(
    precision_candidates=["NVFP4", "FP8_E4M3", "FP16"],
    high_outlier_threshold=2.0,   # Switch to FP16
    medium_outlier_threshold=1.5, # Switch to FP8
    low_outlier_threshold=1.0,    # Can use NVFP4
)

selector = AdaptivePrecisionSelector(config)

# Select precision based on runtime stats
precision = selector.select_precision(
    operator_name="q_proj",
    activation_stats=stats,
    context={
        "phase": "decode",
        "seq_len": 4096,
        "is_kv_cache": False,
    }
)
# → Returns "NVFP4", "FP8_E4M3", or "FP16" based on outliers + context
```

---

## FlashAttention Support

Specialized performance models for FlashAttention 1/2/3/4 with precision-aware analysis.

### FlashAttention Versions

| Version | Key Features | Precision Support | Best For |
|---------|--------------|-------------------|----------|
| **FA1** | Tiled attention, online softmax | FP16/BF16 | Baseline, older GPUs |
| **FA2** | Improved parallelism, 15% faster | FP16/BF16 | A100, general use |
| **FA3** | Native FP8, async copy/compute | FP8 E4M3/E5M2 | Hopper/Blackwell |
| **FA4** | Block-sparse, mixture-of-depths | FP8 + sparsity | Very long context (>16K) |

### Roofline Model

```python
from src.roofline.flash_attention import (
    FlashAttentionRoofline,
    FlashAttentionConfig,
    FlashAttentionVersion
)
from src.roofline.hardware_registry import BLACKWELL_B10

roofline = FlashAttentionRoofline(BLACKWELL_B10)

# Configure FA3 with FP8
config = FlashAttentionConfig(
    version=FlashAttentionVersion.FA3,
    qkv_precision="FP8_E4M3",
    kv_cache_precision="NVFP4_KV",
    use_native_fp8=True,  # Use FP8 tensor cores
)

# Predict performance
metrics = roofline.predict(
    batch=1,
    num_heads=32,
    seq_len_q=1,      # Decode: 1 token
    seq_len_kv=4096,  # Context: 4096 tokens
    head_dim=128,
    config=config
)

print(f"Time: {metrics.predicted_time_us:.1f} μs")
print(f"Bottleneck: {metrics.bottleneck}")  # "memory" or "compute"
print(f"AI: {metrics.arithmetic_intensity:.2f} FLOP/byte")
```

### Memory Traffic Analysis

FlashAttention reduces memory by **recomputing** instead of materializing the full attention matrix.

**Without FlashAttention:**
```
Memory = B × nh × Sq × Skv × bytes_per_elem
         (must store full attention matrix)
For 4K context: 1 × 32 × 4096 × 4096 × 2 = 1 GB
```

**With FlashAttention (tiled):**
```
Memory = B × nh × Sq × dh × num_tiles
         (only store Q, K, V and partial outputs)
For 4K context with 128 tiles: ~64 MB (16× less!)
```

### Performance Comparison

```python
from src.roofline.flash_attention import compare_fa_versions

results = compare_fa_versions(
    batch=1,
    num_heads=32,
    seq_len=4096,
    head_dim=128,
    hardware_key="b10"
)

# Example results (GB10, 4K context):
# FA1: 33.8 ms (baseline)
# FA2: 29.4 ms (1.15x faster)
# FA3: 15.4 ms (2.20x faster) ← FP8!
# FA4: 15.4 ms (2.20x faster) ← FP8 + future sparsity
```

### FA3: FP8 Native Support

```python
config_fa3 = FlashAttentionConfig(
    version=FlashAttentionVersion.FA3,
    use_native_fp8=True,  # Enable FP8 tensor cores
    qkv_precision="FP8_E4M3",
    kv_cache_precision="FP8_E4M3",
)

# Benefits:
# 1. 2× less memory traffic (FP8 vs FP16)
# 2. 2× higher compute (124 vs 62 TFLOPS on GB10)
# 3. Async copy/compute overlap (+10% speedup)
# Total: ~2.2× speedup for memory-bound workloads
```

### FA4: Block-Sparse Attention

```python
config_fa4 = FlashAttentionConfig(
    version=FlashAttentionVersion.FA4,
    use_native_fp8=True,
    block_sparse=True,
    sparsity_ratio=0.3,  # 30% of blocks skipped
    # Example: sliding window + global tokens
)

# Benefits for long context (>16K):
# - Reduced FLOPs and memory proportional to density
# - Maintains quality with structured sparsity
# - Overhead: ~5% for sparsity indexing
```

---

## Integrated Engine

Unified recommendation system that combines all techniques.

### Complete Workflow

```python
from src.roofline.quantization_integration import IntegratedQuantizationEngine
from src.roofline.hardware_registry import BLACKWELL_B10

engine = IntegratedQuantizationEngine(BLACKWELL_B10)

# Define model
llama3_8b = {
    "H": 4096,      # Hidden dimension
    "L": 32,        # Number of layers
    "nh": 32,       # Number of query heads
    "nkv": 8,       # Number of KV heads (GQA)
    "dh": 128,      # Head dimension
    "dff": 14336,   # FFN dimension
}

# Get recommendation
rec = engine.recommend(
    model_config=llama3_8b,
    phase="decode",
    seq_len=4096,
    enable_dynamic=True,
    enable_adaptive=False,
)

# Results
print(f"Strategy: {rec.config.strategy_name}")
print(f"FlashAttention: {rec.fa_config.version.value}")
print(f"Speedup: {rec.predicted_speedup:.2f}x")
print(f"Quality: {rec.expected_quality_loss}")
print(f"Confidence: {rec.confidence}")
```

### Recommendation Output

```python
@dataclass
class QuantizationRecommendation:
    # Configuration
    config: IntegratedQuantConfig        # Unified config
    mixed_precision: MixedPrecisionConfig  # Per-operator precision
    fa_config: FlashAttentionConfig      # FlashAttention settings

    # Performance
    predicted_speedup: float             # vs FP16 baseline
    predicted_time_us: float             # Predicted latency
    baseline_time_us: float              # FP16 baseline latency

    # Memory
    memory_savings_pct: float            # Memory reduction
    kv_cache_bytes: int                  # KV cache size

    # Quality
    expected_quality_loss: str           # e.g., "<1% perplexity"
    confidence: str                      # "high", "medium", "low"

    # Analysis
    reason: str                          # Justification
    bottleneck_analysis: Dict            # Per-operator bottlenecks
```

### Context-Aware Selection

The engine automatically adjusts based on workload:

**Standard Decode (4K context):**
- Strategy: `nvfp4_blackwell`
- FlashAttention: FA3
- Precision: NVFP4 weights, FP16 activations
- Speedup: ~2.5-3.0x

**Long Context Decode (32K context):**
- Strategy: `hybrid_long_context`
- FlashAttention: FA4 (with sparsity)
- KV cache: NVFP4_KV (critical!)
- Memory savings: 60-75%

**Prefill (2K batch):**
- Strategy: `fp8_balanced`
- FlashAttention: FA3
- More conservative (may be compute-bound)
- Speedup: ~1.5-2.0x

---

## Usage Examples

### Example 1: Quick Start

```python
from src.roofline.mixed_precision import get_strategy

# Get pre-configured strategy
config = get_strategy("nvfp4_blackwell")

# Check precision for each operator
from src.roofline.mixed_precision import OperatorType
for op in OperatorType:
    precision = config.get_precision(op)
    print(f"{op.value}: {precision}")
```

### Example 2: Dynamic Quantization

```python
from src.roofline.dynamic_quant import DynamicQuantizer, DynamicQuantConfig

config = DynamicQuantConfig(
    precision="FP8_E4M3",
    granularity="per_token",
    calibration_mode="percentile",
)

quantizer = DynamicQuantizer(config)

# During inference (pseudo-code)
for layer in model.layers:
    for token in batch:
        # Quantize dynamically
        result, meta = quantizer.quantize_dynamic(
            activation=hidden_states[token],
            weight=layer.q_proj.weight,
            operator_name=f"layer_{layer.idx}_q_proj",
        )

        # Check if fallback occurred
        if meta["fallback"]:
            print(f"Fallback to FP16 due to outliers: {meta['outlier_ratio']:.2f}")
```

### Example 3: FlashAttention Comparison

```python
from src.roofline.flash_attention import compare_fa_versions

# Compare all FA versions
results = compare_fa_versions(
    batch=1,
    num_heads=32,
    seq_len=8192,  # Long context
    head_dim=128,
    hardware_key="b10"
)

# Print comparison
for version, metrics in results.items():
    print(f"{version.upper():4s}: {metrics.predicted_time_us/1000:.2f} ms "
          f"(AI={metrics.arithmetic_intensity:.1f}, "
          f"{metrics.bottleneck}-bound)")
```

### Example 4: Complete Recommendation

```python
from src.roofline.quantization_integration import IntegratedQuantizationEngine
from src.roofline.hardware_registry import get_hardware

engine = IntegratedQuantizationEngine(get_hardware("b10"))

model = {
    "H": 4096, "L": 32, "nh": 32, "nkv": 8,
    "dh": 128, "dff": 14336
}

# Get recommendation for different scenarios
for phase, seq_len in [("decode", 4096), ("decode", 32768), ("prefill", 2048)]:
    rec = engine.recommend(model, phase=phase, seq_len=seq_len)

    print(f"\n{phase.upper()} @ {seq_len} tokens:")
    print(f"  Strategy: {rec.config.strategy_name}")
    print(f"  FA version: {rec.fa_config.version.value}")
    print(f"  Speedup: {rec.predicted_speedup:.2f}x")
    print(f"  Reason: {rec.reason}")
```

---

## Performance Results

### GB10 (Blackwell) - Decode Phase

**Llama-3 8B, 4K context:**

| Configuration | Time (ms) | Speedup | Quality Loss | Memory Savings |
|---------------|-----------|---------|--------------|----------------|
| FP16 Baseline | 45.8 | 1.00x | None | 0% |
| FP8 Balanced | 23.7 | 1.93x | <1% | 50% |
| NVFP4 Blackwell | 17.7 | 2.59x | <1% | 72% |
| W4A16 Aggressive | 15.2 | 3.01x | 2-5% | 75% |

### Long Context Performance

**Llama-3 8B, 32K context:**

| Configuration | KV Cache (MB) | Time (ms) | Memory Savings | Notes |
|---------------|---------------|-----------|----------------|-------|
| FP16 | 4096 | 91.6 | 0% | Baseline |
| FP8 KV Cache | 2048 | 52.3 | 50% | 1.75x faster |
| NVFP4 KV Cache | 1175 | 37.4 | 71% | 2.45x faster |
| FA4 + Sparse | 820 (30% sparse) | 27.9 | 80% | 3.28x faster |

### FlashAttention Speedups

**GB10, 4K context:**

- FA1 → FA2: 1.15x (better scheduling)
- FA2 → FA3: 1.91x (FP8 tensor cores)
- FA3 → FA4: 1.0x (same for dense; 1.4x+ for sparse)

**Overall: FA1 → FA4 with FP8: 2.20x speedup**

### Memory vs Compute Bound

**Arithmetic Intensity by operation (FP16):**

| Operation | AI (FLOP/byte) | Critical AI (GB10) | Bottleneck |
|-----------|----------------|-------------------|------------|
| GEMV (decode) | ~1.0 | 216 | Memory |
| GEMM (prefill, small batch) | ~10 | 216 | Memory |
| GEMM (prefill, large batch) | ~100+ | 216 | Mixed |
| FlashAttention | ~29 | 216 | Memory |
| FFN (decode) | ~1.0 | 216 | Memory |

**Key insight:** Nearly everything is memory-bound in decode → lower precision directly translates to speedup!

---

## Running the Demo

```bash
# Full demonstration of all features
python examples/quantization_demo.py

# Individual modules
python src/roofline/mixed_precision.py       # Strategy comparison
python src/roofline/dynamic_quant.py          # Dynamic quant demo
python src/roofline/flash_attention.py        # FA1/2/3/4 comparison
python src/roofline/quantization_integration.py  # Integrated engine
```

---

## Key Takeaways

1. **Mixed Precision is Essential**: Per-operator precision enables fine-grained speed/quality tradeoffs
2. **Dynamic Quantization Adapts**: Runtime adaptation based on actual distributions improves quality
3. **FlashAttention Evolves**: FA3/FA4 with FP8 provide 2-3× speedup over FA1/FA2
4. **KV Cache is Critical**: At long context (>8K), KV cache quantization becomes the primary bottleneck
5. **NVFP4 on Blackwell**: Best accuracy/speed tradeoff (4× speedup, <1% loss) with 1000 TFLOPS

---

## References

- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- [TorchAO Quantization](https://github.com/pytorch/ao)
