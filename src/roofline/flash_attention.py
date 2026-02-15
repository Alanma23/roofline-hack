"""
FlashAttention 1/2/3/4 Roofline Models & Quantization Support

Specialized performance models for FlashAttention variants with precision-aware
analysis. Each FA version has different memory access patterns and quantization support.

FlashAttention versions:
- FA1: Original fused attention, tiled softmax
- FA2: Improved parallelism, better work partitioning
- FA3: Native FP8 support, optimized for Hopper/Blackwell
- FA4: Block-sparse support, mixture-of-depths, advanced fusion
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from enum import Enum


class FlashAttentionVersion(Enum):
    """FlashAttention algorithm versions."""
    FA1 = "fa1"  # Original: Dao et al. 2022
    FA2 = "fa2"  # FA2: Improved parallelism, 2023
    FA3 = "fa3"  # FA3: FP8 native, Hopper/Blackwell, 2024
    FA4 = "fa4"  # FA4: Block-sparse + MoD, 2025


@dataclass
class FlashAttentionConfig:
    """Configuration for FlashAttention kernel."""
    version: FlashAttentionVersion = FlashAttentionVersion.FA2

    # Precision settings
    qkv_precision: str = "FP16"  # Q, K, V precision
    score_precision: str = "FP16"  # Attention score accumulation
    output_precision: str = "FP16"  # Output precision
    kv_cache_precision: str = "FP16"  # KV cache storage

    # Tiling parameters (for roofline analysis)
    tile_size_q: int = 128  # Query tile size (Br in paper)
    tile_size_kv: int = 128  # KV tile size (Bc in paper)

    # FA3-specific: FP8 support
    use_native_fp8: bool = False  # Use FP8 tensor cores (FA3+)

    # FA4-specific: sparsity
    block_sparse: bool = False  # Enable block-sparse attention
    sparsity_ratio: float = 0.0  # Fraction of blocks skipped (0.0 = dense)

    # Causal masking
    is_causal: bool = True

    # Hardware
    sm_count: int = 108  # Number of SMs (for parallelism analysis)


@dataclass
class FlashAttentionMetrics:
    """Performance metrics for FlashAttention."""
    version: str
    total_flops: int
    total_bytes: int
    arithmetic_intensity: float

    # Memory access breakdown
    qkv_bytes: int
    kv_cache_bytes: int
    output_bytes: int
    intermediate_bytes: int  # Softmax stats (m, l)

    # Time breakdown (microseconds)
    predicted_time_us: float
    qk_time_us: float
    softmax_time_us: float
    sv_time_us: float

    # Bottleneck analysis
    bottleneck: str  # "memory" or "compute"
    memory_efficiency: float  # Fraction of peak bandwidth utilized
    compute_efficiency: float  # Fraction of peak FLOPS utilized

    # Optimizations enabled
    uses_tiling: bool = True
    uses_online_softmax: bool = True
    uses_fp8: bool = False
    uses_sparsity: bool = False


class FlashAttentionRoofline:
    """
    Roofline performance model for FlashAttention variants.

    Accounts for:
    1. Tiled computation (reduces memory by recomputing)
    2. Online softmax (no materialization of full attention matrix)
    3. Precision-specific optimizations (FP8 in FA3, sparsity in FA4)
    """

    def __init__(self, hardware_spec):
        """
        Args:
            hardware_spec: HardwareSpec from hardware_registry
        """
        self.hw = hardware_spec

    def predict_fa1(self,
                    batch: int,
                    num_heads: int,
                    seq_len_q: int,
                    seq_len_kv: int,
                    head_dim: int,
                    config: FlashAttentionConfig) -> FlashAttentionMetrics:
        """
        Predict FA1 performance.

        FA1 characteristics:
        - Tiled QK^T and softmax (Br × Bc blocks)
        - Online softmax with running max/sum
        - No FP8 support (FP16/BF16 only)
        - Memory: O(batch × heads × seq_len × head_dim) instead of O(seq_len^2)
        """
        from .calculator_shell import bytes_per_element

        B, nh, Sq, Skv, dh = batch, num_heads, seq_len_q, seq_len_kv, head_dim
        Br, Bc = config.tile_size_q, config.tile_size_kv

        # Precision
        qkv_bpe = bytes_per_element(config.qkv_precision)
        kv_cache_bpe = bytes_per_element(config.kv_cache_precision)

        # ── FLOPs ──
        # QK^T: [B, nh, Sq, dh] @ [B, nh, dh, Skv] → [B, nh, Sq, Skv]
        qk_flops = 2 * B * nh * Sq * Skv * dh

        # Softmax: exp, sum, divide (≈5 ops per element)
        softmax_flops = 5 * B * nh * Sq * Skv

        # Score @ V: [B, nh, Sq, Skv] @ [B, nh, Skv, dh] → [B, nh, Sq, dh]
        sv_flops = 2 * B * nh * Sq * Skv * dh

        total_flops = qk_flops + softmax_flops + sv_flops

        # ── Memory Traffic (key innovation: tiling reduces this!) ──
        # Q: loaded once per KV tile (ceil(Skv / Bc) times)
        num_kv_tiles = (Skv + Bc - 1) // Bc
        q_bytes = B * nh * Sq * dh * qkv_bpe * num_kv_tiles

        # K, V: loaded once per Q tile (ceil(Sq / Br) times)
        num_q_tiles = (Sq + Br - 1) // Br
        kv_bytes = 2 * B * nh * Skv * dh * kv_cache_bpe * num_q_tiles

        # Output: written once
        output_bytes = B * nh * Sq * dh * qkv_bpe

        # Softmax statistics (m, l): small, amortized
        # Each Q tile stores: max (m) and sum (l) per row
        intermediate_bytes = 2 * B * nh * Sq * 4  # FP32 for stability

        total_bytes = q_bytes + kv_bytes + output_bytes + intermediate_bytes

        # ── Roofline Analysis ──
        ai = total_flops / total_bytes if total_bytes > 0 else 0
        peak_tflops = self.hw.peak_flops_tflops.get(config.qkv_precision,
                                                     self.hw.peak_flops_tflops.get("FP16", 62.0))

        time_compute_s = total_flops / (peak_tflops * 1e12)
        time_memory_s = total_bytes / (self.hw.peak_bandwidth_gb_s * 1e9)
        predicted_time_s = max(time_compute_s, time_memory_s)
        predicted_time_us = predicted_time_s * 1e6

        bottleneck = "memory" if time_memory_s >= time_compute_s else "compute"
        memory_efficiency = (total_bytes / predicted_time_s) / (self.hw.peak_bandwidth_gb_s * 1e9)
        compute_efficiency = (total_flops / predicted_time_s) / (peak_tflops * 1e12)

        # Time breakdown (approximate)
        qk_frac = qk_flops / total_flops
        softmax_frac = softmax_flops / total_flops
        sv_frac = sv_flops / total_flops

        return FlashAttentionMetrics(
            version="FA1",
            total_flops=total_flops,
            total_bytes=int(total_bytes),
            arithmetic_intensity=ai,
            qkv_bytes=int(q_bytes + kv_bytes),
            kv_cache_bytes=int(kv_bytes),
            output_bytes=int(output_bytes),
            intermediate_bytes=int(intermediate_bytes),
            predicted_time_us=predicted_time_us,
            qk_time_us=predicted_time_us * qk_frac,
            softmax_time_us=predicted_time_us * softmax_frac,
            sv_time_us=predicted_time_us * sv_frac,
            bottleneck=bottleneck,
            memory_efficiency=memory_efficiency,
            compute_efficiency=compute_efficiency,
            uses_tiling=True,
            uses_online_softmax=True,
            uses_fp8=False,
            uses_sparsity=False,
        )

    def predict_fa2(self, batch: int, num_heads: int, seq_len_q: int,
                    seq_len_kv: int, head_dim: int,
                    config: FlashAttentionConfig) -> FlashAttentionMetrics:
        """
        Predict FA2 performance.

        FA2 improvements over FA1:
        - Better parallelism: non-matmul FLOPs in epilogue/prologue
        - Reduced shared memory usage
        - Better occupancy on modern GPUs

        For roofline: ~same FLOPs/bytes, but ~15-20% faster due to better scheduling
        """
        # Start with FA1 prediction
        metrics = self.predict_fa1(batch, num_heads, seq_len_q, seq_len_kv, head_dim, config)

        # FA2 improvement factor (empirical, from paper)
        fa2_speedup = 1.15  # 15% faster on average

        metrics.version = "FA2"
        metrics.predicted_time_us /= fa2_speedup
        metrics.qk_time_us /= fa2_speedup
        metrics.softmax_time_us /= fa2_speedup
        metrics.sv_time_us /= fa2_speedup

        # Better efficiency due to improved scheduling
        metrics.compute_efficiency *= fa2_speedup

        return metrics

    def predict_fa3(self, batch: int, num_heads: int, seq_len_q: int,
                    seq_len_kv: int, head_dim: int,
                    config: FlashAttentionConfig) -> FlashAttentionMetrics:
        """
        Predict FA3 performance.

        FA3 key features:
        - Native FP8 support (use_native_fp8=True)
        - 2× throughput on Hopper/Blackwell with FP8
        - Optimized for lower precision compute
        - Async copy/compute overlap (additional speedup)
        """
        from .calculator_shell import bytes_per_element

        # If FP8 enabled, override precision
        if config.use_native_fp8:
            config.qkv_precision = "FP8_E4M3"
            config.score_precision = "FP8_E4M3"

        # Start with FA2 prediction
        metrics = self.predict_fa2(batch, num_heads, seq_len_q, seq_len_kv, head_dim, config)
        metrics.version = "FA3"
        metrics.uses_fp8 = config.use_native_fp8

        if config.use_native_fp8:
            # FP8 benefits:
            # 1. 2× less memory traffic → 2× faster if memory-bound
            # 2. Higher FLOPS (124 vs 62 TFLOPS on GB10)

            # Recompute with FP8
            qkv_bpe = bytes_per_element("FP8_E4M3")
            B, nh, Sq, Skv, dh = batch, num_heads, seq_len_q, seq_len_kv, head_dim
            Br, Bc = config.tile_size_q, config.tile_size_kv

            num_kv_tiles = (Skv + Bc - 1) // Bc
            num_q_tiles = (Sq + Br - 1) // Br

            q_bytes = B * nh * Sq * dh * qkv_bpe * num_kv_tiles
            kv_bytes = 2 * B * nh * Skv * dh * qkv_bpe * num_q_tiles
            output_bytes = B * nh * Sq * dh * qkv_bpe
            intermediate_bytes = 2 * B * nh * Sq * 4

            total_bytes_fp8 = q_bytes + kv_bytes + output_bytes + intermediate_bytes
            metrics.total_bytes = int(total_bytes_fp8)

            # Recompute time with FP8
            peak_tflops_fp8 = self.hw.peak_flops_tflops.get("FP8_E4M3", 124.0)
            time_compute_s = metrics.total_flops / (peak_tflops_fp8 * 1e12)
            time_memory_s = total_bytes_fp8 / (self.hw.peak_bandwidth_gb_s * 1e9)

            # FA3 has async copy/compute overlap → additional 10% speedup
            async_speedup = 1.1
            predicted_time_s = max(time_compute_s, time_memory_s) / async_speedup

            metrics.predicted_time_us = predicted_time_s * 1e6
            metrics.qk_time_us = metrics.predicted_time_us * 0.4
            metrics.softmax_time_us = metrics.predicted_time_us * 0.2
            metrics.sv_time_us = metrics.predicted_time_us * 0.4

        return metrics

    def predict_fa4(self, batch: int, num_heads: int, seq_len_q: int,
                    seq_len_kv: int, head_dim: int,
                    config: FlashAttentionConfig) -> FlashAttentionMetrics:
        """
        Predict FA4 performance.

        FA4 advanced features:
        - Block-sparse attention (config.sparsity_ratio)
        - Mixture-of-Depths (conditional computation)
        - Advanced fusion (GQA, MQA optimizations)
        - FP8 + INT4 hybrid support
        """
        # Start with FA3 prediction
        metrics = self.predict_fa3(batch, num_heads, seq_len_q, seq_len_kv, head_dim, config)
        metrics.version = "FA4"
        metrics.uses_sparsity = config.block_sparse

        if config.block_sparse and config.sparsity_ratio > 0:
            # Block-sparse reduces FLOPs and memory proportionally
            sparsity = config.sparsity_ratio
            dense_ratio = 1.0 - sparsity

            # Reduce compute and memory by density
            metrics.total_flops = int(metrics.total_flops * dense_ratio)
            metrics.total_bytes = int(metrics.total_bytes * dense_ratio)
            metrics.kv_cache_bytes = int(metrics.kv_cache_bytes * dense_ratio)

            # Recompute time
            peak_tflops = self.hw.peak_flops_tflops.get(config.qkv_precision, 62.0)
            time_compute_s = metrics.total_flops / (peak_tflops * 1e12)
            time_memory_s = metrics.total_bytes / (self.hw.peak_bandwidth_gb_s * 1e9)

            # FA4 has additional overhead for sparsity indexing (~5%)
            sparsity_overhead = 1.05
            predicted_time_s = max(time_compute_s, time_memory_s) * sparsity_overhead

            metrics.predicted_time_us = predicted_time_s * 1e6
            metrics.qk_time_us = metrics.predicted_time_us * 0.35
            metrics.softmax_time_us = metrics.predicted_time_us * 0.15
            metrics.sv_time_us = metrics.predicted_time_us * 0.50

            # Update AI
            metrics.arithmetic_intensity = metrics.total_flops / metrics.total_bytes if metrics.total_bytes > 0 else 0

        return metrics

    def predict(self, batch: int, num_heads: int, seq_len_q: int,
                seq_len_kv: int, head_dim: int,
                config: FlashAttentionConfig) -> FlashAttentionMetrics:
        """Predict performance for specified FA version."""
        predictors = {
            FlashAttentionVersion.FA1: self.predict_fa1,
            FlashAttentionVersion.FA2: self.predict_fa2,
            FlashAttentionVersion.FA3: self.predict_fa3,
            FlashAttentionVersion.FA4: self.predict_fa4,
        }
        return predictors[config.version](batch, num_heads, seq_len_q, seq_len_kv, head_dim, config)


# ═══════════════════════════════════════════════
#  QUANTIZATION STRATEGIES FOR FLASHATTENTION
# ═══════════════════════════════════════════════

def recommend_fa_precision(
    version: FlashAttentionVersion,
    seq_len: int,
    hardware_name: str = "GB10",
    phase: str = "decode"
) -> FlashAttentionConfig:
    """
    Recommend optimal FlashAttention configuration based on version and workload.

    Args:
        version: FA1, FA2, FA3, or FA4
        seq_len: Sequence length (context length)
        hardware_name: Target hardware
        phase: "prefill" or "decode"

    Returns:
        Optimized FlashAttentionConfig
    """
    # Base config
    config = FlashAttentionConfig(version=version)

    if version == FlashAttentionVersion.FA1:
        # FA1: FP16/BF16 only
        config.qkv_precision = "FP16"
        config.kv_cache_precision = "FP16"
        config.use_native_fp8 = False

    elif version == FlashAttentionVersion.FA2:
        # FA2: Still FP16, but can use lower precision KV cache
        config.qkv_precision = "FP16"
        config.use_native_fp8 = False

        # Long context: aggressive KV cache quantization
        if seq_len > 4096:
            config.kv_cache_precision = "FP8_E4M3"
        else:
            config.kv_cache_precision = "FP16"

    elif version == FlashAttentionVersion.FA3:
        # FA3: Native FP8 support on Hopper/Blackwell
        if hardware_name in ("GB10", "B200", "H100"):
            config.use_native_fp8 = True
            config.qkv_precision = "FP8_E4M3"
            config.score_precision = "FP8_E4M3"

            # KV cache: NVFP4 for long context
            if seq_len > 8192:
                config.kv_cache_precision = "NVFP4_KV"
            else:
                config.kv_cache_precision = "FP8_E4M3"
        else:
            # Fallback for non-Hopper/Blackwell
            config.use_native_fp8 = False
            config.qkv_precision = "FP16"
            config.kv_cache_precision = "FP16"

    elif version == FlashAttentionVersion.FA4:
        # FA4: Most aggressive optimizations
        if hardware_name in ("GB10", "B200"):
            config.use_native_fp8 = True
            config.qkv_precision = "FP8_E4M3"
            config.kv_cache_precision = "NVFP4_KV"  # Always use NVFP4 for KV

            # Enable sparsity for very long context
            if seq_len > 16384:
                config.block_sparse = True
                config.sparsity_ratio = 0.3  # 30% sparse (e.g., sliding window + global tokens)
        else:
            config.use_native_fp8 = False
            config.qkv_precision = "FP16"
            config.kv_cache_precision = "FP8_E4M3" if seq_len > 4096 else "FP16"

    # Adjust tile sizes based on phase
    if phase == "prefill":
        # Prefill: larger tiles for better compute utilization
        config.tile_size_q = 128
        config.tile_size_kv = 128
    else:
        # Decode: smaller tiles (Sq=1, so Br can be small)
        config.tile_size_q = 32
        config.tile_size_kv = 128

    return config


def compare_fa_versions(
    batch: int = 1,
    num_heads: int = 32,
    seq_len: int = 4096,
    head_dim: int = 128,
    hardware_key: str = "b10"
) -> Dict[str, FlashAttentionMetrics]:
    """Compare all FA versions for given workload."""
    from .hardware_registry import get_hardware

    hw = get_hardware(hardware_key)
    roofline = FlashAttentionRoofline(hw)

    results = {}
    for version in FlashAttentionVersion:
        config = recommend_fa_precision(version, seq_len, "GB10" if hardware_key == "b10" else "H100")
        metrics = roofline.predict(batch, num_heads, seq_len, seq_len, head_dim, config)
        results[version.value] = metrics

    return results


if __name__ == "__main__":
    from .hardware_registry import BLACKWELL_B10

    print("=" * 90)
    print("FlashAttention 1/2/3/4 Performance Comparison")
    print("=" * 90)

    # Test workload: Llama-3 8B decode at 4K context
    B, nh, S, dh = 1, 32, 4096, 128

    roofline = FlashAttentionRoofline(BLACKWELL_B10)

    print(f"\nWorkload: batch={B}, heads={nh}, seq_len={S}, head_dim={dh}")
    print(f"Hardware: {BLACKWELL_B10.name} ({BLACKWELL_B10.peak_bandwidth_gb_s} GB/s)")
    print("-" * 90)

    for version in FlashAttentionVersion:
        config = recommend_fa_precision(version, S, "GB10", "decode")
        metrics = roofline.predict(B, nh, S, S, dh, config)

        print(f"\n{version.value.upper()} ({config.qkv_precision}, KV cache: {config.kv_cache_precision})")
        print(f"  Time: {metrics.predicted_time_us:.1f} μs")
        print(f"  FLOPs: {metrics.total_flops / 1e9:.2f} GFLOPs")
        print(f"  Bytes: {metrics.total_bytes / 1e6:.2f} MB")
        print(f"  AI: {metrics.arithmetic_intensity:.2f} FLOP/byte")
        print(f"  Bottleneck: {metrics.bottleneck}")
        print(f"  Memory efficiency: {metrics.memory_efficiency * 100:.1f}%")
        print(f"  Compute efficiency: {metrics.compute_efficiency * 100:.1f}%")

        if metrics.uses_fp8:
            print(f"  ✓ Using FP8 tensor cores")
        if metrics.uses_sparsity:
            print(f"  ✓ Block-sparse attention enabled")

    # Speedup comparison
    print("\n" + "=" * 90)
    print("Speedup vs FA1:")
    print("-" * 90)

    config_fa1 = recommend_fa_precision(FlashAttentionVersion.FA1, S, "GB10", "decode")
    metrics_fa1 = roofline.predict(B, nh, S, S, dh, config_fa1)
    baseline_time = metrics_fa1.predicted_time_us

    for version in FlashAttentionVersion:
        config = recommend_fa_precision(version, S, "GB10", "decode")
        metrics = roofline.predict(B, nh, S, S, dh, config)
        speedup = baseline_time / metrics.predicted_time_us
        print(f"  {version.value.upper():4s}: {speedup:.2f}x")

    # Long context comparison
    print("\n" + "=" * 90)
    print("Long Context Performance (32K tokens)")
    print("=" * 90)

    S_long = 32768
    print(f"\nWorkload: seq_len={S_long}")
    print("-" * 90)

    for version in [FlashAttentionVersion.FA3, FlashAttentionVersion.FA4]:
        config = recommend_fa_precision(version, S_long, "GB10", "decode")
        metrics = roofline.predict(B, nh, S_long, S_long, dh, config)

        print(f"\n{version.value.upper()} ({config.qkv_precision}, KV: {config.kv_cache_precision})")
        print(f"  Time: {metrics.predicted_time_us:.1f} μs")
        print(f"  KV cache bytes: {metrics.kv_cache_bytes / 1e6:.1f} MB")
        if config.block_sparse:
            print(f"  Sparsity: {config.sparsity_ratio * 100:.0f}% sparse")
