"""
Integrated Quantization System

Combines mixed precision strategies, dynamic quantization, and FlashAttention
support into a unified recommendation and execution engine.
"""

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
from .mixed_precision import (
    MixedPrecisionConfig,
    OperatorType,
    get_strategy,
    STRATEGY_REGISTRY,
)
from .dynamic_quant import (
    DynamicQuantizer,
    AdaptivePrecisionSelector,
    PerTokenQuantizer,
    DynamicQuantConfig,
    AdaptiveQuantConfig,
)
from .flash_attention import (
    FlashAttentionVersion,
    FlashAttentionConfig,
    FlashAttentionRoofline,
    recommend_fa_precision,
)


@dataclass
class IntegratedQuantConfig:
    """
    Unified configuration for complete quantization strategy.

    Combines:
    - Mixed precision strategy (per-operator precision)
    - Dynamic quantization settings (runtime adaptation)
    - FlashAttention version and settings
    """
    # High-level strategy
    strategy_name: str = "nvfp4_blackwell"  # From STRATEGY_REGISTRY

    # FlashAttention configuration
    fa_version: FlashAttentionVersion = FlashAttentionVersion.FA3
    use_flash_attention: bool = True

    # Dynamic quantization
    enable_dynamic_quant: bool = True
    dynamic_quant_config: Optional[DynamicQuantConfig] = None

    # Adaptive precision
    enable_adaptive_precision: bool = False
    adaptive_config: Optional[AdaptiveQuantConfig] = None

    # Per-token quantization (for decode)
    enable_per_token_quant: bool = False

    # Context-aware settings
    phase: str = "decode"  # "prefill" or "decode"
    seq_len: int = 4096
    max_seq_len: int = 8192


@dataclass
class QuantizationRecommendation:
    """Complete quantization recommendation with justification."""
    config: IntegratedQuantConfig
    mixed_precision: MixedPrecisionConfig
    fa_config: FlashAttentionConfig

    # Performance predictions
    predicted_speedup: float
    predicted_time_us: float
    baseline_time_us: float

    # Memory analysis
    memory_savings_pct: float
    kv_cache_bytes: int

    # Quality estimates
    expected_quality_loss: str
    confidence: str  # "high", "medium", "low"

    # Justification
    reason: str
    bottleneck_analysis: Dict[str, str]


class IntegratedQuantizationEngine:
    """
    Main engine for quantization recommendation and execution.

    Analyzes workload, selects optimal mixed precision + FlashAttention config,
    and provides both static and dynamic quantization support.
    """

    def __init__(self, hardware_spec):
        """
        Args:
            hardware_spec: HardwareSpec from hardware_registry
        """
        self.hw = hardware_spec
        self.fa_roofline = FlashAttentionRoofline(hardware_spec)

    def recommend(self,
                  model_config: Dict,
                  phase: str = "decode",
                  seq_len: int = 4096,
                  enable_dynamic: bool = True,
                  enable_adaptive: bool = False) -> QuantizationRecommendation:
        """
        Recommend optimal quantization configuration.

        Args:
            model_config: Dict with H, L, nh, nkv, dh, etc.
            phase: "prefill" or "decode"
            seq_len: Sequence length
            enable_dynamic: Enable dynamic quantization
            enable_adaptive: Enable adaptive precision selection

        Returns:
            Complete quantization recommendation
        """
        H = model_config.get("H", 4096)
        L = model_config.get("L", 32)
        nh = model_config.get("nh", 32)
        nkv = model_config.get("nkv", 8)
        dh = model_config.get("dh", 128)

        # ── Step 1: Select base mixed precision strategy ──
        strategy_name = self._select_base_strategy(phase, seq_len, model_config)
        mixed_precision = get_strategy(strategy_name)

        # ── Step 2: Select FlashAttention version and config ──
        fa_version = self._select_fa_version(seq_len, phase)
        fa_config = recommend_fa_precision(fa_version, seq_len, self.hw.name, phase)

        # ── Step 3: Predict performance ──
        # Attention performance (per layer)
        fa_metrics = self.fa_roofline.predict(
            batch=1, num_heads=nh, seq_len_q=1 if phase == "decode" else seq_len,
            seq_len_kv=seq_len, head_dim=dh, config=fa_config
        )

        # FFN performance (per layer) - use mixed precision config
        from .calculator_shell import RooflineCalculator
        calc = RooflineCalculator(self.hw)
        ffn_dim = model_config.get("dff", int(8 * H / 3))

        ffn_gate_prec = mixed_precision.get_precision(OperatorType.FFN_GATE)
        ffn_gate_metrics = calc.predict_gemv(ffn_dim, H, ffn_gate_prec)

        # Total time per layer (attention + FFN)
        time_per_layer_us = fa_metrics.predicted_time_us + 3 * ffn_gate_metrics["predicted_time_us"]
        predicted_time_us = time_per_layer_us * L

        # Baseline: FP16 everywhere
        fa_config_fp16 = FlashAttentionConfig(
            version=FlashAttentionVersion.FA2,
            qkv_precision="FP16",
            kv_cache_precision="FP16"
        )
        fa_metrics_fp16 = self.fa_roofline.predict(
            1, nh, 1 if phase == "decode" else seq_len, seq_len, dh, fa_config_fp16
        )
        ffn_metrics_fp16 = calc.predict_gemv(ffn_dim, H, "FP16")
        baseline_time_per_layer = fa_metrics_fp16.predicted_time_us + 3 * ffn_metrics_fp16["predicted_time_us"]
        baseline_time_us = baseline_time_per_layer * L

        predicted_speedup = baseline_time_us / predicted_time_us

        # ── Step 4: Memory analysis ──
        # KV cache size: L × 2 × nkv × seq_len × dh × bytes_per_element(kv_precision)
        from .calculator_shell import bytes_per_element
        kv_bpe = bytes_per_element(fa_config.kv_cache_precision)
        kv_cache_bytes = L * 2 * nkv * seq_len * dh * kv_bpe

        kv_bpe_fp16 = bytes_per_element("FP16")
        kv_cache_bytes_fp16 = L * 2 * nkv * seq_len * dh * kv_bpe_fp16
        memory_savings_pct = (1 - kv_cache_bytes / kv_cache_bytes_fp16) * 100

        # ── Step 5: Build recommendation ──
        integrated_config = IntegratedQuantConfig(
            strategy_name=strategy_name,
            fa_version=fa_version,
            use_flash_attention=True,
            enable_dynamic_quant=enable_dynamic,
            enable_adaptive_precision=enable_adaptive,
            enable_per_token_quant=(phase == "decode" and seq_len < 128),
            phase=phase,
            seq_len=seq_len,
        )

        # Dynamic quant config
        if enable_dynamic:
            integrated_config.dynamic_quant_config = DynamicQuantConfig(
                precision=fa_config.qkv_precision,
                granularity="per_token" if phase == "decode" else "per_tensor",
                calibration_mode="percentile",
                fallback_to_fp16=True,
            )

        # Adaptive config
        if enable_adaptive:
            integrated_config.adaptive_config = AdaptiveQuantConfig(
                precision_candidates=["NVFP4", "FP8_E4M3", "FP16"],
                target_speedup=3.0,
            )

        # Generate justification
        reason = self._generate_justification(
            strategy_name, fa_version, phase, seq_len, predicted_speedup, fa_metrics
        )

        bottleneck_analysis = {
            "attention": fa_metrics.bottleneck,
            "ffn": ffn_gate_metrics["bottleneck"],
            "overall": "memory" if fa_metrics.bottleneck == "memory" else "mixed",
        }

        # Confidence based on speedup and known accuracy
        if predicted_speedup > 3.0 and strategy_name in ("nvfp4_blackwell", "fp8_balanced"):
            confidence = "high"
        elif predicted_speedup > 2.0:
            confidence = "medium"
        else:
            confidence = "low"

        return QuantizationRecommendation(
            config=integrated_config,
            mixed_precision=mixed_precision,
            fa_config=fa_config,
            predicted_speedup=predicted_speedup,
            predicted_time_us=predicted_time_us,
            baseline_time_us=baseline_time_us,
            memory_savings_pct=memory_savings_pct,
            kv_cache_bytes=int(kv_cache_bytes),
            expected_quality_loss=mixed_precision.expected_quality_loss,
            confidence=confidence,
            reason=reason,
            bottleneck_analysis=bottleneck_analysis,
        )

    def _select_base_strategy(self, phase: str, seq_len: int, model_config: Dict) -> str:
        """Select base mixed precision strategy based on workload."""
        # Long context: prioritize KV cache quantization
        if seq_len > 8192:
            return "hybrid_long_context"

        # Decode: aggressive weight quantization
        if phase == "decode":
            # Blackwell: use NVFP4
            if "GB10" in self.hw.name or "B200" in self.hw.name:
                return "nvfp4_blackwell"
            else:
                return "w4a16_aggressive"

        # Prefill: more conservative (can be compute-bound)
        if phase == "prefill":
            return "fp8_balanced"

        # Default
        return "fp8_balanced"

    def _select_fa_version(self, seq_len: int, phase: str) -> FlashAttentionVersion:
        """Select FlashAttention version based on hardware and workload."""
        # Blackwell/Hopper: FA3 or FA4
        if "GB10" in self.hw.name or "B200" in self.hw.name or "H100" in self.hw.name:
            # Very long context: FA4 with sparsity
            if seq_len > 16384:
                return FlashAttentionVersion.FA4
            # Standard long context: FA3 with FP8
            return FlashAttentionVersion.FA3

        # Older hardware: FA2
        return FlashAttentionVersion.FA2

    def _generate_justification(self,
                                 strategy: str,
                                 fa_version: FlashAttentionVersion,
                                 phase: str,
                                 seq_len: int,
                                 speedup: float,
                                 fa_metrics) -> str:
        """Generate human-readable justification."""
        parts = []

        # Strategy
        parts.append(f"Selected '{strategy}' strategy for {phase} phase")

        # FlashAttention
        fa_desc = {
            FlashAttentionVersion.FA1: "FA1 (baseline tiled attention)",
            FlashAttentionVersion.FA2: "FA2 (improved parallelism)",
            FlashAttentionVersion.FA3: "FA3 (native FP8 on Hopper/Blackwell)",
            FlashAttentionVersion.FA4: "FA4 (block-sparse + advanced fusion)",
        }
        parts.append(f"Using {fa_desc[fa_version]}")

        # Bottleneck
        if fa_metrics.bottleneck == "memory":
            parts.append(f"Memory-bound workload (AI={fa_metrics.arithmetic_intensity:.1f} < critical AI)")
        else:
            parts.append(f"Compute-bound workload (AI={fa_metrics.arithmetic_intensity:.1f} > critical AI)")

        # Context length
        if seq_len > 8192:
            parts.append(f"Long context ({seq_len} tokens) → aggressive KV cache quantization")

        # Expected result
        parts.append(f"Predicted {speedup:.2f}x speedup vs FP16 baseline")

        return ". ".join(parts) + "."


def demo_integrated_system():
    """Demonstrate the integrated quantization system."""
    from .hardware_registry import BLACKWELL_B10

    print("=" * 90)
    print("Integrated Quantization System Demo")
    print("=" * 90)

    engine = IntegratedQuantizationEngine(BLACKWELL_B10)

    # Test cases: different workloads
    test_cases = [
        {
            "name": "Llama-3 8B Decode (4K context)",
            "model": {"H": 4096, "L": 32, "nh": 32, "nkv": 8, "dh": 128, "dff": 14336},
            "phase": "decode",
            "seq_len": 4096,
        },
        {
            "name": "Llama-3 8B Decode (32K long context)",
            "model": {"H": 4096, "L": 32, "nh": 32, "nkv": 8, "dh": 128, "dff": 14336},
            "phase": "decode",
            "seq_len": 32768,
        },
        {
            "name": "Llama-3 8B Prefill (2K batch)",
            "model": {"H": 4096, "L": 32, "nh": 32, "nkv": 8, "dh": 128, "dff": 14336},
            "phase": "prefill",
            "seq_len": 2048,
        },
    ]

    for test_case in test_cases:
        print(f"\n{'─' * 90}")
        print(f"Test: {test_case['name']}")
        print(f"{'─' * 90}")

        rec = engine.recommend(
            model_config=test_case["model"],
            phase=test_case["phase"],
            seq_len=test_case["seq_len"],
            enable_dynamic=True,
            enable_adaptive=False,
        )

        print(f"\n📊 Recommendation:")
        print(f"  Strategy: {rec.config.strategy_name}")
        print(f"  FlashAttention: {rec.fa_config.version.value}")
        print(f"  QKV Precision: {rec.fa_config.qkv_precision}")
        print(f"  KV Cache Precision: {rec.fa_config.kv_cache_precision}")
        print(f"  Dynamic Quantization: {'✓' if rec.config.enable_dynamic_quant else '✗'}")

        print(f"\n⚡ Performance:")
        print(f"  Predicted Speedup: {rec.predicted_speedup:.2f}x")
        print(f"  Predicted Time: {rec.predicted_time_us / 1000:.2f} ms")
        print(f"  Baseline Time: {rec.baseline_time_us / 1000:.2f} ms")

        print(f"\n💾 Memory:")
        print(f"  KV Cache Size: {rec.kv_cache_bytes / 1e6:.1f} MB")
        print(f"  Memory Savings: {rec.memory_savings_pct:.1f}%")

        print(f"\n🎯 Analysis:")
        print(f"  Confidence: {rec.confidence}")
        print(f"  Expected Quality Loss: {rec.expected_quality_loss}")
        print(f"  Bottleneck: {rec.bottleneck_analysis['overall']}")

        print(f"\n📝 Justification:")
        print(f"  {rec.reason}")

        # Show operator-level breakdown
        print(f"\n🔍 Operator Precision Breakdown (sample):")
        for op in [OperatorType.Q_PROJ, OperatorType.FFN_GATE, OperatorType.KV_CACHE]:
            prec = rec.mixed_precision.get_precision(op)
            print(f"  {op.value:<15}: {prec}")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    demo_integrated_system()
