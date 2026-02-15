"""
Mixed Precision Strategy System

Enables per-operator precision configuration for optimal performance/accuracy tradeoff.
Supports layer-wise, operator-wise, and component-wise quantization strategies.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from enum import Enum


class OperatorType(Enum):
    """Transformer operator types with distinct memory/compute characteristics"""
    # Attention projections (memory-bound in decode)
    Q_PROJ = "q_proj"
    K_PROJ = "k_proj"
    V_PROJ = "v_proj"
    O_PROJ = "o_proj"

    # Attention compute (can be compute-bound in prefill)
    QK_MATMUL = "qk_matmul"
    SV_MATMUL = "sv_matmul"
    SOFTMAX = "softmax"

    # FFN projections (memory-bound in decode, 2/3 of total compute)
    FFN_GATE = "ffn_gate"
    FFN_UP = "ffn_up"
    FFN_DOWN = "ffn_down"

    # Activations
    SILU = "silu"
    GELU = "gelu"

    # Norms (always memory-bound)
    RMSNORM = "rmsnorm"
    LAYERNORM = "layernorm"

    # Other
    ROPE = "rope"
    KV_CACHE = "kv_cache"
    EMBEDDINGS = "embeddings"


@dataclass
class MixedPrecisionConfig:
    """
    Mixed precision configuration for transformer layers.

    Allows different precision for each operator type, enabling fine-grained
    optimization of the speed/accuracy tradeoff.
    """
    name: str
    description: str

    # Precision mapping: operator_type -> precision format
    operator_precisions: Dict[str, str] = field(default_factory=dict)

    # Layer-specific overrides: layer_idx -> {operator_type -> precision}
    layer_overrides: Dict[int, Dict[str, str]] = field(default_factory=dict)

    # Activation precision (default for all activations if not specified)
    default_activation_precision: str = "FP16"

    # Accumulator precision (for matmuls)
    accumulator_precision: str = "FP32"

    # KV cache precision (critical for long context)
    kv_cache_precision: Optional[str] = None

    # Metadata
    expected_speedup: float = 1.0
    expected_quality_loss: str = "minimal"
    memory_savings_pct: float = 0.0

    def get_precision(self, operator_type: OperatorType, layer_idx: Optional[int] = None) -> str:
        """Get precision for a specific operator, considering layer overrides."""
        op_name = operator_type.value

        # Check layer-specific override
        if layer_idx is not None and layer_idx in self.layer_overrides:
            if op_name in self.layer_overrides[layer_idx]:
                return self.layer_overrides[layer_idx][op_name]

        # Check operator-level config
        if op_name in self.operator_precisions:
            return self.operator_precisions[op_name]

        # Fall back to default activation precision
        return self.default_activation_precision

    def get_kv_precision(self) -> str:
        """Get KV cache precision, with fallback."""
        if self.kv_cache_precision:
            return self.kv_cache_precision
        return self.operator_precisions.get(OperatorType.KV_CACHE.value, self.default_activation_precision)


# ═══════════════════════════════════════════════
#  PRESET MIXED PRECISION STRATEGIES
# ═══════════════════════════════════════════════

def create_fp16_baseline() -> MixedPrecisionConfig:
    """FP16 everywhere - baseline configuration."""
    return MixedPrecisionConfig(
        name="FP16 Baseline",
        description="Full FP16 precision, no quantization",
        operator_precisions={op.value: "FP16" for op in OperatorType},
        default_activation_precision="FP16",
        accumulator_precision="FP32",
        expected_speedup=1.0,
        expected_quality_loss="none",
        memory_savings_pct=0.0,
    )


def create_w4a16_aggressive() -> MixedPrecisionConfig:
    """
    W4A16: All weights INT4, activations FP16.
    Maximizes speed for decode, moderate quality loss.
    """
    weight_precision = "INT4"
    activation_precision = "FP16"

    operator_precisions = {
        # All weight matrices use INT4
        OperatorType.Q_PROJ.value: weight_precision,
        OperatorType.K_PROJ.value: weight_precision,
        OperatorType.V_PROJ.value: weight_precision,
        OperatorType.O_PROJ.value: weight_precision,
        OperatorType.FFN_GATE.value: weight_precision,
        OperatorType.FFN_UP.value: weight_precision,
        OperatorType.FFN_DOWN.value: weight_precision,
        OperatorType.EMBEDDINGS.value: weight_precision,

        # Activations and compute stay FP16
        OperatorType.QK_MATMUL.value: activation_precision,
        OperatorType.SV_MATMUL.value: activation_precision,
        OperatorType.SOFTMAX.value: activation_precision,
        OperatorType.KV_CACHE.value: activation_precision,
    }

    return MixedPrecisionConfig(
        name="W4A16 Aggressive",
        description="INT4 weights, FP16 activations - max speed for decode",
        operator_precisions=operator_precisions,
        default_activation_precision=activation_precision,
        accumulator_precision="FP32",
        kv_cache_precision="FP16",
        expected_speedup=4.0,  # 4x faster weights loading
        expected_quality_loss="2-5% perplexity increase",
        memory_savings_pct=75.0,  # Weights 4x smaller
    )


def create_nvfp4_blackwell() -> MixedPrecisionConfig:
    """
    NVFP4 W4A16: Blackwell-optimized with native FP4 support.
    Better accuracy than INT4 at similar speed.
    """
    weight_precision = "NVFP4"
    activation_precision = "FP16"

    operator_precisions = {
        OperatorType.Q_PROJ.value: weight_precision,
        OperatorType.K_PROJ.value: weight_precision,
        OperatorType.V_PROJ.value: weight_precision,
        OperatorType.O_PROJ.value: weight_precision,
        OperatorType.FFN_GATE.value: weight_precision,
        OperatorType.FFN_UP.value: weight_precision,
        OperatorType.FFN_DOWN.value: weight_precision,
        OperatorType.EMBEDDINGS.value: weight_precision,
        OperatorType.QK_MATMUL.value: activation_precision,
        OperatorType.SV_MATMUL.value: activation_precision,
        OperatorType.SOFTMAX.value: activation_precision,
        OperatorType.KV_CACHE.value: "NVFP4_KV",  # Specialized KV cache format
    }

    return MixedPrecisionConfig(
        name="NVFP4 Blackwell",
        description="NVFP4 weights + FP16 activations, 1000 TFLOPS on GB10",
        operator_precisions=operator_precisions,
        default_activation_precision=activation_precision,
        accumulator_precision="FP32",
        kv_cache_precision="NVFP4_KV",
        expected_speedup=4.0,
        expected_quality_loss="<1% perplexity increase",
        memory_savings_pct=72.0,
    )


def create_fp8_balanced() -> MixedPrecisionConfig:
    """
    FP8 everywhere: 2x speedup with minimal quality loss.
    Best for Hopper/Blackwell with native FP8 tensor cores.
    """
    precision = "FP8_E4M3"

    operator_precisions = {
        OperatorType.Q_PROJ.value: precision,
        OperatorType.K_PROJ.value: precision,
        OperatorType.V_PROJ.value: precision,
        OperatorType.O_PROJ.value: precision,
        OperatorType.FFN_GATE.value: precision,
        OperatorType.FFN_UP.value: precision,
        OperatorType.FFN_DOWN.value: precision,
        OperatorType.QK_MATMUL.value: precision,
        OperatorType.SV_MATMUL.value: precision,
        OperatorType.SOFTMAX.value: "FP16",  # Keep softmax in FP16 for stability
        OperatorType.KV_CACHE.value: precision,
        OperatorType.EMBEDDINGS.value: precision,
    }

    return MixedPrecisionConfig(
        name="FP8 Balanced",
        description="FP8 E4M3 everywhere (except softmax), 2x speedup",
        operator_precisions=operator_precisions,
        default_activation_precision=precision,
        accumulator_precision="FP32",
        kv_cache_precision=precision,
        expected_speedup=2.0,
        expected_quality_loss="<1% perplexity increase",
        memory_savings_pct=50.0,
    )


def create_hybrid_long_context() -> MixedPrecisionConfig:
    """
    Hybrid strategy optimized for long context (>8K tokens).
    Aggressive KV cache quantization, conservative elsewhere.
    """
    operator_precisions = {
        # Weights: moderate quantization
        OperatorType.Q_PROJ.value: "FP8_E4M3",
        OperatorType.K_PROJ.value: "FP8_E4M3",
        OperatorType.V_PROJ.value: "FP8_E4M3",
        OperatorType.O_PROJ.value: "FP8_E4M3",
        OperatorType.FFN_GATE.value: "NVFP4",  # FFN gets aggressive (2/3 of compute)
        OperatorType.FFN_UP.value: "NVFP4",
        OperatorType.FFN_DOWN.value: "NVFP4",

        # Activations: FP16 for quality
        OperatorType.QK_MATMUL.value: "FP16",
        OperatorType.SV_MATMUL.value: "FP16",
        OperatorType.SOFTMAX.value: "FP16",

        # KV cache: aggressive quantization (bottleneck at long context!)
        OperatorType.KV_CACHE.value: "NVFP4_KV",

        OperatorType.EMBEDDINGS.value: "FP16",
    }

    return MixedPrecisionConfig(
        name="Hybrid Long Context",
        description="Optimized for >8K context: NVFP4 KV cache + mixed weights",
        operator_precisions=operator_precisions,
        default_activation_precision="FP16",
        accumulator_precision="FP32",
        kv_cache_precision="NVFP4_KV",
        expected_speedup=3.0,  # Depends on context length
        expected_quality_loss="1-2% perplexity increase",
        memory_savings_pct=60.0,
    )


def create_layerwise_gradient(num_layers: int = 32) -> MixedPrecisionConfig:
    """
    Layer-wise quantization gradient: early layers more precise, later layers more aggressive.
    Based on observation that later layers are more robust to quantization.
    """
    base_config = MixedPrecisionConfig(
        name="Layerwise Gradient",
        description="Progressive quantization: FP16 → FP8 → NVFP4 across layers",
        default_activation_precision="FP16",
        accumulator_precision="FP32",
        kv_cache_precision="FP8_E4M3",
    )

    # Define precision tiers
    for layer_idx in range(num_layers):
        progress = layer_idx / num_layers

        if progress < 0.33:
            # First 1/3: FP16 (most sensitive)
            weight_prec = "FP16"
        elif progress < 0.67:
            # Middle 1/3: FP8 (moderate)
            weight_prec = "FP8_E4M3"
        else:
            # Final 1/3: NVFP4 (most aggressive)
            weight_prec = "NVFP4"

        base_config.layer_overrides[layer_idx] = {
            OperatorType.Q_PROJ.value: weight_prec,
            OperatorType.K_PROJ.value: weight_prec,
            OperatorType.V_PROJ.value: weight_prec,
            OperatorType.O_PROJ.value: weight_prec,
            OperatorType.FFN_GATE.value: weight_prec,
            OperatorType.FFN_UP.value: weight_prec,
            OperatorType.FFN_DOWN.value: weight_prec,
        }

    base_config.expected_speedup = 2.5
    base_config.expected_quality_loss = "0.5-1% perplexity increase"
    base_config.memory_savings_pct = 55.0

    return base_config


# ═══════════════════════════════════════════════
#  STRATEGY REGISTRY
# ═══════════════════════════════════════════════

STRATEGY_REGISTRY = {
    "fp16_baseline": create_fp16_baseline,
    "w4a16_aggressive": create_w4a16_aggressive,
    "nvfp4_blackwell": create_nvfp4_blackwell,
    "fp8_balanced": create_fp8_balanced,
    "hybrid_long_context": create_hybrid_long_context,
    "layerwise_gradient": lambda: create_layerwise_gradient(32),
}


def get_strategy(name: str, **kwargs) -> MixedPrecisionConfig:
    """Get a mixed precision strategy by name."""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[name](**kwargs) if kwargs else STRATEGY_REGISTRY[name]()


def list_strategies() -> List[str]:
    """List available mixed precision strategies."""
    return list(STRATEGY_REGISTRY.keys())


def compare_strategies(strategies: Optional[List[str]] = None) -> None:
    """Print comparison table of strategies."""
    if strategies is None:
        strategies = list_strategies()

    print("=" * 100)
    print("Mixed Precision Strategy Comparison")
    print("=" * 100)
    print(f"{'Strategy':<25} {'Speedup':<10} {'Quality Loss':<30} {'Memory Savings':<15}")
    print("-" * 100)

    for name in strategies:
        try:
            config = get_strategy(name)
            print(f"{config.name:<25} {config.expected_speedup:<10.1f}x "
                  f"{config.expected_quality_loss:<30} {config.memory_savings_pct:<15.1f}%")
        except Exception as e:
            print(f"{name:<25} ERROR: {e}")
    print("=" * 100)


# ═══════════════════════════════════════════════
#  ANALYSIS UTILITIES
# ═══════════════════════════════════════════════

def analyze_strategy_per_operator(config: MixedPrecisionConfig,
                                   calculator,
                                   model_config: Dict) -> Dict:
    """
    Analyze performance impact of mixed precision strategy operator-by-operator.

    Returns breakdown of time/bytes/speedup per operator type.
    """
    from .calculator_shell import bytes_per_element

    H = model_config.get("H", 4096)
    B = model_config.get("B", 1)
    T = model_config.get("T", 1)  # Decode
    dff = model_config.get("dff", 14336)

    results = {}

    # Analyze projections (GEMMs)
    for op_type in [OperatorType.Q_PROJ, OperatorType.K_PROJ, OperatorType.V_PROJ,
                    OperatorType.O_PROJ, OperatorType.FFN_GATE, OperatorType.FFN_UP,
                    OperatorType.FFN_DOWN]:
        prec = config.get_precision(op_type)

        # Determine shape
        if "FFN" in op_type.value:
            N, K = dff, H
            if op_type == OperatorType.FFN_DOWN:
                N, K = H, dff
        else:
            N, K = H, H

        # Predict performance
        pred = calculator.predict_gemv(N, K, prec)

        # Compare to FP16 baseline
        pred_fp16 = calculator.predict_gemv(N, K, "FP16")
        speedup = pred_fp16["predicted_time_us"] / pred["predicted_time_us"]

        results[op_type.value] = {
            "precision": prec,
            "time_us": pred["predicted_time_us"],
            "bytes": pred["bytes"],
            "speedup_vs_fp16": speedup,
            "bottleneck": pred["bottleneck"],
        }

    return results


if __name__ == "__main__":
    # Demo: compare all strategies
    compare_strategies()

    print("\n" + "=" * 100)
    print("Example: NVFP4 Blackwell Strategy Details")
    print("=" * 100)

    config = create_nvfp4_blackwell()
    print(f"\nName: {config.name}")
    print(f"Description: {config.description}")
    print(f"Expected speedup: {config.expected_speedup}x")
    print(f"Expected quality loss: {config.expected_quality_loss}")
    print(f"Memory savings: {config.memory_savings_pct}%")
    print(f"\nKV cache precision: {config.get_kv_precision()}")

    print("\nOperator precision breakdown:")
    for op_type in OperatorType:
        prec = config.get_precision(op_type)
        print(f"  {op_type.value:<20} → {prec}")
