"""
Comprehensive Quantization Demo

Demonstrates:
1. Mixed precision strategies comparison
2. Dynamic quantization with runtime adaptation
3. FlashAttention 1/2/3/4 performance comparison
4. Integrated quantization engine recommendations
"""

import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.roofline.mixed_precision import (
    compare_strategies,
    create_nvfp4_blackwell,
    create_layerwise_gradient,
    OperatorType,
)
from src.roofline.dynamic_quant import (
    DynamicQuantizer,
    AdaptivePrecisionSelector,
    ActivationStats,
    DynamicQuantConfig,
    AdaptiveQuantConfig,
)
from src.roofline.flash_attention import (
    FlashAttentionVersion,
    FlashAttentionRoofline,
    FlashAttentionConfig,
    compare_fa_versions,
)
from src.roofline.quantization_integration import (
    IntegratedQuantizationEngine,
)
from src.roofline.hardware_registry import BLACKWELL_B10


def demo_1_mixed_precision():
    """Demo 1: Mixed Precision Strategies"""
    print("\n" + "=" * 100)
    print("DEMO 1: Mixed Precision Strategies")
    print("=" * 100)

    # Show all available strategies
    compare_strategies()

    # Deep dive into NVFP4 Blackwell strategy
    print("\n" + "─" * 100)
    print("Deep Dive: NVFP4 Blackwell Strategy")
    print("─" * 100)

    config = create_nvfp4_blackwell()
    print(f"\nDescription: {config.description}")
    print(f"Expected speedup: {config.expected_speedup}x")
    print(f"Expected quality loss: {config.expected_quality_loss}")
    print(f"Memory savings: {config.memory_savings_pct}%\n")

    # Show per-operator precision
    print("Operator Precision Mapping:")
    print(f"{'Operator':<20} {'Precision':<15}")
    print("─" * 35)
    for op_type in OperatorType:
        prec = config.get_precision(op_type)
        print(f"{op_type.value:<20} {prec:<15}")

    # Layerwise gradient strategy
    print("\n" + "─" * 100)
    print("Layer-wise Gradient Strategy (32 layers)")
    print("─" * 100)

    layerwise = create_layerwise_gradient(32)
    print("\nProgressive quantization across layers:")
    print(f"{'Layer Range':<15} {'Precision':<15}")
    print("─" * 30)
    print(f"{'0-10 (early)':<15} {'FP16':<15}")
    print(f"{'11-21 (middle)':<15} {'FP8_E4M3':<15}")
    print(f"{'22-31 (late)':<15} {'NVFP4':<15}")


def demo_2_dynamic_quantization():
    """Demo 2: Dynamic Quantization"""
    print("\n\n" + "=" * 100)
    print("DEMO 2: Dynamic Quantization with Runtime Adaptation")
    print("=" * 100)

    # Create dynamic quantizer
    config = DynamicQuantConfig(
        precision="FP8_E4M3",
        granularity="per_tensor",
        calibration_mode="percentile",
        fallback_to_fp16=True,
    )
    quantizer = DynamicQuantizer(config)

    # Simulate different activation patterns
    test_cases = [
        {
            "name": "Normal Activations",
            "stats": ActivationStats(
                absmax=10.0, mean=0.0, std=2.0,
                percentile_99=8.0, percentile_999=9.5,
                shape=(1024, 4096)
            ),
        },
        {
            "name": "Heavy Tailed (outliers)",
            "stats": ActivationStats(
                absmax=100.0, mean=0.0, std=2.0,
                percentile_99=8.0, percentile_999=15.0,
                shape=(1024, 4096)
            ),
        },
        {
            "name": "Uniform Distribution",
            "stats": ActivationStats(
                absmax=5.0, mean=2.5, std=1.0,
                percentile_99=4.8, percentile_999=4.95,
                shape=(1024, 4096)
            ),
        },
    ]

    print("\nDynamic Scale Computation:")
    print(f"{'Scenario':<25} {'Outlier Ratio':<15} {'Scale':<10} {'Precision':<15} {'Fallback?':<10}")
    print("─" * 85)

    for case in test_cases:
        stats = case["stats"]
        scale, prec = quantizer.compute_scale(stats, "FP8_E4M3")
        fallback = "Yes" if prec == "FP16" else "No"
        print(f"{case['name']:<25} {stats.outlier_ratio:<15.2f} {scale:<10.2f} {prec:<15} {fallback:<10}")

    # Adaptive precision selector
    print("\n" + "─" * 100)
    print("Adaptive Precision Selection")
    print("─" * 100)

    adaptive_config = AdaptiveQuantConfig(
        precision_candidates=["NVFP4", "FP8_E4M3", "FP16"],
        high_outlier_threshold=2.0,
        medium_outlier_threshold=1.5,
    )
    selector = AdaptivePrecisionSelector(adaptive_config)

    print("\nContext-Aware Precision Selection:")
    print(f"{'Scenario':<40} {'Outliers':<12} {'Selected':<15} {'Reason':<40}")
    print("─" * 107)

    scenarios = [
        (test_cases[0]["stats"], {"phase": "decode", "seq_len": 100}, "Normal decode"),
        (test_cases[1]["stats"], {"phase": "decode", "seq_len": 100}, "Outlier-heavy decode"),
        (test_cases[0]["stats"], {"is_kv_cache": True, "seq_len": 16384}, "Long context KV cache"),
        (test_cases[2]["stats"], {"phase": "prefill", "seq_len": 2048}, "Prefill"),
    ]

    for stats, ctx, desc in scenarios:
        prec = selector.select_precision("test_op", stats, ctx)
        print(f"{desc:<40} {stats.outlier_ratio:<12.2f} {prec:<15} {ctx}")


def demo_3_flashattention():
    """Demo 3: FlashAttention 1/2/3/4 Comparison"""
    print("\n\n" + "=" * 100)
    print("DEMO 3: FlashAttention Version Comparison")
    print("=" * 100)

    # Standard workload
    print("\nWorkload: Llama-3 8B, 4K context, decode")
    print("─" * 100)

    results = compare_fa_versions(
        batch=1,
        num_heads=32,
        seq_len=4096,
        head_dim=128,
        hardware_key="b10"
    )

    print(f"\n{'Version':<10} {'Precision':<15} {'KV Cache':<15} {'Time (μs)':<12} {'AI':<8} {'Bottleneck':<12} {'Features':<30}")
    print("─" * 110)

    baseline_time = results["fa1"].predicted_time_us

    for version_name, metrics in results.items():
        speedup = baseline_time / metrics.predicted_time_us
        features = []
        if metrics.uses_fp8:
            features.append("FP8")
        if metrics.uses_sparsity:
            features.append("Sparse")
        features_str = ", ".join(features) if features else "Standard"

        # Get config to show precision
        from src.roofline.flash_attention import recommend_fa_precision, FlashAttentionVersion
        fa_ver = FlashAttentionVersion(version_name)
        cfg = recommend_fa_precision(fa_ver, 4096, "GB10", "decode")

        print(f"{version_name.upper():<10} {cfg.qkv_precision:<15} {cfg.kv_cache_precision:<15} "
              f"{metrics.predicted_time_us:<12.1f} {metrics.arithmetic_intensity:<8.2f} "
              f"{metrics.bottleneck:<12} {features_str:<30}")

    # Speedup summary
    print("\n" + "─" * 100)
    print("Speedup vs FA1:")
    for version_name, metrics in results.items():
        speedup = baseline_time / metrics.predicted_time_us
        bar = "█" * int(speedup * 10)
        print(f"  {version_name.upper():<6} {speedup:>5.2f}x {bar}")

    # Long context analysis
    print("\n" + "─" * 100)
    print("Long Context Performance (32K tokens)")
    print("─" * 100)

    results_long = compare_fa_versions(
        batch=1,
        num_heads=32,
        seq_len=32768,
        head_dim=128,
        hardware_key="b10"
    )

    print(f"\n{'Version':<10} {'Time (ms)':<12} {'KV Cache (MB)':<15} {'Memory Efficiency':<20}")
    print("─" * 60)

    for version_name, metrics in results_long.items():
        time_ms = metrics.predicted_time_us / 1000
        kv_mb = metrics.kv_cache_bytes / 1e6
        mem_eff_pct = metrics.memory_efficiency * 100

        print(f"{version_name.upper():<10} {time_ms:<12.2f} {kv_mb:<15.1f} {mem_eff_pct:<20.1f}%")


def demo_4_integrated_system():
    """Demo 4: Integrated Quantization Engine"""
    print("\n\n" + "=" * 100)
    print("DEMO 4: Integrated Quantization Engine - Complete Recommendations")
    print("=" * 100)

    engine = IntegratedQuantizationEngine(BLACKWELL_B10)

    # Model configs
    llama3_8b = {
        "name": "Llama-3 8B",
        "H": 4096, "L": 32, "nh": 32, "nkv": 8, "dh": 128, "dff": 14336
    }

    # Different scenarios
    scenarios = [
        {
            "desc": "Standard Decode (4K context)",
            "model": llama3_8b,
            "phase": "decode",
            "seq_len": 4096,
        },
        {
            "desc": "Long Context Decode (32K context)",
            "model": llama3_8b,
            "phase": "decode",
            "seq_len": 32768,
        },
        {
            "desc": "Prefill (2K batch)",
            "model": llama3_8b,
            "phase": "prefill",
            "seq_len": 2048,
        },
    ]

    for scenario in scenarios:
        print(f"\n{'═' * 100}")
        print(f"Scenario: {llama3_8b['name']} - {scenario['desc']}")
        print(f"{'═' * 100}")

        rec = engine.recommend(
            model_config=scenario["model"],
            phase=scenario["phase"],
            seq_len=scenario["seq_len"],
            enable_dynamic=True,
            enable_adaptive=False,
        )

        # Header
        print(f"\n🎯 RECOMMENDATION (Confidence: {rec.confidence.upper()})")
        print("─" * 100)

        # Strategy
        print(f"\n📋 Strategy Configuration:")
        print(f"  Base Strategy:        {rec.config.strategy_name}")
        print(f"  FlashAttention:       {rec.fa_config.version.value.upper()}")
        print(f"  QKV Precision:        {rec.fa_config.qkv_precision}")
        print(f"  KV Cache Precision:   {rec.fa_config.kv_cache_precision}")
        print(f"  Dynamic Quant:        {'Enabled' if rec.config.enable_dynamic_quant else 'Disabled'}")
        if rec.config.enable_dynamic_quant:
            dyn_cfg = rec.config.dynamic_quant_config
            print(f"    - Granularity:      {dyn_cfg.granularity}")
            print(f"    - Calibration:      {dyn_cfg.calibration_mode}")
            print(f"    - Fallback to FP16: {dyn_cfg.fallback_to_fp16}")

        # Performance
        print(f"\n⚡ Performance Prediction:")
        print(f"  Baseline (FP16):      {rec.baseline_time_us / 1000:.2f} ms")
        print(f"  Optimized:            {rec.predicted_time_us / 1000:.2f} ms")
        print(f"  Speedup:              {rec.predicted_speedup:.2f}x")
        print(f"  Tokens/sec (decode):  {1e6 / rec.predicted_time_us:.1f}")

        # Memory
        print(f"\n💾 Memory Analysis:")
        print(f"  KV Cache Size:        {rec.kv_cache_bytes / 1e6:.1f} MB")
        print(f"  Memory Savings:       {rec.memory_savings_pct:.1f}%")

        # Quality
        print(f"\n🎨 Quality Impact:")
        print(f"  Expected Loss:        {rec.expected_quality_loss}")

        # Bottleneck
        print(f"\n🔍 Bottleneck Analysis:")
        print(f"  Attention:            {rec.bottleneck_analysis['attention']}-bound")
        print(f"  FFN:                  {rec.bottleneck_analysis['ffn']}-bound")
        print(f"  Overall:              {rec.bottleneck_analysis['overall']}")

        # Justification
        print(f"\n📝 Justification:")
        print(f"  {rec.reason}")

        # Operator breakdown (sample)
        print(f"\n🔧 Operator Precision (sample):")
        sample_ops = [
            OperatorType.Q_PROJ,
            OperatorType.K_PROJ,
            OperatorType.V_PROJ,
            OperatorType.O_PROJ,
            OperatorType.FFN_GATE,
            OperatorType.FFN_UP,
            OperatorType.FFN_DOWN,
            OperatorType.KV_CACHE,
        ]
        for op in sample_ops:
            prec = rec.mixed_precision.get_precision(op)
            print(f"  {op.value:<15} → {prec}")


def main():
    """Run all demos"""
    print("╔" + "═" * 98 + "╗")
    print("║" + " " * 30 + "QUANTIZATION SYSTEM DEMONSTRATION" + " " * 35 + "║")
    print("║" + " " * 98 + "║")
    print("║  Mixed Precision • Dynamic Quantization • FlashAttention • Integrated Engine  ║")
    print("╚" + "═" * 98 + "╝")

    # Run all demos
    demo_1_mixed_precision()
    demo_2_dynamic_quantization()
    demo_3_flashattention()
    demo_4_integrated_system()

    print("\n\n" + "=" * 100)
    print("DEMONSTRATION COMPLETE")
    print("=" * 100)
    print("\nKey Takeaways:")
    print("  1. Mixed precision strategies offer different speed/quality tradeoffs")
    print("  2. Dynamic quantization adapts to runtime activation distributions")
    print("  3. FlashAttention versions provide progressive improvements (FA1 → FA4)")
    print("  4. Integrated engine combines all techniques for optimal recommendations")
    print("\nNext Steps:")
    print("  - Run benchmarks on real hardware to validate predictions")
    print("  - Fine-tune quantization parameters for specific models")
    print("  - Implement custom strategies for domain-specific workloads")
    print("=" * 100)


if __name__ == "__main__":
    main()
