"""
Unit tests for mixed precision strategies.
"""

import unittest
import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.roofline.mixed_precision import (
    MixedPrecisionConfig,
    OperatorType,
    create_fp16_baseline,
    create_w4a16_aggressive,
    create_nvfp4_blackwell,
    create_fp8_balanced,
    create_hybrid_long_context,
    create_layerwise_gradient,
    get_strategy,
    list_strategies,
)


class TestMixedPrecisionConfig(unittest.TestCase):
    """Test MixedPrecisionConfig class."""

    def test_get_precision_default(self):
        """Test getting precision with default values."""
        config = MixedPrecisionConfig(
            name="Test",
            description="Test config",
            default_activation_precision="FP16",
        )

        # Should return default for unspecified operators
        precision = config.get_precision(OperatorType.Q_PROJ)
        self.assertEqual(precision, "FP16")

    def test_get_precision_operator_level(self):
        """Test getting precision with operator-level config."""
        config = MixedPrecisionConfig(
            name="Test",
            description="Test config",
            operator_precisions={
                OperatorType.Q_PROJ.value: "NVFP4",
                OperatorType.FFN_GATE.value: "FP8_E4M3",
            },
            default_activation_precision="FP16",
        )

        self.assertEqual(config.get_precision(OperatorType.Q_PROJ), "NVFP4")
        self.assertEqual(config.get_precision(OperatorType.FFN_GATE), "FP8_E4M3")
        self.assertEqual(config.get_precision(OperatorType.K_PROJ), "FP16")  # default

    def test_get_precision_layer_override(self):
        """Test getting precision with layer-specific overrides."""
        config = MixedPrecisionConfig(
            name="Test",
            description="Test config",
            operator_precisions={
                OperatorType.Q_PROJ.value: "FP16",
            },
            layer_overrides={
                5: {OperatorType.Q_PROJ.value: "FP8_E4M3"},
                10: {OperatorType.Q_PROJ.value: "NVFP4"},
            },
            default_activation_precision="FP16",
        )

        # Layer 0: uses operator-level config
        self.assertEqual(config.get_precision(OperatorType.Q_PROJ, layer_idx=0), "FP16")

        # Layer 5: uses layer override
        self.assertEqual(config.get_precision(OperatorType.Q_PROJ, layer_idx=5), "FP8_E4M3")

        # Layer 10: uses layer override
        self.assertEqual(config.get_precision(OperatorType.Q_PROJ, layer_idx=10), "NVFP4")

    def test_get_kv_precision(self):
        """Test getting KV cache precision."""
        config = MixedPrecisionConfig(
            name="Test",
            description="Test config",
            kv_cache_precision="NVFP4_KV",
        )

        self.assertEqual(config.get_kv_precision(), "NVFP4_KV")

    def test_get_kv_precision_fallback(self):
        """Test KV cache precision fallback."""
        config = MixedPrecisionConfig(
            name="Test",
            description="Test config",
            operator_precisions={
                OperatorType.KV_CACHE.value: "FP8_E4M3",
            },
            default_activation_precision="FP16",
        )

        # Should fall back to operator precision
        self.assertEqual(config.get_kv_precision(), "FP8_E4M3")


class TestPresetStrategies(unittest.TestCase):
    """Test preset mixed precision strategies."""

    def test_fp16_baseline(self):
        """Test FP16 baseline strategy."""
        config = create_fp16_baseline()

        self.assertEqual(config.name, "FP16 Baseline")
        self.assertEqual(config.expected_speedup, 1.0)
        self.assertEqual(config.memory_savings_pct, 0.0)

        # All operators should be FP16
        for op_type in OperatorType:
            self.assertEqual(config.get_precision(op_type), "FP16")

    def test_w4a16_aggressive(self):
        """Test W4A16 aggressive strategy."""
        config = create_w4a16_aggressive()

        self.assertEqual(config.name, "W4A16 Aggressive")
        self.assertEqual(config.expected_speedup, 4.0)
        self.assertEqual(config.memory_savings_pct, 75.0)

        # Weights should be INT4
        self.assertEqual(config.get_precision(OperatorType.Q_PROJ), "INT4")
        self.assertEqual(config.get_precision(OperatorType.FFN_GATE), "INT4")

        # Activations should be FP16
        self.assertEqual(config.get_precision(OperatorType.QK_MATMUL), "FP16")

    def test_nvfp4_blackwell(self):
        """Test NVFP4 Blackwell strategy."""
        config = create_nvfp4_blackwell()

        self.assertEqual(config.name, "NVFP4 Blackwell")
        self.assertEqual(config.expected_speedup, 4.0)
        self.assertLess(config.memory_savings_pct, 75.0)  # Less than INT4 due to overhead

        # Weights should be NVFP4
        self.assertEqual(config.get_precision(OperatorType.Q_PROJ), "NVFP4")
        self.assertEqual(config.get_precision(OperatorType.FFN_GATE), "NVFP4")

        # KV cache should be NVFP4_KV
        self.assertEqual(config.get_kv_precision(), "NVFP4_KV")

    def test_fp8_balanced(self):
        """Test FP8 balanced strategy."""
        config = create_fp8_balanced()

        self.assertEqual(config.name, "FP8 Balanced")
        self.assertEqual(config.expected_speedup, 2.0)
        self.assertEqual(config.memory_savings_pct, 50.0)

        # Most operators should be FP8
        self.assertEqual(config.get_precision(OperatorType.Q_PROJ), "FP8_E4M3")

        # Softmax should stay FP16 for stability
        self.assertEqual(config.get_precision(OperatorType.SOFTMAX), "FP16")

    def test_hybrid_long_context(self):
        """Test hybrid long context strategy."""
        config = create_hybrid_long_context()

        self.assertEqual(config.name, "Hybrid Long Context")

        # FFN should be aggressive (NVFP4)
        self.assertEqual(config.get_precision(OperatorType.FFN_GATE), "NVFP4")

        # KV cache should be NVFP4_KV
        self.assertEqual(config.get_kv_precision(), "NVFP4_KV")

    def test_layerwise_gradient(self):
        """Test layerwise gradient strategy."""
        config = create_layerwise_gradient(num_layers=32)

        self.assertEqual(config.name, "Layerwise Gradient")

        # Early layers: FP16
        self.assertEqual(config.get_precision(OperatorType.Q_PROJ, layer_idx=5), "FP16")

        # Middle layers: FP8
        self.assertEqual(config.get_precision(OperatorType.Q_PROJ, layer_idx=15), "FP8_E4M3")

        # Late layers: NVFP4
        self.assertEqual(config.get_precision(OperatorType.Q_PROJ, layer_idx=28), "NVFP4")


class TestStrategyRegistry(unittest.TestCase):
    """Test strategy registry functions."""

    def test_list_strategies(self):
        """Test listing available strategies."""
        strategies = list_strategies()

        self.assertIsInstance(strategies, list)
        self.assertGreater(len(strategies), 0)
        self.assertIn("fp16_baseline", strategies)
        self.assertIn("nvfp4_blackwell", strategies)

    def test_get_strategy(self):
        """Test getting strategy by name."""
        config = get_strategy("fp16_baseline")

        self.assertIsInstance(config, MixedPrecisionConfig)
        self.assertEqual(config.name, "FP16 Baseline")

    def test_get_strategy_invalid(self):
        """Test getting invalid strategy raises error."""
        with self.assertRaises(ValueError):
            get_strategy("invalid_strategy_name")


class TestOperatorType(unittest.TestCase):
    """Test OperatorType enum."""

    def test_operator_types_defined(self):
        """Test that all expected operator types are defined."""
        expected_types = [
            "Q_PROJ", "K_PROJ", "V_PROJ", "O_PROJ",
            "QK_MATMUL", "SV_MATMUL", "SOFTMAX",
            "FFN_GATE", "FFN_UP", "FFN_DOWN",
            "RMSNORM", "KV_CACHE",
        ]

        for type_name in expected_types:
            self.assertTrue(hasattr(OperatorType, type_name))

    def test_operator_type_values(self):
        """Test operator type string values."""
        self.assertEqual(OperatorType.Q_PROJ.value, "q_proj")
        self.assertEqual(OperatorType.FFN_GATE.value, "ffn_gate")
        self.assertEqual(OperatorType.KV_CACHE.value, "kv_cache")


if __name__ == "__main__":
    unittest.main()
