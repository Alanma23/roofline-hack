"""
Unit tests for integrated quantization system.
"""

import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.roofline.quantization_integration import (
    IntegratedQuantConfig,
    IntegratedQuantizationEngine,
    QuantizationRecommendation,
)
from src.roofline.flash_attention import FlashAttentionVersion
from src.roofline.hardware_registry import BLACKWELL_B10


class TestIntegratedQuantConfig(unittest.TestCase):
    """Test IntegratedQuantConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = IntegratedQuantConfig()

        self.assertEqual(config.strategy_name, "nvfp4_blackwell")
        self.assertEqual(config.fa_version, FlashAttentionVersion.FA3)
        self.assertTrue(config.use_flash_attention)
        self.assertTrue(config.enable_dynamic_quant)
        self.assertFalse(config.enable_adaptive_precision)
        self.assertEqual(config.phase, "decode")

    def test_custom_config(self):
        """Test custom configuration."""
        config = IntegratedQuantConfig(
            strategy_name="fp8_balanced",
            fa_version=FlashAttentionVersion.FA4,
            enable_dynamic_quant=False,
            enable_adaptive_precision=True,
            phase="prefill",
            seq_len=8192,
        )

        self.assertEqual(config.strategy_name, "fp8_balanced")
        self.assertEqual(config.fa_version, FlashAttentionVersion.FA4)
        self.assertFalse(config.enable_dynamic_quant)
        self.assertTrue(config.enable_adaptive_precision)
        self.assertEqual(config.phase, "prefill")
        self.assertEqual(config.seq_len, 8192)


class TestIntegratedQuantizationEngine(unittest.TestCase):
    """Test IntegratedQuantizationEngine."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = IntegratedQuantizationEngine(BLACKWELL_B10)
        self.llama3_8b = {
            "H": 4096,
            "L": 32,
            "nh": 32,
            "nkv": 8,
            "dh": 128,
            "dff": 14336,
        }

    def test_recommend_basic(self):
        """Test basic recommendation."""
        rec = self.engine.recommend(
            model_config=self.llama3_8b,
            phase="decode",
            seq_len=4096,
        )

        self.assertIsInstance(rec, QuantizationRecommendation)
        self.assertIsNotNone(rec.config)
        self.assertIsNotNone(rec.mixed_precision)
        self.assertIsNotNone(rec.fa_config)
        self.assertGreater(rec.predicted_speedup, 1.0)

    def test_recommend_decode(self):
        """Test recommendation for decode phase."""
        rec = self.engine.recommend(
            model_config=self.llama3_8b,
            phase="decode",
            seq_len=4096,
        )

        # Decode should use aggressive quantization
        self.assertIn(rec.config.strategy_name, ["nvfp4_blackwell", "w4a16_aggressive"])
        self.assertIn(rec.fa_config.version, [FlashAttentionVersion.FA3, FlashAttentionVersion.FA4])

    def test_recommend_prefill(self):
        """Test recommendation for prefill phase."""
        rec = self.engine.recommend(
            model_config=self.llama3_8b,
            phase="prefill",
            seq_len=2048,
        )

        # Prefill should be more conservative
        self.assertEqual(rec.config.strategy_name, "fp8_balanced")

    def test_recommend_long_context(self):
        """Test recommendation for long context."""
        rec = self.engine.recommend(
            model_config=self.llama3_8b,
            phase="decode",
            seq_len=32768,
        )

        # Long context should prioritize KV cache quantization
        self.assertEqual(rec.config.strategy_name, "hybrid_long_context")
        self.assertGreaterEqual(rec.memory_savings_pct, 50.0)

    def test_select_base_strategy_decode(self):
        """Test base strategy selection for decode."""
        strategy = self.engine._select_base_strategy("decode", 4096, self.llama3_8b)

        self.assertEqual(strategy, "nvfp4_blackwell")

    def test_select_base_strategy_long_context(self):
        """Test base strategy selection for long context."""
        strategy = self.engine._select_base_strategy("decode", 16384, self.llama3_8b)

        self.assertEqual(strategy, "hybrid_long_context")

    def test_select_base_strategy_prefill(self):
        """Test base strategy selection for prefill."""
        strategy = self.engine._select_base_strategy("prefill", 2048, self.llama3_8b)

        self.assertEqual(strategy, "fp8_balanced")

    def test_select_fa_version_short_context(self):
        """Test FA version selection for short context."""
        version = self.engine._select_fa_version(4096, "decode")

        self.assertEqual(version, FlashAttentionVersion.FA3)

    def test_select_fa_version_very_long_context(self):
        """Test FA version selection for very long context."""
        version = self.engine._select_fa_version(32768, "decode")

        self.assertEqual(version, FlashAttentionVersion.FA4)

    def test_justification_generated(self):
        """Test that justification is generated."""
        rec = self.engine.recommend(
            model_config=self.llama3_8b,
            phase="decode",
            seq_len=4096,
        )

        self.assertIsInstance(rec.reason, str)
        self.assertGreater(len(rec.reason), 0)

    def test_bottleneck_analysis(self):
        """Test bottleneck analysis."""
        rec = self.engine.recommend(
            model_config=self.llama3_8b,
            phase="decode",
            seq_len=4096,
        )

        self.assertIn("attention", rec.bottleneck_analysis)
        self.assertIn("ffn", rec.bottleneck_analysis)
        self.assertIn("overall", rec.bottleneck_analysis)

    def test_confidence_levels(self):
        """Test confidence level assignment."""
        rec = self.engine.recommend(
            model_config=self.llama3_8b,
            phase="decode",
            seq_len=4096,
        )

        self.assertIn(rec.confidence, ["high", "medium", "low"])

    def test_speedup_greater_than_baseline(self):
        """Test predicted speedup is greater than 1.0."""
        rec = self.engine.recommend(
            model_config=self.llama3_8b,
            phase="decode",
            seq_len=4096,
        )

        self.assertGreater(rec.predicted_speedup, 1.0)
        self.assertGreater(rec.baseline_time_us, rec.predicted_time_us)

    def test_memory_savings(self):
        """Test memory savings calculation."""
        rec = self.engine.recommend(
            model_config=self.llama3_8b,
            phase="decode",
            seq_len=4096,
        )

        self.assertGreaterEqual(rec.memory_savings_pct, 0.0)
        self.assertLessEqual(rec.memory_savings_pct, 100.0)


class TestQuantizationRecommendation(unittest.TestCase):
    """Test QuantizationRecommendation dataclass."""

    def test_recommendation_structure(self):
        """Test recommendation has all required fields."""
        engine = IntegratedQuantizationEngine(BLACKWELL_B10)
        llama3_8b = {
            "H": 4096, "L": 32, "nh": 32, "nkv": 8, "dh": 128, "dff": 14336
        }

        rec = engine.recommend(llama3_8b, phase="decode", seq_len=4096)

        # Check all fields are present and valid
        self.assertIsNotNone(rec.config)
        self.assertIsNotNone(rec.mixed_precision)
        self.assertIsNotNone(rec.fa_config)
        self.assertIsInstance(rec.predicted_speedup, float)
        self.assertIsInstance(rec.predicted_time_us, float)
        self.assertIsInstance(rec.baseline_time_us, float)
        self.assertIsInstance(rec.memory_savings_pct, float)
        self.assertIsInstance(rec.kv_cache_bytes, int)
        self.assertIsInstance(rec.expected_quality_loss, str)
        self.assertIsInstance(rec.confidence, str)
        self.assertIsInstance(rec.reason, str)
        self.assertIsInstance(rec.bottleneck_analysis, dict)


class TestDynamicQuantizationIntegration(unittest.TestCase):
    """Test dynamic quantization integration."""

    def test_dynamic_quant_enabled(self):
        """Test dynamic quantization is enabled when requested."""
        engine = IntegratedQuantizationEngine(BLACKWELL_B10)
        llama3_8b = {
            "H": 4096, "L": 32, "nh": 32, "nkv": 8, "dh": 128, "dff": 14336
        }

        rec = engine.recommend(
            llama3_8b,
            phase="decode",
            seq_len=4096,
            enable_dynamic=True,
        )

        self.assertTrue(rec.config.enable_dynamic_quant)
        self.assertIsNotNone(rec.config.dynamic_quant_config)

    def test_dynamic_quant_disabled(self):
        """Test dynamic quantization is disabled when not requested."""
        engine = IntegratedQuantizationEngine(BLACKWELL_B10)
        llama3_8b = {
            "H": 4096, "L": 32, "nh": 32, "nkv": 8, "dh": 128, "dff": 14336
        }

        rec = engine.recommend(
            llama3_8b,
            phase="decode",
            seq_len=4096,
            enable_dynamic=False,
        )

        self.assertFalse(rec.config.enable_dynamic_quant)


class TestAdaptivePrecisionIntegration(unittest.TestCase):
    """Test adaptive precision integration."""

    def test_adaptive_enabled(self):
        """Test adaptive precision is enabled when requested."""
        engine = IntegratedQuantizationEngine(BLACKWELL_B10)
        llama3_8b = {
            "H": 4096, "L": 32, "nh": 32, "nkv": 8, "dh": 128, "dff": 14336
        }

        rec = engine.recommend(
            llama3_8b,
            phase="decode",
            seq_len=4096,
            enable_adaptive=True,
        )

        self.assertTrue(rec.config.enable_adaptive_precision)
        self.assertIsNotNone(rec.config.adaptive_config)


if __name__ == "__main__":
    unittest.main()
