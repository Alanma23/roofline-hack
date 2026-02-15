"""
Unit tests for dynamic quantization.
"""

import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.roofline.dynamic_quant import (
    ActivationStats,
    DynamicQuantConfig,
    DynamicQuantizer,
    AdaptiveQuantConfig,
    AdaptivePrecisionSelector,
)


class TestActivationStats(unittest.TestCase):
    """Test ActivationStats dataclass."""

    def test_dynamic_range(self):
        """Test dynamic range property."""
        stats = ActivationStats(
            absmax=10.0,
            mean=0.0,
            std=2.0,
            percentile_99=8.0,
            percentile_999=9.5,
            shape=(1024, 4096),
        )

        self.assertEqual(stats.dynamic_range, 9.5)

    def test_outlier_ratio_normal(self):
        """Test outlier ratio for normal distribution."""
        stats = ActivationStats(
            absmax=10.0,
            mean=0.0,
            std=2.0,
            percentile_99=8.0,
            percentile_999=9.5,
            shape=(1024, 4096),
        )

        # absmax / (mean + 3*std) = 10 / 6 ≈ 1.67
        self.assertAlmostEqual(stats.outlier_ratio, 10.0 / 6.0, places=2)

    def test_outlier_ratio_heavy_tailed(self):
        """Test outlier ratio for heavy-tailed distribution."""
        stats = ActivationStats(
            absmax=100.0,
            mean=0.0,
            std=2.0,
            percentile_99=8.0,
            percentile_999=9.5,
            shape=(1024, 4096),
        )

        # absmax / (mean + 3*std) = 100 / 6 ≈ 16.67
        self.assertGreater(stats.outlier_ratio, 10.0)


class TestDynamicQuantConfig(unittest.TestCase):
    """Test DynamicQuantConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = DynamicQuantConfig()

        self.assertEqual(config.precision, "FP8_E4M3")
        self.assertEqual(config.granularity, "per_tensor")
        self.assertEqual(config.calibration_mode, "absmax")
        self.assertTrue(config.fallback_to_fp16)

    def test_custom_config(self):
        """Test custom configuration."""
        config = DynamicQuantConfig(
            precision="NVFP4",
            granularity="per_token",
            calibration_mode="percentile",
            fallback_to_fp16=False,
        )

        self.assertEqual(config.precision, "NVFP4")
        self.assertEqual(config.granularity, "per_token")
        self.assertEqual(config.calibration_mode, "percentile")
        self.assertFalse(config.fallback_to_fp16)


class TestDynamicQuantizer(unittest.TestCase):
    """Test DynamicQuantizer class."""

    def test_get_format_range(self):
        """Test getting format range."""
        config = DynamicQuantConfig()
        quantizer = DynamicQuantizer(config)

        self.assertEqual(quantizer._get_format_range("FP8_E4M3"), 448.0)
        self.assertEqual(quantizer._get_format_range("FP8_E5M2"), 57344.0)
        self.assertEqual(quantizer._get_format_range("FP16"), 65504.0)
        self.assertEqual(quantizer._get_format_range("INT8"), 127.0)

    def test_compute_scale_normal(self):
        """Test scale computation for normal activations."""
        config = DynamicQuantConfig(calibration_mode="absmax")
        quantizer = DynamicQuantizer(config)

        stats = ActivationStats(
            absmax=10.0,
            mean=0.0,
            std=2.0,
            percentile_99=8.0,
            percentile_999=9.5,
            shape=(1024, 4096),
        )

        scale, precision = quantizer.compute_scale(stats, "FP8_E4M3")

        # scale = 448.0 / 10.0 = 44.8
        self.assertAlmostEqual(scale, 44.8, places=1)
        self.assertEqual(precision, "FP8_E4M3")

    def test_compute_scale_percentile(self):
        """Test scale computation with percentile calibration."""
        config = DynamicQuantConfig(calibration_mode="percentile", fallback_to_fp16=False)
        quantizer = DynamicQuantizer(config)

        stats = ActivationStats(
            absmax=15.0,  # Reduce absmax to avoid fallback
            mean=0.0,
            std=2.0,
            percentile_99=8.0,
            percentile_999=9.5,
            shape=(1024, 4096),
        )

        scale, precision = quantizer.compute_scale(stats, "FP8_E4M3")

        # Uses percentile_999 instead of absmax
        # scale = 448.0 / 9.5 ≈ 47.2
        self.assertAlmostEqual(scale, 448.0 / 9.5, places=1)
        self.assertEqual(precision, "FP8_E4M3")

    def test_compute_scale_fallback(self):
        """Test automatic fallback to FP16 for severe outliers."""
        config = DynamicQuantConfig(fallback_to_fp16=True)
        quantizer = DynamicQuantizer(config)

        stats = ActivationStats(
            absmax=100.0,
            mean=0.0,
            std=2.0,
            percentile_99=8.0,
            percentile_999=9.5,
            shape=(1024, 4096),
        )

        scale, precision = quantizer.compute_scale(stats, "FP8_E4M3")

        # Should fallback to FP16 due to high outlier ratio
        self.assertEqual(precision, "FP16")

    def test_compute_scale_smoothing(self):
        """Test scale smoothing across batches."""
        config = DynamicQuantConfig(smooth_alpha=0.5, fallback_to_fp16=False)
        quantizer = DynamicQuantizer(config)

        stats1 = ActivationStats(
            absmax=10.0, mean=0.0, std=5.0,  # Increase std to avoid fallback
            percentile_99=8.0, percentile_999=9.5,
            shape=(1024, 4096)
        )

        stats2 = ActivationStats(
            absmax=8.0, mean=0.0, std=5.0,
            percentile_99=7.0, percentile_999=7.5,
            shape=(1024, 4096)
        )

        # First call
        scale1, prec1 = quantizer.compute_scale(stats1, "FP8_E4M3")
        self.assertEqual(prec1, "FP8_E4M3")

        # Second call should be smoothed
        scale2, prec2 = quantizer.compute_scale(stats2, "FP8_E4M3")
        self.assertEqual(prec2, "FP8_E4M3")

        # scale2 should be smoothed average of scale1 and raw scale
        # Verify it's different from raw scale
        raw_scale2 = 448.0 / 8.0
        self.assertNotAlmostEqual(scale2, raw_scale2, places=1)


class TestAdaptivePrecisionSelector(unittest.TestCase):
    """Test AdaptivePrecisionSelector class."""

    def test_select_precision_low_outliers(self):
        """Test precision selection for low outliers."""
        config = AdaptiveQuantConfig(
            precision_candidates=["NVFP4", "FP8_E4M3", "FP16"],
            low_outlier_threshold=1.0,
            medium_outlier_threshold=2.0,  # Raise threshold
        )
        selector = AdaptivePrecisionSelector(config)

        stats = ActivationStats(
            absmax=5.0, mean=0.0, std=2.0,  # Lower absmax for lower outlier ratio
            percentile_99=4.5, percentile_999=4.9,
            shape=(1024, 4096)
        )

        precision = selector.select_precision("test_op", stats)

        # Low outliers (ratio < 1.0) → should use aggressive quantization
        self.assertEqual(precision, "NVFP4")

    def test_select_precision_high_outliers(self):
        """Test precision selection for high outliers."""
        config = AdaptiveQuantConfig(
            high_outlier_threshold=2.0,
        )
        selector = AdaptivePrecisionSelector(config)

        stats = ActivationStats(
            absmax=100.0, mean=0.0, std=2.0,
            percentile_99=8.0, percentile_999=9.5,
            shape=(1024, 4096)
        )

        precision = selector.select_precision("test_op", stats)

        # High outliers → should fallback to FP16
        self.assertEqual(precision, "FP16")

    def test_select_precision_context_aware_kv_cache(self):
        """Test context-aware precision for KV cache."""
        config = AdaptiveQuantConfig()
        selector = AdaptivePrecisionSelector(config)

        stats = ActivationStats(
            absmax=10.0, mean=0.0, std=2.0,
            percentile_99=8.0, percentile_999=9.5,
            shape=(1024, 4096)
        )

        # Long context KV cache
        context = {"is_kv_cache": True, "seq_len": 16384}
        precision = selector.select_precision("kv_cache", stats, context)

        # Should use aggressive quantization for long context KV cache
        self.assertIn(precision, ["NVFP4", "FP8_E4M3"])

    def test_get_precision_distribution(self):
        """Test getting precision distribution."""
        config = AdaptiveQuantConfig()
        selector = AdaptivePrecisionSelector(config)

        stats = ActivationStats(
            absmax=10.0, mean=0.0, std=2.0,
            percentile_99=8.0, percentile_999=9.5,
            shape=(1024, 4096)
        )

        # Make several selections
        for _ in range(5):
            selector.select_precision("test_op", stats)

        dist = selector.get_precision_distribution("test_op")

        self.assertIsInstance(dist, dict)
        self.assertEqual(sum(dist.values()), 5)


class TestAdaptiveQuantConfig(unittest.TestCase):
    """Test AdaptiveQuantConfig."""

    def test_default_candidates(self):
        """Test default precision candidates."""
        config = AdaptiveQuantConfig()

        self.assertEqual(config.precision_candidates, ["NVFP4", "FP8_E4M3", "FP16"])

    def test_custom_config(self):
        """Test custom configuration."""
        config = AdaptiveQuantConfig(
            precision_candidates=["FP8_E4M3", "FP16"],
            high_outlier_threshold=3.0,
            target_speedup=4.0,
        )

        self.assertEqual(config.precision_candidates, ["FP8_E4M3", "FP16"])
        self.assertEqual(config.high_outlier_threshold, 3.0)
        self.assertEqual(config.target_speedup, 4.0)


if __name__ == "__main__":
    unittest.main()
