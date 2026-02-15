"""
Unit tests for FlashAttention roofline models.
"""

import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.roofline.flash_attention import (
    FlashAttentionVersion,
    FlashAttentionConfig,
    FlashAttentionRoofline,
    recommend_fa_precision,
)
from src.roofline.hardware_registry import BLACKWELL_B10


class TestFlashAttentionVersion(unittest.TestCase):
    """Test FlashAttentionVersion enum."""

    def test_versions_defined(self):
        """Test all FA versions are defined."""
        self.assertTrue(hasattr(FlashAttentionVersion, "FA1"))
        self.assertTrue(hasattr(FlashAttentionVersion, "FA2"))
        self.assertTrue(hasattr(FlashAttentionVersion, "FA3"))
        self.assertTrue(hasattr(FlashAttentionVersion, "FA4"))

    def test_version_values(self):
        """Test version string values."""
        self.assertEqual(FlashAttentionVersion.FA1.value, "fa1")
        self.assertEqual(FlashAttentionVersion.FA2.value, "fa2")
        self.assertEqual(FlashAttentionVersion.FA3.value, "fa3")
        self.assertEqual(FlashAttentionVersion.FA4.value, "fa4")


class TestFlashAttentionConfig(unittest.TestCase):
    """Test FlashAttentionConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = FlashAttentionConfig()

        self.assertEqual(config.version, FlashAttentionVersion.FA2)
        self.assertEqual(config.qkv_precision, "FP16")
        self.assertEqual(config.kv_cache_precision, "FP16")
        self.assertFalse(config.use_native_fp8)
        self.assertFalse(config.block_sparse)
        self.assertTrue(config.is_causal)

    def test_fa3_config(self):
        """Test FA3 configuration."""
        config = FlashAttentionConfig(
            version=FlashAttentionVersion.FA3,
            use_native_fp8=True,
            qkv_precision="FP8_E4M3",
        )

        self.assertEqual(config.version, FlashAttentionVersion.FA3)
        self.assertTrue(config.use_native_fp8)
        self.assertEqual(config.qkv_precision, "FP8_E4M3")

    def test_fa4_sparse_config(self):
        """Test FA4 with sparsity."""
        config = FlashAttentionConfig(
            version=FlashAttentionVersion.FA4,
            block_sparse=True,
            sparsity_ratio=0.3,
        )

        self.assertEqual(config.version, FlashAttentionVersion.FA4)
        self.assertTrue(config.block_sparse)
        self.assertEqual(config.sparsity_ratio, 0.3)


class TestFlashAttentionRoofline(unittest.TestCase):
    """Test FlashAttentionRoofline predictions."""

    def setUp(self):
        """Set up test fixtures."""
        self.roofline = FlashAttentionRoofline(BLACKWELL_B10)
        self.batch = 1
        self.num_heads = 32
        self.seq_len = 4096
        self.head_dim = 128

    def test_predict_fa1_basic(self):
        """Test FA1 prediction with basic parameters."""
        config = FlashAttentionConfig(version=FlashAttentionVersion.FA1)

        metrics = self.roofline.predict_fa1(
            self.batch, self.num_heads, self.seq_len, self.seq_len, self.head_dim, config
        )

        self.assertEqual(metrics.version, "FA1")
        self.assertGreater(metrics.total_flops, 0)
        self.assertGreater(metrics.total_bytes, 0)
        self.assertGreater(metrics.arithmetic_intensity, 0)
        self.assertIn(metrics.bottleneck, ["memory", "compute"])
        self.assertTrue(metrics.uses_tiling)
        self.assertTrue(metrics.uses_online_softmax)
        self.assertFalse(metrics.uses_fp8)

    def test_predict_fa2_speedup(self):
        """Test FA2 is faster than FA1."""
        config = FlashAttentionConfig(version=FlashAttentionVersion.FA1)

        metrics_fa1 = self.roofline.predict_fa1(
            self.batch, self.num_heads, self.seq_len, self.seq_len, self.head_dim, config
        )

        metrics_fa2 = self.roofline.predict_fa2(
            self.batch, self.num_heads, self.seq_len, self.seq_len, self.head_dim, config
        )

        # FA2 should be faster
        self.assertLess(metrics_fa2.predicted_time_us, metrics_fa1.predicted_time_us)
        self.assertEqual(metrics_fa2.version, "FA2")

    def test_predict_fa3_fp8(self):
        """Test FA3 with FP8."""
        config = FlashAttentionConfig(
            version=FlashAttentionVersion.FA3,
            use_native_fp8=True,
        )

        metrics = self.roofline.predict_fa3(
            self.batch, self.num_heads, self.seq_len, self.seq_len, self.head_dim, config
        )

        self.assertEqual(metrics.version, "FA3")
        self.assertTrue(metrics.uses_fp8)

        # FP8 should reduce memory traffic
        config_fp16 = FlashAttentionConfig(version=FlashAttentionVersion.FA3)
        metrics_fp16 = self.roofline.predict_fa3(
            self.batch, self.num_heads, self.seq_len, self.seq_len, self.head_dim, config_fp16
        )

        self.assertLess(metrics.total_bytes, metrics_fp16.total_bytes)

    def test_predict_fa4_sparse(self):
        """Test FA4 with sparsity."""
        config = FlashAttentionConfig(
            version=FlashAttentionVersion.FA4,
            block_sparse=True,
            sparsity_ratio=0.3,
        )

        metrics = self.roofline.predict_fa4(
            self.batch, self.num_heads, self.seq_len, self.seq_len, self.head_dim, config
        )

        self.assertEqual(metrics.version, "FA4")
        self.assertTrue(metrics.uses_sparsity)

        # Sparse should reduce FLOPs and bytes
        config_dense = FlashAttentionConfig(version=FlashAttentionVersion.FA4)
        metrics_dense = self.roofline.predict_fa4(
            self.batch, self.num_heads, self.seq_len, self.seq_len, self.head_dim, config_dense
        )

        self.assertLess(metrics.total_flops, metrics_dense.total_flops)
        self.assertLess(metrics.total_bytes, metrics_dense.total_bytes)

    def test_predict_generic(self):
        """Test generic predict method."""
        config = FlashAttentionConfig(version=FlashAttentionVersion.FA3)

        metrics = self.roofline.predict(
            self.batch, self.num_heads, self.seq_len, self.seq_len, self.head_dim, config
        )

        self.assertEqual(metrics.version, "FA3")

    def test_memory_vs_compute_bound(self):
        """Test bottleneck detection."""
        config = FlashAttentionConfig(version=FlashAttentionVersion.FA2)

        # Decode: typically memory-bound
        metrics_decode = self.roofline.predict(
            1, 32, 1, 4096, 128, config
        )
        self.assertEqual(metrics_decode.bottleneck, "memory")

        # Large batch prefill: may be compute-bound
        metrics_prefill = self.roofline.predict(
            1, 32, 8192, 8192, 128, config
        )
        # Check if AI is higher for prefill
        self.assertGreater(metrics_prefill.arithmetic_intensity, metrics_decode.arithmetic_intensity)


class TestRecommendFAPrecision(unittest.TestCase):
    """Test FlashAttention precision recommendation."""

    def test_recommend_fa1(self):
        """Test FA1 recommendation."""
        config = recommend_fa_precision(
            FlashAttentionVersion.FA1,
            seq_len=4096,
            hardware_name="GB10",
        )

        self.assertEqual(config.version, FlashAttentionVersion.FA1)
        self.assertEqual(config.qkv_precision, "FP16")
        self.assertFalse(config.use_native_fp8)

    def test_recommend_fa2_short_context(self):
        """Test FA2 recommendation for short context."""
        config = recommend_fa_precision(
            FlashAttentionVersion.FA2,
            seq_len=2048,
            hardware_name="GB10",
        )

        self.assertEqual(config.kv_cache_precision, "FP16")

    def test_recommend_fa2_long_context(self):
        """Test FA2 recommendation for long context."""
        config = recommend_fa_precision(
            FlashAttentionVersion.FA2,
            seq_len=8192,
            hardware_name="GB10",
        )

        # Long context should use aggressive KV cache quantization
        self.assertIn(config.kv_cache_precision, ["FP8_E4M3", "NVFP4_KV"])

    def test_recommend_fa3_blackwell(self):
        """Test FA3 recommendation on Blackwell."""
        config = recommend_fa_precision(
            FlashAttentionVersion.FA3,
            seq_len=4096,
            hardware_name="GB10",
        )

        # Should enable FP8 on Blackwell
        self.assertTrue(config.use_native_fp8)
        self.assertEqual(config.qkv_precision, "FP8_E4M3")

    def test_recommend_fa3_non_blackwell(self):
        """Test FA3 recommendation on non-Blackwell hardware."""
        config = recommend_fa_precision(
            FlashAttentionVersion.FA3,
            seq_len=4096,
            hardware_name="A100",
        )

        # Should fallback to FP16
        self.assertFalse(config.use_native_fp8)
        self.assertEqual(config.qkv_precision, "FP16")

    def test_recommend_fa4_very_long_context(self):
        """Test FA4 recommendation for very long context."""
        config = recommend_fa_precision(
            FlashAttentionVersion.FA4,
            seq_len=32768,
            hardware_name="GB10",
        )

        # Should enable sparsity for very long context
        self.assertTrue(config.block_sparse)
        self.assertGreater(config.sparsity_ratio, 0)

    def test_recommend_prefill_vs_decode(self):
        """Test recommendation differs for prefill vs decode."""
        config_prefill = recommend_fa_precision(
            FlashAttentionVersion.FA3,
            seq_len=4096,
            phase="prefill",
        )

        config_decode = recommend_fa_precision(
            FlashAttentionVersion.FA3,
            seq_len=4096,
            phase="decode",
        )

        # Tile sizes should differ
        self.assertGreater(config_prefill.tile_size_q, config_decode.tile_size_q)


class TestFlashAttentionMetrics(unittest.TestCase):
    """Test FlashAttentionMetrics."""

    def test_metrics_structure(self):
        """Test metrics contain expected fields."""
        roofline = FlashAttentionRoofline(BLACKWELL_B10)
        config = FlashAttentionConfig(version=FlashAttentionVersion.FA2)

        metrics = roofline.predict(1, 32, 4096, 4096, 128, config)

        # Check all required fields exist
        self.assertIsNotNone(metrics.version)
        self.assertIsNotNone(metrics.total_flops)
        self.assertIsNotNone(metrics.total_bytes)
        self.assertIsNotNone(metrics.arithmetic_intensity)
        self.assertIsNotNone(metrics.predicted_time_us)
        self.assertIsNotNone(metrics.bottleneck)
        self.assertIsNotNone(metrics.memory_efficiency)
        self.assertIsNotNone(metrics.compute_efficiency)

    def test_time_breakdown(self):
        """Test time breakdown sums correctly."""
        roofline = FlashAttentionRoofline(BLACKWELL_B10)
        config = FlashAttentionConfig(version=FlashAttentionVersion.FA2)

        metrics = roofline.predict(1, 32, 4096, 4096, 128, config)

        # Time breakdown should approximately sum to total
        breakdown_sum = (
            metrics.qk_time_us +
            metrics.softmax_time_us +
            metrics.sv_time_us
        )

        self.assertAlmostEqual(
            breakdown_sum,
            metrics.predicted_time_us,
            delta=metrics.predicted_time_us * 0.01  # 1% tolerance
        )


if __name__ == "__main__":
    unittest.main()
