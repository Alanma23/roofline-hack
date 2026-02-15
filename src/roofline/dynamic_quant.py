"""
Dynamic Quantization System

Enables runtime quantization based on actual activation statistics.
Supports per-token, per-batch, and adaptive precision selection.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Callable
import math


@dataclass
class ActivationStats:
    """Statistics for activation tensor quantization."""
    absmax: float
    mean: float
    std: float
    percentile_99: float
    percentile_999: float
    shape: Tuple[int, ...]
    dtype: str = "FP16"

    @property
    def dynamic_range(self) -> float:
        """Effective dynamic range (99.9th percentile)."""
        return self.percentile_999

    @property
    def outlier_ratio(self) -> float:
        """Ratio of outliers (values > 3 std from mean)."""
        threshold = abs(self.mean) + 3 * self.std
        return self.absmax / threshold if threshold > 0 else 1.0


@dataclass
class DynamicQuantConfig:
    """Configuration for dynamic quantization."""
    precision: str = "FP8_E4M3"
    granularity: str = "per_tensor"  # per_tensor, per_token, per_channel
    calibration_mode: str = "absmax"  # absmax, percentile, mse_optimal
    percentile_clip: float = 99.9  # For percentile-based calibration
    smooth_alpha: float = 0.5  # For smoothing across batches (EMA)
    outlier_threshold: float = 3.0  # Std deviations for outlier detection
    fallback_to_fp16: bool = True  # Fall back to FP16 if outliers detected


class DynamicQuantizer:
    """
    Runtime quantization with activation-aware scaling.

    Computes optimal scales based on actual activation distributions,
    enabling better accuracy than static quantization.
    """

    def __init__(self, config: DynamicQuantConfig):
        self.config = config
        self.running_stats: Dict[str, ActivationStats] = {}
        self.scale_history: Dict[str, float] = {}

    def compute_scale(self, stats: ActivationStats, target_precision: str) -> Tuple[float, str]:
        """
        Compute optimal quantization scale for given activation stats.

        Returns:
            (scale, precision) - precision may differ from target if fallback triggered
        """
        # Get target format range
        target_range = self._get_format_range(target_precision)

        # Determine clipping value based on calibration mode
        if self.config.calibration_mode == "absmax":
            clip_val = stats.absmax
        elif self.config.calibration_mode == "percentile":
            clip_val = stats.percentile_999  # Use 99.9th percentile
        else:  # mse_optimal
            clip_val = self._compute_mse_optimal_clip(stats, target_precision)

        # Check for outliers - may need to fall back to higher precision
        if self.config.fallback_to_fp16 and stats.outlier_ratio > 2.0:
            # Severe outliers detected, use FP16
            return 1.0, "FP16"

        # Compute scale: scale = target_range / clip_val
        if clip_val > 0:
            scale = target_range / clip_val
        else:
            scale = 1.0

        # Apply smoothing if we have history
        key = f"{stats.shape}_{target_precision}"
        if key in self.scale_history:
            alpha = self.config.smooth_alpha
            scale = alpha * scale + (1 - alpha) * self.scale_history[key]

        self.scale_history[key] = scale
        return scale, target_precision

    def quantize_dynamic(self,
                         activation: "torch.Tensor",
                         weight: "torch.Tensor",
                         operator_name: str,
                         target_precision: str = "FP8_E4M3") -> Tuple["torch.Tensor", Dict]:
        """
        Dynamically quantize activation based on runtime statistics.

        Args:
            activation: Input activation tensor
            weight: Weight tensor (may be pre-quantized)
            operator_name: Name for tracking stats
            target_precision: Target precision (may be overridden)

        Returns:
            (result, metadata) where metadata contains scale, actual precision, etc.
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required for dynamic quantization")

        # Compute activation statistics
        stats = self._compute_stats(activation)
        self.running_stats[operator_name] = stats

        # Determine optimal scale and precision
        scale, actual_precision = self.compute_scale(stats, target_precision)

        # Perform quantization
        if actual_precision == "FP16":
            # Fallback: no quantization
            result = torch.matmul(activation, weight)
            metadata = {
                "scale": 1.0,
                "precision": "FP16",
                "fallback": True,
                "outlier_ratio": stats.outlier_ratio,
            }
        elif actual_precision.startswith("FP8"):
            # FP8 quantization with scaling
            dtype_map = {
                "FP8_E4M3": torch.float8_e4m3fn,
                "FP8_E5M2": torch.float8_e5m2,
            }
            dtype = dtype_map.get(actual_precision, torch.float8_e4m3fn)

            # Quantize activation
            act_scaled = activation * scale
            act_quant = act_scaled.to(dtype)

            # Compute (with dequant implicit in _scaled_mm)
            scale_tensor = torch.tensor(1.0 / scale, device=activation.device)
            result = torch._scaled_mm(
                act_quant.unsqueeze(0) if act_quant.dim() == 1 else act_quant,
                weight.t() if hasattr(weight, 't') else weight,
                scale_a=scale_tensor,
                scale_b=torch.tensor(1.0, device=activation.device),
                out_dtype=torch.float16,
            )

            metadata = {
                "scale": scale,
                "precision": actual_precision,
                "fallback": False,
                "outlier_ratio": stats.outlier_ratio,
                "dynamic_range": stats.dynamic_range,
            }
        else:
            # Other precisions: simple cast
            result = torch.matmul(activation, weight)
            metadata = {"scale": 1.0, "precision": actual_precision, "fallback": False}

        return result, metadata

    def _compute_stats(self, tensor: "torch.Tensor") -> ActivationStats:
        """Compute statistics for activation tensor."""
        import torch

        absmax = tensor.abs().max().item()
        mean = tensor.mean().item()
        std = tensor.std().item()

        # Compute percentiles
        flat = tensor.flatten()
        p99 = torch.quantile(flat.abs(), 0.99).item()
        p999 = torch.quantile(flat.abs(), 0.999).item()

        return ActivationStats(
            absmax=absmax,
            mean=mean,
            std=std,
            percentile_99=p99,
            percentile_999=p999,
            shape=tuple(tensor.shape),
        )

    def _get_format_range(self, precision: str) -> float:
        """Get maximum representable value for precision format."""
        ranges = {
            "FP8_E4M3": 448.0,
            "FP8_E5M2": 57344.0,
            "FP16": 65504.0,
            "BF16": 3.4e38,
            "INT8": 127.0,
            "INT4": 7.0,
        }
        return ranges.get(precision, 448.0)

    def _compute_mse_optimal_clip(self, stats: ActivationStats, precision: str) -> float:
        """
        Compute MSE-optimal clipping value.

        Balances quantization error vs clipping error.
        """
        # Simplified heuristic: clip at point that minimizes expected MSE
        # Full implementation would require histogram of activation distribution

        # For now: use mean + 2.5 * std as good heuristic
        return min(stats.absmax, stats.mean + 2.5 * stats.std)

    def get_stats_summary(self) -> Dict[str, Dict]:
        """Get summary of all tracked operator statistics."""
        summary = {}
        for name, stats in self.running_stats.items():
            summary[name] = {
                "absmax": stats.absmax,
                "dynamic_range": stats.dynamic_range,
                "outlier_ratio": stats.outlier_ratio,
                "shape": stats.shape,
            }
        return summary


# ═══════════════════════════════════════════════
#  ADAPTIVE PRECISION SELECTION
# ═══════════════════════════════════════════════

@dataclass
class AdaptiveQuantConfig:
    """Configuration for adaptive precision selection."""
    # Precision options to consider (ordered by preference)
    precision_candidates: list = None  # ["NVFP4", "FP8_E4M3", "FP16"]

    # Thresholds for automatic selection
    high_outlier_threshold: float = 2.0  # Switch to FP16
    medium_outlier_threshold: float = 1.5  # Switch to FP8
    low_outlier_threshold: float = 1.0  # Can use NVFP4/INT4

    # Performance targets
    target_speedup: float = 3.0
    max_quality_loss: float = 0.02  # 2% perplexity increase

    def __post_init__(self):
        if self.precision_candidates is None:
            self.precision_candidates = ["NVFP4", "FP8_E4M3", "FP16"]


class AdaptivePrecisionSelector:
    """
    Automatically select precision based on activation characteristics.

    Monitors activation distributions and dynamically chooses optimal
    precision to meet speed/quality targets.
    """

    def __init__(self, config: AdaptiveQuantConfig):
        self.config = config
        self.quantizer = DynamicQuantizer(DynamicQuantConfig())
        self.precision_history: Dict[str, list] = {}

    def select_precision(self,
                         operator_name: str,
                         activation_stats: ActivationStats,
                         context: Optional[Dict] = None) -> str:
        """
        Select optimal precision for operator based on activation stats.

        Args:
            operator_name: Name of operator
            activation_stats: Statistics of input activations
            context: Additional context (phase, seq_len, etc.)

        Returns:
            Selected precision format
        """
        outlier_ratio = activation_stats.outlier_ratio

        # Rule-based selection based on outliers
        if outlier_ratio >= self.config.high_outlier_threshold:
            # Severe outliers: use FP16
            precision = "FP16"
            reason = f"High outliers ({outlier_ratio:.2f}), using FP16 for accuracy"

        elif outlier_ratio >= self.config.medium_outlier_threshold:
            # Moderate outliers: use FP8
            precision = "FP8_E4M3"
            reason = f"Moderate outliers ({outlier_ratio:.2f}), using FP8"

        else:
            # Low outliers: can use aggressive quantization
            precision = self.config.precision_candidates[0]  # NVFP4 by default
            reason = f"Low outliers ({outlier_ratio:.2f}), using {precision}"

        # Context-based adjustments
        if context:
            # Long context: prefer aggressive KV cache quantization
            if context.get("is_kv_cache") and context.get("seq_len", 0) > 4096:
                if precision == "FP16":
                    precision = "FP8_E4M3"
                    reason += " (long context override)"

            # Prefill: may benefit from higher precision
            if context.get("phase") == "prefill" and precision.startswith("INT"):
                precision = "FP8_E4M3"
                reason += " (prefill override)"

        # Track history
        if operator_name not in self.precision_history:
            self.precision_history[operator_name] = []
        self.precision_history[operator_name].append((precision, reason))

        return precision

    def get_precision_distribution(self, operator_name: str) -> Dict[str, int]:
        """Get distribution of precision selections for an operator."""
        if operator_name not in self.precision_history:
            return {}

        from collections import Counter
        precisions = [p for p, _ in self.precision_history[operator_name]]
        return dict(Counter(precisions))


# ═══════════════════════════════════════════════
#  PER-TOKEN QUANTIZATION (for decode optimization)
# ═══════════════════════════════════════════════

class PerTokenQuantizer:
    """
    Per-token quantization for decode phase.

    Each token gets its own quantization scale, optimal for
    batch-1 decode where tokens may have very different activation ranges.
    """

    def __init__(self, precision: str = "FP8_E4M3"):
        self.precision = precision
        self.token_scales: Dict[int, float] = {}

    def quantize_per_token(self,
                           activations: "torch.Tensor",
                           weights: "torch.Tensor",
                           token_ids: Optional[list] = None) -> "torch.Tensor":
        """
        Quantize with per-token scales.

        Args:
            activations: [batch, seq_len, hidden] or [seq_len, hidden]
            weights: Weight matrix
            token_ids: Optional token IDs for scale caching

        Returns:
            Quantized result
        """
        import torch

        # Compute per-token absmax
        if activations.dim() == 3:
            # [batch, seq_len, hidden] → [batch, seq_len]
            absmax = activations.abs().max(dim=-1, keepdim=True)[0]
        else:
            # [seq_len, hidden] → [seq_len, 1]
            absmax = activations.abs().max(dim=-1, keepdim=True)[0]

        # Compute per-token scales
        format_range = 448.0 if self.precision == "FP8_E4M3" else 65504.0
        scales = format_range / (absmax + 1e-8)  # Avoid div by zero

        # Quantize
        act_scaled = activations * scales
        if self.precision.startswith("FP8"):
            dtype = torch.float8_e4m3fn if self.precision == "FP8_E4M3" else torch.float8_e5m2
            act_quant = act_scaled.to(dtype)
            # Compute with dequant
            result = torch.matmul(act_quant.to(torch.float16), weights)
            result = result / scales
        else:
            result = torch.matmul(act_scaled, weights) / scales

        return result


if __name__ == "__main__":
    print("=" * 80)
    print("Dynamic Quantization System Demo")
    print("=" * 80)

    # Create config
    config = DynamicQuantConfig(
        precision="FP8_E4M3",
        granularity="per_tensor",
        calibration_mode="percentile",
        fallback_to_fp16=True,
    )

    quantizer = DynamicQuantizer(config)

    # Simulate activation stats
    print("\nTest Case 1: Normal activations (low outliers)")
    stats1 = ActivationStats(
        absmax=10.0,
        mean=0.0,
        std=2.0,
        percentile_99=8.0,
        percentile_999=9.5,
        shape=(1024, 4096),
    )

    scale1, prec1 = quantizer.compute_scale(stats1, "FP8_E4M3")
    print(f"  Scale: {scale1:.2f}")
    print(f"  Precision: {prec1}")
    print(f"  Outlier ratio: {stats1.outlier_ratio:.2f}")

    print("\nTest Case 2: High outliers (should fallback)")
    stats2 = ActivationStats(
        absmax=100.0,
        mean=0.0,
        std=2.0,
        percentile_99=8.0,
        percentile_999=9.5,
        shape=(1024, 4096),
    )

    scale2, prec2 = quantizer.compute_scale(stats2, "FP8_E4M3")
    print(f"  Scale: {scale2:.2f}")
    print(f"  Precision: {prec2}")
    print(f"  Outlier ratio: {stats2.outlier_ratio:.2f}")
    print(f"  Fallback triggered: {prec2 == 'FP16'}")

    # Test adaptive selector
    print("\n" + "=" * 80)
    print("Adaptive Precision Selector Demo")
    print("=" * 80)

    adaptive_config = AdaptiveQuantConfig()
    selector = AdaptivePrecisionSelector(adaptive_config)

    print("\nSelecting precision for different scenarios:")
    test_cases = [
        (stats1, {"phase": "decode", "seq_len": 100}, "Normal decode"),
        (stats2, {"phase": "decode", "seq_len": 100}, "Outlier-heavy decode"),
        (stats1, {"is_kv_cache": True, "seq_len": 8192}, "Long context KV cache"),
        (stats1, {"phase": "prefill", "seq_len": 2048}, "Prefill"),
    ]

    for stats, ctx, desc in test_cases:
        prec = selector.select_precision("test_op", stats, ctx)
        print(f"\n  {desc}:")
        print(f"    Selected: {prec}")
        print(f"    Context: {ctx}")
