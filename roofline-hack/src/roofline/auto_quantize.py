"""
Auto-select quantization precision based on roofline analysis.

Blackwell-aware: considers FP8 (E4M3/E5M2), NVFP4, MXFP4, INT8, INT4
as first-class precision options, in addition to FP16/BF16 baseline.

Decision logic:
  1. Predict GEMM performance at each precision
  2. Rank by predicted speedup over FP16 baseline
  3. Pick the best precision, report method + reason
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple

from .calculator_shell import (
    RooflineCalculator,
    HardwareSpec,
    JETSON_ORIN_NANO,
    bytes_per_element,
)


@dataclass
class QuantizationRecommendation:
    """Recommended quantization based on roofline analysis."""
    precision: str           # e.g. "FP8_E4M3", "NVFP4", "INT8", "INT4"
    method: str              # e.g. "native_fp8", "native_fp4", "PTQ", "AWQ"
    reason: str
    predicted_speedup: float
    memory_bound: bool
    memory_savings_pct: float = 0.0  # reduction vs FP16


# Precision -> method mapping
_METHOD_MAP = {
    "FP8_E4M3": "native_fp8",
    "FP8_E5M2": "native_fp8",
    "FP8": "native_fp8",
    "NVFP4": "native_fp4",
    "MXFP4": "native_fp4",
    "MXFP8": "native_mxfp8",
    "INT8": "PTQ",
    "INT4": "AWQ",
    "NF4": "AWQ",
}


def recommend_quantization(
    hardware: Optional[HardwareSpec] = None,
    M: int = 1,
    N: int = 4096,
    K: int = 4096,
    phase: str = "decode",
    memory_limit_gb: Optional[float] = None,
    model_config: Optional[dict] = None,
) -> QuantizationRecommendation:
    """
    Recommend optimal precision for a GEMM workload on given hardware.

    For decode (M=1): uses predict_gemv (memory-bound, lower precision helps a lot).
    For prefill (M>1): uses predict_gemm (may be compute-bound at large M).

    Returns the precision with highest predicted speedup, along with
    the method (native_fp8, AWQ, PTQ, etc.) and reason.
    """
    # Default to Blackwell B10 if available, otherwise Jetson
    if hardware is None:
        try:
            from .hardware_registry import BLACKWELL_B10
            hw = BLACKWELL_B10
        except ImportError:
            hw = JETSON_ORIN_NANO
    else:
        hw = hardware

    calc = RooflineCalculator(hw)

    # Determine GEMM shape
    if model_config:
        H = model_config.get("H", N)
        N = H
        K = H
        if phase == "decode":
            M = 1
        else:
            M = model_config.get("S", M)

    # Candidate precisions: whatever the hardware supports
    supported = [k for k, v in hw.peak_flops_tflops.items() if v > 0]
    # Always include baseline
    candidates_to_test = ["FP16"] + [p for p in supported if p != "FP16"]

    # Predict performance for each precision
    predictions = {}
    for prec in candidates_to_test:
        try:
            if M <= 1:
                pred = calc.predict_gemv(N, K, prec)
            else:
                pred = calc.predict_gemm(M, N, K, prec)
            predictions[prec] = pred
        except ValueError:
            continue

    baseline = predictions.get("FP16") or predictions.get("BF16")
    if not baseline:
        return QuantizationRecommendation(
            "FP16", "none", "No baseline available", 1.0, False, 0.0,
        )

    ai = baseline["ai"]
    critical_ai = baseline["critical_ai"]
    memory_bound = ai < critical_ai
    baseline_time = baseline["predicted_time_us"]

    # Rank candidates by speedup
    ranked: List[Tuple[str, float, float, str]] = []
    baseline_bpe = bytes_per_element("FP16")
    for prec, pred in predictions.items():
        if prec in ("FP16", "BF16"):
            continue
        t = pred["predicted_time_us"]
        speedup = baseline_time / t if t > 0 else 1.0
        bpe = bytes_per_element(prec)
        mem_save = (1.0 - bpe / baseline_bpe) * 100
        method = _METHOD_MAP.get(prec, "PTQ")
        ranked.append((prec, speedup, mem_save, method))

    ranked.sort(key=lambda x: x[1], reverse=True)

    # Memory-constrained: force smallest format
    if memory_limit_gb and memory_limit_gb <= 8:
        # Find smallest precision with maximum memory savings
        ranked.sort(key=lambda x: x[2], reverse=True)
        if ranked:
            best_prec, best_speedup, best_mem_save, best_method = ranked[0]
            return QuantizationRecommendation(
                precision=best_prec,
                method=best_method,
                reason=f"Memory limit {memory_limit_gb}GB -> {best_prec} "
                       f"({best_mem_save:.0f}% smaller, ~{best_speedup:.1f}x speedup)",
                predicted_speedup=best_speedup,
                memory_bound=memory_bound,
                memory_savings_pct=best_mem_save,
            )

    # Pick best speedup
    if ranked:
        best_prec, best_speedup, best_mem_save, best_method = ranked[0]
        bound_str = "Memory" if memory_bound else "Compute"
        reason = (
            f"{bound_str}-bound (AI={ai:.1f}, critical={critical_ai:.0f}) -> "
            f"{best_prec} gives ~{best_speedup:.1f}x speedup, "
            f"{best_mem_save:.0f}% memory savings"
        )
        return QuantizationRecommendation(
            precision=best_prec,
            method=best_method,
            reason=reason,
            predicted_speedup=best_speedup,
            memory_bound=memory_bound,
            memory_savings_pct=best_mem_save,
        )

    return QuantizationRecommendation(
        "FP16", "none",
        f"{'Memory' if memory_bound else 'Compute'}-bound, FP16 baseline",
        1.0, memory_bound, 0.0,
    )


if __name__ == "__main__":
    try:
        from .hardware_registry import BLACKWELL_B10, BLACKWELL_B200, HARDWARE_REGISTRY
    except ImportError:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from src.roofline.hardware_registry import BLACKWELL_B10, BLACKWELL_B200

    print("=" * 70)
    print("Auto-Quantizer — Blackwell B10")
    print("=" * 70)

    # Decode scenario (M=1, memory-bound)
    rec = recommend_quantization(hardware=BLACKWELL_B10, M=1, N=4096, K=4096, phase="decode")
    print(f"\nDecode (GEMV 4096x4096):")
    print(f"  Precision: {rec.precision}")
    print(f"  Method: {rec.method}")
    print(f"  Speedup: {rec.predicted_speedup:.1f}x")
    print(f"  Memory savings: {rec.memory_savings_pct:.0f}%")
    print(f"  Reason: {rec.reason}")

    # Prefill scenario (M=2048, may be compute-bound)
    rec = recommend_quantization(hardware=BLACKWELL_B10, M=2048, N=4096, K=4096, phase="prefill")
    print(f"\nPrefill (GEMM 2048x4096x4096):")
    print(f"  Precision: {rec.precision}")
    print(f"  Method: {rec.method}")
    print(f"  Speedup: {rec.predicted_speedup:.1f}x")
    print(f"  Reason: {rec.reason}")

    # Large batch (likely compute-bound)
    rec = recommend_quantization(hardware=BLACKWELL_B10, M=8192, N=4096, K=4096)
    print(f"\nLarge batch (GEMM 8192x4096x4096):")
    print(f"  Precision: {rec.precision}")
    print(f"  Method: {rec.method}")
    print(f"  Speedup: {rec.predicted_speedup:.1f}x")
    print(f"  Reason: {rec.reason}")
