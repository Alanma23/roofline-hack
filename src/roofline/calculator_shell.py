"""
Roofline Calculator - Predict performance from hardware specs and precision formats.

Supports Blackwell GB10/B10, B200, H100, A100 and custom ASICs.
Handles FP16, FP8, NVFP4, MXFP4, INT8, INT4 and other formats.

Arithmetic intensity (AI) and roofline math: see roofline_math.py
"""

from dataclasses import dataclass
from typing import Dict, Optional

from .roofline_math import (
    gemv_flops,
    gemv_bytes,
    gemm_flops,
    gemm_bytes,
    arithmetic_intensity,
    roofline_time,
)


# ═══════════════════════════════════════════════
#  PRECISION FORMATS (from frontend)
# ═══════════════════════════════════════════════

def bytes_per_element(precision: str) -> float:
    """
    Returns the effective bytes per element for the given precision format.
    Covers all Blackwell-native formats (FP8, FP4, MX block formats).
    """
    p = precision.strip().upper()

    # Scalar floating point
    if p == "FP64":
        return 8.0
    if p == "FP32":
        return 4.0
    if p == "TF32":
        return 4.0  # 19-bit mantissa stored in 32-bit container
    if p in {"FP16", "BF16"}:
        return 2.0

    # FP8 (Hopper+/Blackwell native)
    if p in {"FP8", "FP8_E4M3", "FP8_E5M2"}:
        return 1.0

    # Integer scalar
    if p == "INT8":
        return 1.0
    if p == "INT4":
        return 0.5
    if p == "INT2":
        return 0.25

    # MX block formats (OCP standard) — element bits + scale overhead
    if p in {"MXFP8", "MXFP8_E4M3", "MXFP8_E5M2"}:
        # 8 bits + (8-bit E8M0 scale / 32 elements) = 8.25 bits
        return 8.25 / 8.0
    if p in {"MXFP6", "MXFP6_E3M2", "MXFP6_E2M3"}:
        # 6 bits + (8-bit scale / 32) = 6.25 bits
        return 6.25 / 8.0
    if p == "MXFP4":
        # 4 bits + (8-bit scale / 32) = 4.25 bits
        return 4.25 / 8.0
    if p == "MXINT8":
        # 8 bits + (8-bit scale / 32) = 8.25 bits
        return 8.25 / 8.0

    # NVIDIA FP4 (Blackwell, two-level scaling)
    if p == "NVFP4":
        # 4 bits + (8-bit scale / 16) + (32-bit tensor scale / 1024)
        bits = 4.0 + (8.0 / 16.0) + (32.0 / 1024.0)  # = 4.53125 bits
        return bits / 8.0
    if p == "NVFP4_KV":
        # Same as NVFP4 for KV cache
        bits = 4.0 + (8.0 / 16.0) + (32.0 / 1024.0)
        return bits / 8.0

    # Lookup table format
    if p == "NF4":
        # 4 bits + (16-bit absmax scale / 64) = 4.25 bits
        return 4.25 / 8.0

    raise ValueError(f"Unknown precision format: {precision!r}")

# ═══════════════════════════════════════════════
#  HARDWARE SPECIFICATIONS
# ═══════════════════════════════════════════════

@dataclass
class HardwareSpec:
    """Hardware specifications for roofline analysis"""
    name: str
    peak_bandwidth_gb_s: float  # GB/s
    peak_flops_tflops: Dict[str, float]  # TFLOPS by precision
    
    def critical_ai(self, precision: str) -> float:
        """
        Calculate critical arithmetic intensity.
        Formula: Critical_AI = Peak_FLOPS / Peak_Bandwidth (FLOP/byte)
        AI < Critical_AI → memory-bound; AI > Critical_AI → compute-bound.
        """
        peak_tflops = self.peak_flops_tflops.get(
            precision, self.peak_flops_tflops.get("FP16", 1.0)
        )
        if peak_tflops <= 0:
            peak_tflops = self.peak_flops_tflops.get("FP16", 1.0)
        return (peak_tflops * 1e12) / (self.peak_bandwidth_gb_s * 1e9)

# ═══════════════════════════════════════════════
#  ROOFLINE CALCULATOR
# ═══════════════════════════════════════════════

class RooflineCalculator:
    """Roofline performance predictor"""
    
    def __init__(self, hardware: HardwareSpec):
        self.hardware = hardware
    
    def predict_gemv(self, N: int, K: int, precision: str) -> Dict:
        """
        Predict GEMV performance: y[N] = W[N,K] @ x[K]
        AI = FLOPs / Bytes, output typically FP16 (2 bytes).
        """
        bpe = bytes_per_element(precision)
        flop_count = gemv_flops(N, K)
        bytes_used = gemv_bytes(N, K, bpe, bpe_output=2.0)

        peak_tflops = self._peak_tflops(precision)
        pred_time_s, predicted_tflops, critical_ai, bottleneck = roofline_time(
            flop_count, bytes_used, peak_tflops, self.hardware.peak_bandwidth_gb_s
        )
        ai = arithmetic_intensity(flop_count, bytes_used)

        return {
            "predicted_time_us": pred_time_s * 1e6,
            "predicted_tflops": predicted_tflops,
            "ai": ai,
            "critical_ai": critical_ai,
            "bottleneck": bottleneck,
            "flops": flop_count,
            "bytes": int(bytes_used),
        }

    def _peak_tflops(self, precision: str) -> float:
        """Get peak TFLOPS for precision, with fallback to FP16."""
        peak = self.hardware.peak_flops_tflops.get(
            precision, self.hardware.peak_flops_tflops.get("FP16", 1.0)
        )
        if peak <= 0:
            peak = self.hardware.peak_flops_tflops.get("FP16", 1.0)
        return peak

    def predict_gemm(self, M: int, N: int, K: int, precision: str) -> Dict:
        """
        Predict GEMM performance: C[M,N] = A[M,K] @ B[K,N]
        Output C is typically FP16 (2 bytes) from accumulation.
        """
        bpe = bytes_per_element(precision)
        flop_count = gemm_flops(M, N, K)
        bytes_used = gemm_bytes(M, N, K, bpe, bpe_output=2.0)

        peak_tflops = self._peak_tflops(precision)
        pred_time_s, predicted_tflops, critical_ai, bottleneck = roofline_time(
            flop_count, bytes_used, peak_tflops, self.hardware.peak_bandwidth_gb_s
        )
        ai = arithmetic_intensity(flop_count, bytes_used)

        return {
            "predicted_time_us": pred_time_s * 1e6,
            "predicted_tflops": predicted_tflops,
            "ai": ai,
            "critical_ai": critical_ai,
            "bottleneck": bottleneck,
            "flops": flop_count,
            "bytes": int(bytes_used),
        }

    def predict_attention(
        self,
        batch: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        precision: str,
    ) -> Dict:
        """
        Predict attention: softmax(Q @ K^T) @ V.
        FLOPs: 2 matmuls (QK^T + Score@V). Bytes: Q,K,V,O tensors (simplified).
        """
        flop_count = 4 * batch * num_heads * seq_len * seq_len * head_dim
        bpe = bytes_per_element(precision)
        # Q, K, V, O each: batch * num_heads * seq * head_dim elements
        bytes_used = 4 * batch * num_heads * seq_len * head_dim * bpe

        ai = arithmetic_intensity(flop_count, bytes_used)
        peak_tflops = self._peak_tflops(precision)
        pred_time_s, predicted_tflops, critical_ai, bottleneck = roofline_time(
            flop_count, bytes_used, peak_tflops, self.hardware.peak_bandwidth_gb_s
        )

        return {
            "predicted_time_us": pred_time_s * 1e6,
            "predicted_tflops": predicted_tflops,
            "ai": ai,
            "critical_ai": critical_ai,
            "bottleneck": bottleneck,
            "flops": flop_count,
            "bytes": int(bytes_used),
        }

    def predict_ffn(
        self,
        batch: int,
        seq_len: int,
        hidden: int,
        ffn_dim: int,
        precision: str,
        gate: bool = True,
    ) -> Dict:
        """
        Predict FFN (SwiGLU): gate, up, down projections.
        gate=True: 6*B*T*H*dff FLOPs; gate=False: 4*B*T*H*dff.
        Bytes: weights (3*H*dff) + activations.
        """
        T = seq_len if seq_len > 0 else 1
        flop_count = 6 * batch * T * hidden * ffn_dim if gate else 4 * batch * T * hidden * ffn_dim
        bpe = bytes_per_element(precision)
        # Weights: gate(H,dff) + up(H,dff) + down(dff,H) = 3*H*dff
        # Activations: input, gate/up outputs, down output
        bytes_used = (3 * hidden * ffn_dim + 2 * batch * T * ffn_dim + batch * T * hidden) * bpe

        ai = arithmetic_intensity(flop_count, bytes_used)
        peak_tflops = self._peak_tflops(precision)
        pred_time_s, predicted_tflops, critical_ai, bottleneck = roofline_time(
            flop_count, bytes_used, peak_tflops, self.hardware.peak_bandwidth_gb_s
        )

        return {
            "predicted_time_us": pred_time_s * 1e6,
            "predicted_tflops": predicted_tflops,
            "ai": ai,
            "critical_ai": critical_ai,
            "bottleneck": bottleneck,
            "flops": flop_count,
            "bytes": int(bytes_used),
        }


# ═══════════════════════════════════════════════
#  TESTING
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    from .hardware_registry import BLACKWELL_B10, BLACKWELL_B200, list_hardware, get_hardware

    print("=" * 60)
    print("Roofline Calculator — Blackwell B10")
    print("=" * 60)

    calc = RooflineCalculator(BLACKWELL_B10)

    # GEMM sweep across precisions
    M, N, K = 4096, 4096, 4096
    print(f"\nGEMM {M}×{N}×{K}")
    print("-" * 60)
    precisions = ["FP16", "BF16", "FP8_E4M3", "NVFP4", "INT8", "INT4"]
    results = {}
    for prec in precisions:
        try:
            r = calc.predict_gemm(M, N, K, prec)
            results[prec] = r
            print(f"  [{prec:12s}] {r['predicted_time_us']:8.1f} μs | "
                  f"AI={r['ai']:.2f} | crit={r['critical_ai']:.0f} | {r['bottleneck']}")
        except ValueError:
            pass

    # Speedups vs FP16
    if "FP16" in results:
        t_base = results["FP16"]["predicted_time_us"]
        print(f"\n  Speedups vs FP16 ({t_base:.1f} μs):")
        for prec, r in results.items():
            if prec != "FP16":
                print(f"    {prec:12s}: {t_base / r['predicted_time_us']:.2f}x")

    # Also show GEMV (decode-like, M=1)
    print(f"\nGEMV 4096×4096 (decode)")
    print("-" * 60)
    for prec in precisions:
        try:
            r = calc.predict_gemv(4096, 4096, prec)
            print(f"  [{prec:12s}] {r['predicted_time_us']:8.1f} μs | "
                  f"AI={r['ai']:.2f} | {r['bottleneck']}")
        except ValueError:
            pass
