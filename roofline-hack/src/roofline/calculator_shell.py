"""
Roofline Calculator Shell - Implement the formulas yourself!

Reference: frontend/roofline-calc-v2.jsx lines 148-208 for operator math
"""

from dataclasses import dataclass
from typing import Dict, Optional


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

# Hardware specs from JETSON_VALIDATION.md (measured, not theoretical)
JETSON_ORIN_NANO = HardwareSpec(
    name="Jetson Orin Nano",
    peak_bandwidth_gb_s=60.0,
    peak_flops_tflops={
        "FP16": 2.7,
        "INT8": 5.5,
        "INT4": 0.0,  # W4A16 dequant - use FP16 peak
    },
)



# ═══════════════════════════════════════════════
#  ROOFLINE CALCULATOR
# ═══════════════════════════════════════════════

class RooflineCalculator:
    """Roofline performance predictor"""
    
    def __init__(self, hardware: HardwareSpec):
        self.hardware = hardware
    
    def predict_gemv(self, N: int, K: int, precision: str) -> Dict:
        """
        TODO: Predict GEMV performance: y[N] = W[N,K] @ x[K]
        
        Reference: frontend/roofline-calc-v2.jsx lines 157-161
        
        Step 1: Calculate FLOPs
        -----------------------
        GEMV is fused multiply-add: for each of N outputs, do K MAC operations
        Formula: FLOPs = 2 * N * K
        
        Step 2: Calculate Bytes
        -----------------------
        Memory traffic:
        - Read input: x[K] 
        - Read weights: W[N, K]
        - Write output: y[N]
        
        Formula: 
        Bytes = K * bytes_per_elem(precision)       // input
              + N * K * bytes_per_elem(precision)   // weights
              + N * bytes_per_elem(precision)       // output (may be different precision)
        
        Step 3: Calculate Arithmetic Intensity
        ---------------------------------------
        Formula: AI = FLOPs / Bytes (FLOP/byte)
        
        Step 4: Roofline Prediction
        ---------------------------
        Time is limited by either memory or compute, whichever is SLOWER:
        
        time_memory = Bytes / (bandwidth in bytes/sec)
        time_compute = FLOPs / (peak_flops in FLOPS/sec)
        predicted_time = max(time_memory, time_compute)
        
        If time_memory > time_compute: memory-bound
        If time_compute > time_memory: compute-bound
        
        Units conversion:
        - bandwidth: GB/s → bytes/s (× 10^9)
        - peak_flops: TFLOPS → FLOPS/s (× 10^12)
        - time: seconds → microseconds (× 10^6)
        
        Returns
        -------
        Dict with keys:
            predicted_time_us: float
            predicted_tflops: float
            ai: float
            critical_ai: float
            bottleneck: str ('memory' or 'compute')
            flops: int
            bytes: int
        """
        flop_count = 2 * N * K
        bpe = bytes_per_element(precision)
        bytes_used = K * bpe + N * K * bpe + N * bpe

        ai = flop_count / bytes_used
        peak_tflops = self.hardware.peak_flops_tflops.get(
            precision, self.hardware.peak_flops_tflops.get("FP16", 1.0)
        )
        if peak_tflops <= 0:
            peak_tflops = self.hardware.peak_flops_tflops.get("FP16", 1.0)

        t_math = flop_count / (peak_tflops * 1e12)
        t_comm = bytes_used / (self.hardware.peak_bandwidth_gb_s * 1e9)
        pred_time_s = max(t_math, t_comm)
        predicted_time_us = pred_time_s * 1e6
        predicted_tflops = (flop_count / pred_time_s) / 1e12 if pred_time_s > 0 else 0
        critical_ai = self.hardware.critical_ai(precision)
        bottleneck = "memory" if t_comm >= t_math else "compute"

        return {
            "predicted_time_us": predicted_time_us,
            "predicted_tflops": predicted_tflops,
            "ai": ai,
            "critical_ai": critical_ai,
            "bottleneck": bottleneck,
            "flops": flop_count,
            "bytes": int(bytes_used),
        }

    def predict_gemm(self, M: int, N: int, K: int, precision: str) -> Dict:
        """
        Predict GEMM performance: C[M,N] = A[M,K] @ B[K,N]
        FLOPs = 2*M*N*K, Bytes = M*K + K*N + M*N (in bytes_per_elem units)
        """
        flop_count = 2 * M * N * K
        bpe = bytes_per_element(precision)
        bytes_used = M * K * bpe + K * N * bpe + M * N * bpe

        ai = flop_count / bytes_used
        peak_tflops = self.hardware.peak_flops_tflops.get(
            precision, self.hardware.peak_flops_tflops.get("FP16", 1.0)
        )
        if peak_tflops <= 0:
            peak_tflops = self.hardware.peak_flops_tflops.get("FP16", 1.0)

        t_math = flop_count / (peak_tflops * 1e12)
        t_comm = bytes_used / (self.hardware.peak_bandwidth_gb_s * 1e9)
        pred_time_s = max(t_math, t_comm)
        predicted_time_us = pred_time_s * 1e6
        predicted_tflops = (flop_count / pred_time_s) / 1e12 if pred_time_s > 0 else 0
        critical_ai = self.hardware.critical_ai(precision)
        bottleneck = "memory" if t_comm >= t_math else "compute"

        return {
            "predicted_time_us": predicted_time_us,
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
        Predict attention performance: softmax(Q @ K^T) @ V
        Two matmuls: QK^T and (QK^T) @ V. From THEORY_MATH.md.
        """
        qk_flops = 2 * batch * num_heads * seq_len * seq_len * head_dim
        qkv_flops = 2 * batch * num_heads * seq_len * seq_len * head_dim
        flop_count = qk_flops + qkv_flops
        bpe = bytes_per_element(precision)
        bytes_used = 4 * batch * num_heads * seq_len * head_dim * bpe

        ai = flop_count / bytes_used if bytes_used > 0 else 0
        peak_tflops = self.hardware.peak_flops_tflops.get(
            precision, self.hardware.peak_flops_tflops.get("FP16", 1.0)
        )
        if peak_tflops <= 0:
            peak_tflops = self.hardware.peak_flops_tflops.get("FP16", 1.0)

        t_math = flop_count / (peak_tflops * 1e12)
        t_comm = bytes_used / (self.hardware.peak_bandwidth_gb_s * 1e9)
        pred_time_s = max(t_math, t_comm)
        predicted_time_us = pred_time_s * 1e6
        predicted_tflops = (flop_count / pred_time_s) / 1e12 if pred_time_s > 0 else 0
        critical_ai = self.hardware.critical_ai(precision)
        bottleneck = "memory" if t_comm >= t_math else "compute"

        return {
            "predicted_time_us": predicted_time_us,
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
        Predict FFN (SwiGLU) performance. From THEORY_MATH.md.
        gate=True: 3 projections (gate, up, down); gate=False: 2 (up, down).
        """
        T = seq_len if seq_len > 0 else 1
        if gate:
            flop_count = 6 * batch * T * hidden * ffn_dim
        else:
            flop_count = 4 * batch * T * hidden * ffn_dim
        bpe = bytes_per_element(precision)
        bytes_used = (3 * hidden * ffn_dim + 2 * batch * T * ffn_dim + batch * T * hidden) * bpe

        ai = flop_count / bytes_used if bytes_used > 0 else 0
        peak_tflops = self.hardware.peak_flops_tflops.get(
            precision, self.hardware.peak_flops_tflops.get("FP16", 1.0)
        )
        if peak_tflops <= 0:
            peak_tflops = self.hardware.peak_flops_tflops.get("FP16", 1.0)

        t_math = flop_count / (peak_tflops * 1e12)
        t_comm = bytes_used / (self.hardware.peak_bandwidth_gb_s * 1e9)
        pred_time_s = max(t_math, t_comm)
        predicted_time_us = pred_time_s * 1e6
        predicted_tflops = (flop_count / pred_time_s) / 1e12 if pred_time_s > 0 else 0
        critical_ai = self.hardware.critical_ai(precision)
        bottleneck = "memory" if t_comm >= t_math else "compute"

        return {
            "predicted_time_us": predicted_time_us,
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
