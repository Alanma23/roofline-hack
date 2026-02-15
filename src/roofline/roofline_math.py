"""
Roofline arithmetic intensity (AI) and performance math.

All formulas follow the standard roofline model:
  AI = FLOPs / Bytes (FLOP/byte)
  Critical_AI = Peak_FLOPS / Bandwidth
  time = max(Bytes/Bandwidth, FLOPs/Peak_FLOPS)
  bottleneck = "memory" if time_memory >= time_compute else "compute"

References: THEORY.md, Williams et al. Roofline model.
"""

from typing import Tuple


# ═══════════════════════════════════════════════
#  FLOP COUNTS
# ═══════════════════════════════════════════════

def gemv_flops(N: int, K: int) -> int:
    """
    GEMV: y[N] = W[N,K] @ x[K]
    Each of N outputs: K multiply-adds = 2*K FLOPs.
    Total: 2 * N * K
    """
    return 2 * N * K


def gemm_flops(M: int, N: int, K: int) -> int:
    """
    GEMM: C[M,N] = A[M,K] @ B[K,N]
    Each output element: K multiply-adds = 2*K FLOPs.
    Total: 2 * M * N * K
    """
    return 2 * M * N * K


# ═══════════════════════════════════════════════
#  BYTES (memory traffic)
# ═══════════════════════════════════════════════

def gemv_bytes(
    N: int,
    K: int,
    bpe_input: float,
    bpe_output: float = 2.0,
) -> float:
    """
    GEMV memory traffic: y[N] = W[N,K] @ x[K]
    - Read x[K]: K * bpe_input
    - Read W[N,K]: N * K * bpe_input
    - Write y[N]: N * bpe_output (typically FP16 = 2.0 from accumulation)
    """
    return K * bpe_input + N * K * bpe_input + N * bpe_output


def gemm_bytes(
    M: int,
    N: int,
    K: int,
    bpe_input: float,
    bpe_output: float = 2.0,
) -> float:
    """
    GEMM memory traffic: C[M,N] = A[M,K] @ B[K,N]
    - Read A[M,K]: M * K * bpe_input
    - Read B[K,N]: K * N * bpe_input
    - Write C[M,N]: M * N * bpe_output (typically FP16 = 2.0)
    """
    return M * K * bpe_input + K * N * bpe_input + M * N * bpe_output


# ═══════════════════════════════════════════════
#  ARITHMETIC INTENSITY
# ═══════════════════════════════════════════════

def arithmetic_intensity(flops: float, bytes_: float) -> float:
    """
    AI = FLOPs / Bytes (FLOP per byte).
    Returns 0.0 if bytes <= 0 to avoid division by zero.
    """
    if bytes_ <= 0:
        return 0.0
    return flops / bytes_


def critical_ai(peak_tflops: float, bandwidth_gb_s: float) -> float:
    """
    Critical arithmetic intensity: Peak_FLOPS / Bandwidth.
    AI < critical_ai → memory-bound; AI > critical_ai → compute-bound.
    """
    if bandwidth_gb_s <= 0:
        return float("inf")
    return (peak_tflops * 1e12) / (bandwidth_gb_s * 1e9)


# ═══════════════════════════════════════════════
#  ROOFLINE PREDICTION
# ═══════════════════════════════════════════════

def roofline_time(
    flops: float,
    bytes_: float,
    peak_tflops: float,
    bandwidth_gb_s: float,
) -> Tuple[float, float, float, str]:
    """
    Roofline time prediction.

    Returns:
        pred_time_s: predicted time in seconds
        pred_tflops: achieved TFLOPS (flops / pred_time_s / 1e12)
        critical_ai: critical arithmetic intensity
        bottleneck: "memory" or "compute"
    """
    t_math = flops / (peak_tflops * 1e12) if peak_tflops > 0 else float("inf")
    t_comm = bytes_ / (bandwidth_gb_s * 1e9) if bandwidth_gb_s > 0 else float("inf")
    pred_time_s = max(t_math, t_comm)
    pred_tflops = (flops / pred_time_s) / 1e12 if pred_time_s > 0 else 0.0
    cai = critical_ai(peak_tflops, bandwidth_gb_s)
    bottleneck = "memory" if t_comm >= t_math else "compute"
    return pred_time_s, pred_tflops, cai, bottleneck


# ═══════════════════════════════════════════════
#  CONVENIENCE: GEMV/GEMM full prediction
# ═══════════════════════════════════════════════

def predict_gemv_ai(N: int, K: int, bpe: float, bpe_out: float = 2.0) -> Tuple[int, float, float]:
    """Return (flops, bytes, ai) for GEMV."""
    flops = gemv_flops(N, K)
    bytes_ = gemv_bytes(N, K, bpe, bpe_out)
    ai = arithmetic_intensity(flops, bytes_)
    return flops, bytes_, ai


def predict_gemm_ai(M: int, N: int, K: int, bpe: float, bpe_out: float = 2.0) -> Tuple[int, float, float]:
    """Return (flops, bytes, ai) for GEMM."""
    flops = gemm_flops(M, N, K)
    bytes_ = gemm_bytes(M, N, K, bpe, bpe_out)
    ai = arithmetic_intensity(flops, bytes_)
    return flops, bytes_, ai


if __name__ == "__main__":
    # Validation against THEORY.md known values
    assert arithmetic_intensity(33_554_432, 33_570_816) < 1.01  # GEMV 4096x4096 FP16 ≈ 1
    assert 1360 < arithmetic_intensity(137_438_953_472, 100_663_296) < 1370  # GEMM 4096^3 FP16 ≈ 1365
    assert abs(critical_ai(62, 287) - 216) < 1  # B10 FP16 critical AI
    print("roofline_math: all validation checks passed")
