"""
Dual Roofline Comparison — Theory vs Reality.

Validates roofline predictions against actual measurements.
Supports all Blackwell-native precisions (FP8, INT8, TF32, etc.)
and any hardware in the registry.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.roofline.calculator_shell import RooflineCalculator, bytes_per_element
from src.roofline.hardware_registry import get_hardware, list_hardware, BLACKWELL_B10
from benchmarks.kernel_shell import GEMMKernel, GEMVKernel, check_cuda, get_gpu_info


def compare_gemm(hardware, M, N, K, precision):
    """
    Compare simulated vs measured for a single GEMM.
    Returns dict with both predictions and measurements.
    """
    calc = RooflineCalculator(hardware)

    if M <= 1:
        pred = calc.predict_gemv(N, K, precision)
        kernel = GEMVKernel(N, K, precision)
    else:
        pred = calc.predict_gemm(M, N, K, precision)
        kernel = GEMMKernel(M, N, K, precision)

    meas = kernel.benchmark(num_iters=50)

    error_pct = (
        abs(meas["measured_time_us"] - pred["predicted_time_us"])
        / pred["predicted_time_us"] * 100
    ) if pred["predicted_time_us"] > 0 else 0

    return {
        "shape": (M, N, K),
        "precision": precision,
        "predicted_time_us": pred["predicted_time_us"],
        "predicted_tflops": pred["predicted_tflops"],
        "measured_time_us": meas["measured_time_us"],
        "measured_tflops": meas["measured_tflops"],
        "measured_bandwidth_gb_s": meas["measured_bandwidth_gb_s"],
        "error_pct": error_pct,
        "bottleneck": pred["bottleneck"],
        "ai": pred["ai"],
        "critical_ai": pred["critical_ai"],
    }


def dual_roofline_sweep(hardware, shapes, precisions):
    """
    Run dual roofline comparison across shapes and precisions.
    """
    results = []
    for M, N, K in shapes:
        for prec in precisions:
            try:
                r = compare_gemm(hardware, M, N, K, prec)
                results.append(r)
                status = "OK" if r["error_pct"] < 20 else "HIGH"
                print(f"  [{prec:12s}] {M:>5d}x{N}x{K} | "
                      f"pred={r['predicted_time_us']:>8.1f}μs | "
                      f"meas={r['measured_time_us']:>8.1f}μs | "
                      f"err={r['error_pct']:>5.1f}% [{status}] | "
                      f"{r['bottleneck']}")
            except Exception as e:
                print(f"  [{prec:12s}] {M:>5d}x{N}x{K} | SKIP: {e}")
    return results


if __name__ == "__main__":
    if not check_cuda():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    info = get_gpu_info()
    print("=" * 80)
    print("Dual Roofline: Theory vs Reality")
    print("=" * 80)
    print(f"GPU: {info['name']} (SM {info['compute_capability']})")
    print(f"Blackwell: {info['is_blackwell']} | FP8 support: {info['has_fp8']}")

    # Pick hardware spec
    hw = BLACKWELL_B10
    print(f"Hardware model: {hw.name} ({hw.peak_bandwidth_gb_s} GB/s)")
    print("=" * 80)

    # Choose precisions based on GPU capability
    precisions = ["FP16", "BF16", "TF32"]
    if info["has_fp8"]:
        precisions.append("FP8_E4M3")
    precisions.append("INT8")

    # Shapes: GEMV (decode) + various GEMM sizes
    shapes = [
        (1, 4096, 4096),      # decode GEMV
        (128, 4096, 4096),    # medium batch
        (1024, 4096, 4096),   # large batch
        (4096, 4096, 4096),   # square GEMM
    ]

    print(f"\nPrecisions: {', '.join(precisions)}")
    print(f"Shapes: {len(shapes)} configs")
    print("-" * 80)

    results = dual_roofline_sweep(hw, shapes, precisions)

    # Summary statistics
    if results:
        errors = [r["error_pct"] for r in results]
        print(f"\n{'='*80}")
        print("Summary")
        print(f"{'='*80}")
        print(f"  Total configs: {len(results)}")
        print(f"  Mean error: {sum(errors)/len(errors):.1f}%")
        print(f"  Max error: {max(errors):.1f}%")
        print(f"  Configs < 15% error: {sum(1 for e in errors if e < 15)}/{len(results)}")

        # Speedups vs FP16 (measured)
        print(f"\nMeasured speedups vs FP16:")
        for M, N, K in shapes:
            fp16 = [r for r in results
                    if r["shape"] == (M, N, K) and r["precision"] == "FP16"]
            if fp16:
                t_base = fp16[0]["measured_time_us"]
                shape_results = [r for r in results if r["shape"] == (M, N, K)]
                for r in shape_results:
                    if r["precision"] != "FP16":
                        speedup = t_base / r["measured_time_us"]
                        print(f"  {M:>5d}x{N}x{K} {r['precision']:>12s}: "
                              f"{speedup:.2f}x (pred: {fp16[0]['predicted_time_us']/r['predicted_time_us']:.2f}x)")
