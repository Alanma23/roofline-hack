"""
Power-aware roofline benchmark on Jetson Orin.

Runs GEMV benchmarks at 15W and 7W, reports scaled roofline predictions.
Requires: sudo nvpmodel -m 0 (15W) and sudo nvpmodel -m 1 (7W)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.jetson.lowlevel import (
    get_jetson_status,
    set_power_mode,
    scale_roofline_for_power,
)
from src.roofline.calculator_shell import RooflineCalculator, HardwareSpec


def run_power_sweep():
    """Benchmark at 15W and 7W, compare roofline predictions."""
    try:
        import torch
        from benchmarks.kernel_shell import GEMVKernel, check_cuda
    except ImportError as e:
        print("Requires torch and kernel_shell:", e)
        return

    if not check_cuda():
        print("CUDA not available")
        return

    status = get_jetson_status()
    if not status.is_jetson:
        print("Not running on Jetson - power sweep skipped")
        return

    print("=" * 60)
    print("Power-Aware Roofline Benchmark")
    print("=" * 60)
    print(f"Device: {status.cuda_device_name}")
    print(f"Power mode: {status.power_mode_id} ({status.power_mode_name})")
    print(f"GPU freq: {status.gpu_freq_mhz or 0:.0f} MHz")
    print()

    # Base specs (Jetson Orin Nano 8GB at 15W)
    BASE_BW = 60.0
    BASE_FP16 = 2.7
    BASE_INT8 = 5.5

    N, K = 4096, 4096
    results = []

    for mode_id, mode_label in [(0, "15W"), (1, "7W")]:
        print(f"\n--- Setting {mode_label} (mode {mode_id}) ---")
        if not set_power_mode(mode_id):
            print(f"  Failed to set mode (need sudo)")
            continue

        import time
        time.sleep(2)  # Allow clocks to settle

        status = get_jetson_status()
        scaled = scale_roofline_for_power(BASE_BW, BASE_FP16, BASE_INT8, status)
        hw = HardwareSpec(
            name=f"Jetson Orin Nano ({mode_label})",
            peak_bandwidth_gb_s=scaled["peak_bandwidth_gb_s"],
            peak_flops_tflops=scaled["peak_flops_tflops"],
        )
        calc = RooflineCalculator(hw)

        pred_fp16 = calc.predict_gemv(N, K, "FP16")
        pred_int8 = calc.predict_gemv(N, K, "INT8")

        kernel_fp16 = GEMVKernel(N, K, "FP16")
        kernel_int8 = GEMVKernel(N, K, "INT8")
        meas_fp16 = kernel_fp16.benchmark()
        meas_int8 = kernel_int8.benchmark()

        err_fp16 = abs(meas_fp16["measured_time_us"] - pred_fp16["predicted_time_us"]) / pred_fp16["predicted_time_us"] * 100
        err_int8 = abs(meas_int8["measured_time_us"] - pred_int8["predicted_time_us"]) / pred_int8["predicted_time_us"] * 100

        results.append({
            "mode": mode_label,
            "gpu_mhz": status.gpu_freq_mhz,
            "scale": scaled["scale_factor"],
            "pred_fp16_us": pred_fp16["predicted_time_us"],
            "meas_fp16_us": meas_fp16["measured_time_us"],
            "err_fp16": err_fp16,
            "pred_int8_us": pred_int8["predicted_time_us"],
            "meas_int8_us": meas_int8["measured_time_us"],
            "err_int8": err_int8,
        })

        print(f"  GPU: {status.gpu_freq_mhz or 0:.0f} MHz, scale: {scaled['scale_factor']:.2f}")
        print(f"  FP16: pred {pred_fp16['predicted_time_us']:.0f} μs, meas {meas_fp16['measured_time_us']:.0f} μs, err {err_fp16:.1f}%")
        print(f"  INT8: pred {pred_int8['predicted_time_us']:.0f} μs, meas {meas_int8['measured_time_us']:.0f} μs, err {err_int8:.1f}%")

    if len(results) >= 2:
        speedup_15w = results[0]["meas_fp16_us"] / results[0]["meas_int8_us"]
        speedup_7w = results[1]["meas_fp16_us"] / results[1]["meas_int8_us"]
        perf_ratio = results[0]["meas_fp16_us"] / results[1]["meas_fp16_us"]
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"INT8 speedup @ 15W: {speedup_15w:.2f}x")
        print(f"INT8 speedup @ 7W:  {speedup_7w:.2f}x")
        print(f"15W vs 7W perf ratio: {perf_ratio:.2f}x (15W faster)")


if __name__ == "__main__":
    run_power_sweep()
