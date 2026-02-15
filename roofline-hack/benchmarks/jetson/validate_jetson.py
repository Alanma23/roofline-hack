"""
Jetson Orin Nano validation suite
Adapted for edge hardware constraints
"""

import torch
import time
import pandas as pd
import os


def get_jetson_specs():
    """Get current Jetson power mode and clocks. Uses lowlevel module when available."""
    try:
        try:
            from benchmarks.jetson.lowlevel import get_jetson_status
        except ImportError:
            from lowlevel import get_jetson_status
        status = get_jetson_status()
        return {
            "device": status.cuda_device_name or torch.cuda.get_device_name(0),
            "gpu_freq_mhz": status.gpu_freq_mhz,
            "memory_gb": status.cuda_memory_gb or torch.cuda.get_device_properties(0).total_memory / 1e9,
            "power_mode": status.power_mode_name,
            "power_budget_w": status.power_budget_w,
        }
    except ImportError:
        pass
    # Fallback: legacy sysfs (Orin uses 17000000.gpu, older uses 57000000.gpu)
    gpu_freq_mhz = None
    for path in [
        "/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/cur_freq",
        "/sys/devices/gpu.0/devfreq/57000000.gpu/cur_freq",
    ]:
        try:
            with open(path) as f:
                gpu_freq_mhz = int(f.read().strip()) / 1000
            break
        except (OSError, ValueError):
            continue
    return {
        "device": torch.cuda.get_device_name(0),
        "gpu_freq_mhz": gpu_freq_mhz,
        "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
    }


def benchmark_gemv_fp16(N=4096, K=4096, num_iters=100):
    """FP16 GEMV on Jetson"""
    x = torch.randn(K, dtype=torch.float16, device='cuda')
    W = torch.randn(N, K, dtype=torch.float16, device='cuda')
    
    # Warmup
    for _ in range(10):
        _ = torch.matmul(W, x)
    torch.cuda.synchronize()
    
    # Time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(num_iters):
        out = torch.matmul(W, x)
    end.record()
    
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    
    # Metrics
    flops = 2 * N * K * num_iters
    avg_time_us = (elapsed_ms * 1000) / num_iters
    tflops = (flops / (elapsed_ms / 1000)) / 1e12
    
    bytes_per_call = K * 2 + N * K * 2 + N * 2
    bw_gb_s = (bytes_per_call * num_iters) / (elapsed_ms / 1000) / 1e9
    ai = (2 * N * K) / bytes_per_call
    
    return {
        'dtype': 'fp16',
        'time_us': avg_time_us,
        'tflops': tflops,
        'bw_gb_s': bw_gb_s,
        'ai': ai,
    }


def benchmark_gemv_int8(N=4096, K=4096, num_iters=100):
    """INT8 GEMV (with cast to FP16 for compute)"""
    x = torch.randint(-128, 127, (K,), dtype=torch.int8, device='cuda')
    W = torch.randint(-128, 127, (N, K), dtype=torch.int8, device='cuda')
    
    # Warmup
    for _ in range(10):
        _ = torch.matmul(W.to(torch.float16), x.to(torch.float16))
    torch.cuda.synchronize()
    
    # Time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(num_iters):
        out = torch.matmul(W.to(torch.float16), x.to(torch.float16))
    end.record()
    
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    
    # Metrics
    flops = 2 * N * K * num_iters
    avg_time_us = (elapsed_ms * 1000) / num_iters
    tflops = (flops / (elapsed_ms / 1000)) / 1e12
    
    bytes_per_call = K * 1 + N * K * 1 + N * 2  # INT8 input + output FP16
    bw_gb_s = (bytes_per_call * num_iters) / (elapsed_ms / 1000) / 1e9
    ai = (2 * N * K) / bytes_per_call
    
    return {
        'dtype': 'int8',
        'time_us': avg_time_us,
        'tflops': tflops,
        'bw_gb_s': bw_gb_s,
        'ai': ai,
    }


def predict_roofline_jetson(N, K, dtype):
    """Roofline prediction for Jetson Orin Nano"""
    # Jetson specs (measured, not theoretical)
    BW_GB_S = 60  # Realized bandwidth
    PEAK_FP16_TFLOPS = 2.7
    PEAK_INT8_TFLOPS = 5.5
    
    flops = 2 * N * K
    
    if dtype == 'fp16':
        bytes = K * 2 + N * K * 2 + N * 2
        peak_tflops = PEAK_FP16_TFLOPS
    elif dtype == 'int8':
        bytes = K * 1 + N * K * 1 + N * 2
        peak_tflops = PEAK_INT8_TFLOPS
    
    ai = flops / bytes
    ai_critical = peak_tflops / (BW_GB_S / 1000)
    
    # Time is max of memory-bound and compute-bound
    time_memory_us = (bytes / (BW_GB_S * 1e9)) * 1e6
    time_compute_us = (flops / (peak_tflops * 1e12)) * 1e6
    
    predicted_time_us = max(time_memory_us, time_compute_us)
    bottleneck = "memory" if time_memory_us > time_compute_us else "compute"
    
    return {
        'predicted_time_us': predicted_time_us,
        'ai': ai,
        'ai_critical': ai_critical,
        'bottleneck': bottleneck,
    }


def run_jetson_validation():
    """Main validation script for Jetson"""
    print("="*60)
    print("JETSON ORIN NANO VALIDATION")
    print("="*60)
    
    specs = get_jetson_specs()
    print(f"Device: {specs['device']}")
    if specs['gpu_freq_mhz']:
        print(f"GPU Frequency: {specs['gpu_freq_mhz']:.0f} MHz")
    print(f"Memory: {specs['memory_gb']:.1f} GB")
    print()
    
    results = []
    
    configs = [
        (2048, 2048, "Small"),
        (4096, 4096, "Medium"),
    ]
    
    for N, K, desc in configs:
        print(f"\n{desc}: {N}x{K}")
        print("-"*60)
        
        # FP16
        pred = predict_roofline_jetson(N, K, 'fp16')
        meas = benchmark_gemv_fp16(N, K)
        error = abs(meas['time_us'] - pred['predicted_time_us']) / pred['predicted_time_us'] * 100
        
        print(f"[FP16]")
        print(f"  Predicted: {pred['predicted_time_us']:.1f} μs ({pred['bottleneck']})")
        print(f"  Measured:  {meas['time_us']:.1f} μs, {meas['tflops']:.3f} TFLOPS")
        print(f"  Error:     {error:.1f}%")
        print(f"  BW:        {meas['bw_gb_s']:.1f} GB/s")
        
        results.append({
            'config': desc,
            'N': N, 'K': K,
            'dtype': 'FP16',
            'predicted_us': pred['predicted_time_us'],
            'measured_us': meas['time_us'],
            'error_pct': error,
            'tflops': meas['tflops'],
            'bw_gb_s': meas['bw_gb_s'],
        })
        
        # INT8
        pred = predict_roofline_jetson(N, K, 'int8')
        meas = benchmark_gemv_int8(N, K)
        error = abs(meas['time_us'] - pred['predicted_time_us']) / pred['predicted_time_us'] * 100
        
        print(f"[INT8]")
        print(f"  Predicted: {pred['predicted_time_us']:.1f} μs ({pred['bottleneck']})")
        print(f"  Measured:  {meas['time_us']:.1f} μs, {meas['tflops']:.3f} TFLOPS")
        print(f"  Error:     {error:.1f}%")
        print(f"  BW:        {meas['bw_gb_s']:.1f} GB/s")
        
        results.append({
            'config': desc,
            'N': N, 'K': K,
            'dtype': 'INT8',
            'predicted_us': pred['predicted_time_us'],
            'measured_us': meas['time_us'],
            'error_pct': error,
            'tflops': meas['tflops'],
            'bw_gb_s': meas['bw_gb_s'],
        })
    
    # Summary
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(df.to_string(index=False))
    print(f"\nMean error: {df['error_pct'].mean():.1f}%")
    
    # Save results
    os.makedirs('benchmarks/validation', exist_ok=True)
    df.to_csv('benchmarks/validation/jetson_results.csv', index=False)
    print("\nSaved to: benchmarks/validation/jetson_results.csv")


if __name__ == "__main__":
    run_jetson_validation()
