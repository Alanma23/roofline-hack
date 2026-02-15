"""
GEMM/GEMV benchmark kernels — Blackwell-native.

Supports FP8 (torch._scaled_mm), INT8 (torch._int_mm), TF32,
and all standard precisions. Integrates NVML power tracking.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from typing import Dict, Optional
from dataclasses import dataclass

from src.roofline.calculator_shell import bytes_per_element


# ═══════════════════════════════════════════════
#  PRECISION MAPPING (Blackwell-aware)
# ═══════════════════════════════════════════════

DTYPE_MAP = {
    "FP64": torch.float64,
    "FP32": torch.float32,
    "TF32": torch.float32,          # stored as FP32, computed via TF32 tensor cores
    "BF16": torch.bfloat16,
    "FP16": torch.float16,
    "INT8": torch.int8,
}

# FP8 dtypes (PyTorch 2.1+)
try:
    DTYPE_MAP["FP8_E4M3"] = torch.float8_e4m3fn
    DTYPE_MAP["FP8_E5M2"] = torch.float8_e5m2
    DTYPE_MAP["FP8"] = torch.float8_e4m3fn
except AttributeError:
    pass  # older PyTorch without FP8


def get_torch_dtype(precision: str) -> torch.dtype:
    """Map precision string to PyTorch dtype."""
    p = precision.strip().upper()
    if p in DTYPE_MAP:
        return DTYPE_MAP[p]
    # Formats without native dtype — use INT8 for packed storage
    if p in ("NVFP4", "MXFP4", "INT4", "MXFP8", "MXFP6"):
        return torch.int8
    raise ValueError(f"Unknown precision: {precision!r}")




# ═══════════════════════════════════════════════
#  TILING CONFIG
# ═══════════════════════════════════════════════

@dataclass
class TilingConfig:
    """GEMM tiling strategy (for analysis, not direct kernel control)."""
    tile_m: int = 128
    tile_n: int = 128
    tile_k: int = 32


# ═══════════════════════════════════════════════
#  GEMV KERNEL
# ═══════════════════════════════════════════════

class GEMVKernel:
    """GEMV: y[N] = W[N,K] @ x[K]"""

    def __init__(self, N: int, K: int, precision: str, device="cuda"):
        self.N = N
        self.K = K
        self.precision = precision.strip().upper()
        self.device = device
        self.dtype = get_torch_dtype(precision)

        if self.precision in {"FP64", "FP32", "TF32", "FP16", "BF16"}:
            self.W = torch.randn(N, K, dtype=self.dtype, device=device)
            self.x = torch.randn(K, dtype=self.dtype, device=device)
        elif self.precision in {"FP8_E4M3", "FP8_E5M2", "FP8"}:
            self.W = torch.randn(N, K, dtype=torch.float16, device=device).to(self.dtype)
            self.x = torch.randn(K, dtype=torch.float16, device=device).to(self.dtype)
            self.scale_w = torch.tensor(1.0, device=device)
            self.scale_x = torch.tensor(1.0, device=device)
        else:
            self.W = torch.randint(-128, 127, (N, K), dtype=torch.int8, device=device)
            self.x = torch.randint(-128, 127, (K,), dtype=torch.int8, device=device)

    def run(self) -> torch.Tensor:
        if self.precision in {"FP8_E4M3", "FP8_E5M2", "FP8"}:
            # FP8 GEMV: reshape x to 2D for _scaled_mm
            x_2d = self.x.unsqueeze(0)  # [1, K]
            return torch._scaled_mm(
                x_2d, self.W.t(),
                scale_a=self.scale_x, scale_b=self.scale_w,
                out_dtype=torch.float16,
            ).squeeze(0)
        if self.precision == "TF32":
            torch.backends.cuda.matmul.allow_tf32 = True
            return torch.matmul(self.W, self.x)
        if self.precision == "INT8":
            # Native INT8 not available for GEMV, cast
            return torch.matmul(self.W.to(torch.float16), self.x.to(torch.float16))
        return torch.matmul(self.W, self.x)

    def benchmark(self, num_iters: int = 100, warmup: int = 25,
                  power_tracker=None) -> Dict:
        """Measure kernel performance with CUDA events + optional NVML tracking."""
        with torch.inference_mode():
            for _ in range(warmup):
                self.run()
        torch.cuda.synchronize()

        if power_tracker:
            power_tracker.start()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.inference_mode():
            start.record()
            for _ in range(num_iters):
                self.run()
            end.record()
        torch.cuda.synchronize()

        nvml_summary = {}
        if power_tracker:
            power_tracker.stop()
            nvml_summary = power_tracker.summary()

        elapsed_ms = start.elapsed_time(end)
        flops_per_call = 2 * self.N * self.K
        bpe = bytes_per_element(self.precision)
        bytes_per_call = self.K * bpe + self.N * self.K * bpe + self.N * 2
        total_time_s = elapsed_ms / 1000
        avg_time_us = (elapsed_ms * 1000) / num_iters
        measured_tflops = (flops_per_call * num_iters) / total_time_s / 1e12
        measured_bandwidth_gb_s = (bytes_per_call * num_iters) / total_time_s / 1e9
        ai = flops_per_call / bytes_per_call

        return {
            "measured_time_us": avg_time_us,
            "measured_tflops": measured_tflops,
            "measured_bandwidth_gb_s": measured_bandwidth_gb_s,
            "ai": ai,
            "flops": flops_per_call,
            "bytes": int(bytes_per_call),
            "precision": self.precision,
            "shape": (self.N, self.K),
            "nvml": nvml_summary,
        }


# ═══════════════════════════════════════════════
#  GEMM KERNEL (Blackwell-native)
# ═══════════════════════════════════════════════

class GEMMKernel:
    """
    GEMM: C[M,N] = A[M,K] @ B[K,N]

    Native paths:
    - FP8: torch._scaled_mm (Hopper+/Blackwell)
    - INT8: torch._int_mm (Ampere+)
    - TF32: torch.matmul with allow_tf32=True (Ampere+)
    - FP16/BF16/FP32: standard torch.matmul
    """

    def __init__(self, M: int, N: int, K: int, precision: str,
                 tiling: Optional[TilingConfig] = None, device="cuda"):
        self.M = M
        self.N = N
        self.K = K
        self.precision = precision.strip().upper()
        self.tiling = tiling or TilingConfig()
        self.device = device
        self.dtype = get_torch_dtype(precision)

        if self.precision in {"FP64", "FP32", "TF32", "FP16", "BF16"}:
            self.A = torch.randn(M, K, dtype=self.dtype, device=device).contiguous()
            self.B = torch.randn(K, N, dtype=self.dtype, device=device).contiguous()
        elif self.precision in {"FP8_E4M3", "FP8_E5M2", "FP8"}:
            # FP8: create in FP16 then cast
            self.A = torch.randn(M, K, dtype=torch.float16, device=device).to(self.dtype).contiguous()
            self.B = torch.randn(K, N, dtype=torch.float16, device=device).to(self.dtype).contiguous()
            self.scale_a = torch.tensor(1.0, device=device)
            self.scale_b = torch.tensor(1.0, device=device)
        elif self.precision == "INT8":
            self.A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
            self.B = torch.randint(-128, 127, (K, N), dtype=torch.int8, device=device)
        else:
            # INT4, NVFP4, MXFP4 — packed in INT8 storage
            self.A = torch.randint(-8, 7, (M, K), dtype=torch.int8, device=device)
            self.B = torch.randint(-8, 7, (K, N), dtype=torch.int8, device=device)

    def run(self) -> torch.Tensor:
        """Execute GEMM using native tensor core paths."""
        if self.precision in {"FP8_E4M3", "FP8_E5M2", "FP8"}:
            # Native FP8 GEMM via scaled matmul
            return torch._scaled_mm(
                self.A, self.B.t(),
                scale_a=self.scale_a,
                scale_b=self.scale_b,
                out_dtype=torch.float16,
            )
        if self.precision == "TF32":
            torch.backends.cuda.matmul.allow_tf32 = True
            return torch.matmul(self.A, self.B)
        if self.precision == "INT8":
            # Try native INT8 tensor core path
            try:
                return torch._int_mm(self.A, self.B)
            except (AttributeError, RuntimeError):
                return torch.matmul(self.A.to(torch.float16), self.B.to(torch.float16))
        if self.precision in {"INT4", "NVFP4", "MXFP4"}:
            # Dequant to FP16 for compute
            return torch.matmul(self.A.to(torch.float16), self.B.to(torch.float16))
        return torch.matmul(self.A, self.B)

    def benchmark(self, num_iters: int = 100, warmup: int = 25,
                  power_tracker=None) -> Dict:
        """Benchmark with CUDA events + optional NVML power tracking."""
        with torch.inference_mode():
            for _ in range(warmup):
                self.run()
        torch.cuda.synchronize()

        if power_tracker:
            power_tracker.start()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.inference_mode():
            start.record()
            for _ in range(num_iters):
                self.run()
            end.record()
        torch.cuda.synchronize()

        nvml_summary = {}
        if power_tracker:
            power_tracker.stop()
            nvml_summary = power_tracker.summary()

        elapsed_ms = start.elapsed_time(end)
        bpe = bytes_per_element(self.precision)
        flops_per_call = 2 * self.M * self.N * self.K
        bytes_per_call = (self.M * self.K + self.K * self.N) * bpe + self.M * self.N * 2
        total_time_s = elapsed_ms / 1000
        avg_time_us = (elapsed_ms * 1000) / num_iters
        measured_tflops = (flops_per_call * num_iters) / total_time_s / 1e12
        measured_bandwidth_gb_s = (bytes_per_call * num_iters) / total_time_s / 1e9
        ai = flops_per_call / bytes_per_call

        return {
            "measured_time_us": avg_time_us,
            "measured_tflops": measured_tflops,
            "measured_bandwidth_gb_s": measured_bandwidth_gb_s,
            "ai": ai,
            "flops": flops_per_call,
            "bytes": int(bytes_per_call),
            "precision": self.precision,
            "shape": (self.M, self.N, self.K),
            "tiling": {
                "tile_m": self.tiling.tile_m,
                "tile_n": self.tiling.tile_n,
                "tile_k": self.tiling.tile_k,
            },
            "nvml": nvml_summary,
        }


# ═══════════════════════════════════════════════
#  DEVICE DETECTION
# ═══════════════════════════════════════════════

def check_cuda() -> bool:
    return torch.cuda.is_available()


def get_gpu_info() -> Dict:
    """Return basic GPU info."""
    if not check_cuda():
        return {"available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_gb": round(getattr(props, "total_memory", getattr(props, "total_mem", 0)) / (1024 ** 3), 1),
        "sm_count": props.multi_processor_count,
        "is_blackwell": props.major >= 10,
        "is_hopper": props.major == 9,
        "has_fp8": hasattr(torch, "float8_e4m3fn"),
    }


# ═══════════════════════════════════════════════
#  SATURATION SWEEP — push GB10 to max
# ═══════════════════════════════════════════════

# Larger shapes for better utilization (multiples of 128 for tiling)
SATURATION_GEMM_SHAPES = [
    (4096, 4096, 4096),   # baseline
    (8192, 4096, 4096),   # 2×M
    (8192, 8192, 4096),   # larger
    (8192, 8192, 8192),   # 8K cube
    (16384, 4096, 4096),  # prefill-style
    (16384, 8192, 4096),  # large prefill
]

# Larger GEMV for memory saturation (working set > L2)
SATURATION_GEMV_SHAPES = [
    (4096, 4096),
    (8192, 8192),
    (16384, 4096),
    (4096, 14336),   # FFN up-proj
]


def run_saturation_sweep(
    precisions: Optional[list] = None,
    warmup: int = 50,
    num_iters: int = 100,
    has_fp8: bool = True,
) -> dict:
    """
    Sweep shapes to find peak TFLOPS and GB/s. Uses larger matrices for better utilization.
    """
    precs = precisions or ["FP16", "FP8_E4M3", "TF32"]
    if "FP8_E4M3" in precs and not has_fp8:
        precs = [p for p in precs if p != "FP8_E4M3"]
    results = {"gemm": [], "gemv": []}

    for M, N, K in SATURATION_GEMM_SHAPES:
        est_gb = (M * K + K * N + M * N) * 2 * 3 / 1e9
        if est_gb > 80:
            continue
        for prec in precs:
            try:
                k = GEMMKernel(M, N, K, prec)
                r = k.benchmark(num_iters=num_iters, warmup=warmup)
                r["shape"] = f"{M}x{N}x{K}"
                results["gemm"].append(r)
            except Exception as e:
                results["gemm"].append({"shape": f"{M}x{N}x{K}", "precision": prec, "error": str(e)})

    for N, K in SATURATION_GEMV_SHAPES:
        for prec in precs:
            try:
                k = GEMVKernel(N, K, prec)
                r = k.benchmark(num_iters=num_iters, warmup=warmup)
                r["shape"] = f"{N}x{K}"
                results["gemv"].append(r)
            except Exception as e:
                results["gemv"].append({"shape": f"{N}x{K}", "precision": prec, "error": str(e)})

    return results


# ═══════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GEMM/GEMV benchmark")
    parser.add_argument("--saturate", action="store_true", help="Saturation sweep: larger shapes to max out GPU")
    args = parser.parse_args()

    if not check_cuda():
        print("ERROR: CUDA not available!")
        exit(1)

    info = get_gpu_info()
    # Optimize for max throughput
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    print("=" * 60)
    print(f"GPU: {info['name']} (SM {info['compute_capability']})")
    print(f"Memory: {info['total_memory_gb']} GB | SMs: {info['sm_count']}")
    print(f"Blackwell: {info['is_blackwell']} | FP8: {info['has_fp8']}")
    print("=" * 60)

    # Determine which precisions to test
    precisions = ["FP16", "BF16", "TF32"]
    if info["has_fp8"]:
        precisions.append("FP8_E4M3")
    precisions.append("INT8")

    # GEMV benchmark
    N, K = 4096, 4096
    print(f"\nGEMV {N}x{K}")
    print("-" * 60)
    gemv_results = {}
    for prec in precisions:
        try:
            kernel = GEMVKernel(N, K, prec)
            r = kernel.benchmark()
            gemv_results[prec] = r
            print(f"  [{prec:12s}] {r['measured_time_us']:8.1f} μs | "
                  f"{r['measured_tflops']:.3f} TFLOPS | "
                  f"{r['measured_bandwidth_gb_s']:.1f} GB/s | AI={r['ai']:.2f}")
        except Exception as e:
            print(f"  [{prec:12s}] SKIP: {e}")

    # GEMM benchmark
    M, N, K = 4096, 4096, 4096
    print(f"\nGEMM {M}x{N}x{K}")
    print("-" * 60)
    gemm_results = {}
    for prec in precisions:
        try:
            kernel = GEMMKernel(M, N, K, prec)
            r = kernel.benchmark()
            gemm_results[prec] = r
            print(f"  [{prec:12s}] {r['measured_time_us']:8.1f} μs | "
                  f"{r['measured_tflops']:.3f} TFLOPS | "
                  f"{r['measured_bandwidth_gb_s']:.1f} GB/s | AI={r['ai']:.2f}")
        except Exception as e:
            print(f"  [{prec:12s}] SKIP: {e}")

    # Speedups
    if "FP16" in gemm_results:
        t_base = gemm_results["FP16"]["measured_time_us"]
        print(f"\nGEMM speedups vs FP16 ({t_base:.1f} μs):")
        for prec, r in gemm_results.items():
            if prec != "FP16":
                print(f"  {prec:12s}: {t_base / r['measured_time_us']:.2f}x")

    # Interpretation (GB10 peak: 287 GB/s, FP16 62T, FP8 124T)
    print("\n" + "=" * 60)
    print("Interpretation (GB10: 287 GB/s BW, FP16 62T, FP8 124T)")
    print("-" * 60)
    print("GEMV: memory-bound (AI~1-2). FP8 > FP16 due to half the bytes.")
    print("GEMM: compute-bound (AI>>200). FP8 exceeds spec (124T) when")
    print("  tensor cores are well utilized. FP16 often underutilizes.")

    # Saturation sweep: larger shapes to push GPU to max
    if args.saturate:
        print("\n" + "=" * 60)
        print("SATURATION SWEEP — larger shapes, max throughput")
        print("=" * 60)
        sweep = run_saturation_sweep(
            precisions=["FP16", "FP8_E4M3", "TF32"],
            warmup=50,
            num_iters=100,
            has_fp8=info.get("has_fp8", True),
        )
        # Best GEMM per precision
        gemm_ok = [r for r in sweep["gemm"] if "measured_tflops" in r]
        if gemm_ok:
            best = max(gemm_ok, key=lambda r: r["measured_tflops"])
            print(f"\nBest GEMM: {best['shape']} {best['precision']} — "
                  f"{best['measured_tflops']:.1f} TFLOPS, {best['measured_bandwidth_gb_s']:.1f} GB/s")
            for prec in ["FP16", "FP8_E4M3", "TF32"]:
                pbest = [r for r in gemm_ok if r["precision"] == prec]
                if pbest:
                    b = max(pbest, key=lambda r: r["measured_tflops"])
                    util = (b["measured_tflops"] / 124 * 100) if prec == "FP8_E4M3" else (b["measured_tflops"] / 62 * 100)
                    print(f"  {prec:12s} best: {b['shape']:20s} {b['measured_tflops']:6.1f} T ({util:.0f}% of peak)")
        # Best GEMV (memory saturation)
        gemv_ok = [r for r in sweep["gemv"] if "measured_bandwidth_gb_s" in r]
        if gemv_ok:
            gemv_best = max(gemv_ok, key=lambda r: r["measured_bandwidth_gb_s"])
            print(f"\nBest GEMV BW: {gemv_best['shape']} {gemv_best['precision']} — "
                  f"{gemv_best['measured_bandwidth_gb_s']:.1f} GB/s ({gemv_best['measured_bandwidth_gb_s']/287*100:.0f}% of 287 GB/s)")
