"""
GEMM/GEMV benchmark kernels — Blackwell-native.

Supports FP8 (torch._scaled_mm), INT8 (torch._int_mm), TF32,
and all standard precisions. Integrates NVML power tracking.
"""

import torch
from typing import Dict, Optional
from dataclasses import dataclass


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


def bytes_per_element(precision: str) -> float:
    """Bytes per element for bandwidth calculation."""
    p = precision.strip().upper()
    if p == "FP64":
        return 8.0
    if p in ("FP32", "TF32"):
        return 4.0
    if p in ("FP16", "BF16"):
        return 2.0
    if p in ("FP8", "FP8_E4M3", "FP8_E5M2", "INT8"):
        return 1.0
    if p in ("INT4", "NVFP4", "MXFP4"):
        return 0.5
    return 2.0  # default


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

    def benchmark(self, num_iters: int = 100, warmup: int = 10,
                  power_tracker=None) -> Dict:
        """Measure kernel performance with CUDA events + optional NVML tracking."""
        for _ in range(warmup):
            self.run()
        torch.cuda.synchronize()

        if power_tracker:
            power_tracker.start()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
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
            self.A = torch.randn(M, K, dtype=self.dtype, device=device)
            self.B = torch.randn(K, N, dtype=self.dtype, device=device)
        elif self.precision in {"FP8_E4M3", "FP8_E5M2", "FP8"}:
            # FP8: create in FP16 then cast
            self.A = torch.randn(M, K, dtype=torch.float16, device=device).to(self.dtype)
            self.B = torch.randn(K, N, dtype=torch.float16, device=device).to(self.dtype)
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

    def benchmark(self, num_iters: int = 50, warmup: int = 10,
                  power_tracker=None) -> Dict:
        """Benchmark with CUDA events + optional NVML power tracking."""
        for _ in range(warmup):
            self.run()
        torch.cuda.synchronize()

        if power_tracker:
            power_tracker.start()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
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
        "total_memory_gb": round(props.total_mem / (1024 ** 3), 1),
        "sm_count": props.multi_processor_count,
        "is_blackwell": props.major >= 10,
        "is_hopper": props.major == 9,
        "has_fp8": hasattr(torch, "float8_e4m3fn"),
    }


# ═══════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    if not check_cuda():
        print("ERROR: CUDA not available!")
        exit(1)

    info = get_gpu_info()
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
