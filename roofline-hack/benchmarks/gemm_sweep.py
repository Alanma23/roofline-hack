"""
Systematic GEMM sweep producing dual roofline data (simulated + measured).

Sweeps across GEMM shapes and precisions, producing data points
that can be plotted on the same roofline chart to show theory vs reality.
"""

import sys
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.roofline.calculator_shell import RooflineCalculator, HardwareSpec, bytes_per_element
from src.roofline.hardware_registry import BLACKWELL_B10, get_hardware


@dataclass
class DualRooflinePoint:
    """A single point on the dual roofline chart."""
    M: int
    N: int
    K: int
    precision: str
    # Simulated (always available)
    sim_ai: float
    sim_tflops: float
    sim_time_us: float
    sim_bottleneck: str
    # Measured (None if no GPU)
    meas_ai: Optional[float] = None
    meas_tflops: Optional[float] = None
    meas_time_us: Optional[float] = None
    meas_bandwidth_gb_s: Optional[float] = None
    # Gap
    gap_pct: Optional[float] = None
    # NVML
    power_w: Optional[float] = None
    energy_j: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


# Standard GEMM shapes spanning low to high arithmetic intensity
DEFAULT_SHAPES = [
    # (M, N, K) — description
    (1, 4096, 4096),         # GEMV-like (decode, very low AI)
    (1, 4096, 14336),        # Decode FFN up projection
    (1, 14336, 4096),        # Decode FFN down projection
    (16, 4096, 4096),        # Small batch
    (128, 4096, 4096),       # Medium batch
    (512, 4096, 4096),       # Large batch
    (1024, 4096, 4096),      # Very large batch
    (2048, 4096, 4096),      # Prefill-like
    (2048, 4096, 14336),     # Prefill FFN
    (4096, 4096, 4096),      # Square GEMM (high AI)
]


def run_gemm_sweep(
    hardware: Optional[HardwareSpec] = None,
    shapes: Optional[List[tuple]] = None,
    precisions: Optional[List[str]] = None,
    run_measured: bool = True,
    power_track: bool = False,
) -> List[DualRooflinePoint]:
    """
    Run a full GEMM sweep producing dual roofline data.

    Args:
        hardware: HardwareSpec to use for predictions (default: BLACKWELL_B10)
        shapes: list of (M, N, K) tuples
        precisions: list of precision strings
        run_measured: if True and CUDA available, also run actual benchmarks
        power_track: if True, enable NVML power tracking during benchmarks
    """
    hw = hardware or BLACKWELL_B10
    calc = RooflineCalculator(hw)
    shapes = shapes or DEFAULT_SHAPES
    precisions = precisions or [k for k, v in hw.peak_flops_tflops.items() if v > 0]

    # Check CUDA
    has_cuda = False
    if run_measured:
        try:
            from benchmarks.kernel_shell import GEMMKernel, check_cuda
            has_cuda = check_cuda()
        except ImportError:
            pass

    # Set up power tracker
    tracker = None
    if power_track and has_cuda:
        try:
            from src.nvml.monitor import NVMLMonitor
            from src.nvml.power_tracker import PowerTracker
            mon = NVMLMonitor()
            mon.init()
            tracker = PowerTracker(mon)
        except Exception:
            tracker = None

    results = []
    for M, N, K in shapes:
        for prec in precisions:
            # Simulated prediction
            try:
                if M <= 1:
                    pred = calc.predict_gemv(N, K, prec)
                else:
                    pred = calc.predict_gemm(M, N, K, prec)
            except ValueError:
                continue

            point = DualRooflinePoint(
                M=M, N=N, K=K, precision=prec,
                sim_ai=pred["ai"],
                sim_tflops=pred["predicted_tflops"],
                sim_time_us=pred["predicted_time_us"],
                sim_bottleneck=pred["bottleneck"],
            )

            # Measured benchmark
            if has_cuda and run_measured:
                try:
                    from benchmarks.kernel_shell import GEMMKernel
                    kernel = GEMMKernel(M if M > 0 else 1, N, K, prec)
                    meas = kernel.benchmark(num_iters=20, power_tracker=tracker)
                    point.meas_ai = meas["ai"]
                    point.meas_tflops = meas["measured_tflops"]
                    point.meas_time_us = meas["measured_time_us"]
                    point.meas_bandwidth_gb_s = meas["measured_bandwidth_gb_s"]
                    if point.sim_time_us > 0:
                        point.gap_pct = (
                            abs(point.sim_time_us - point.meas_time_us)
                            / point.sim_time_us * 100
                        )
                    if meas.get("nvml"):
                        point.power_w = meas["nvml"].get("power_avg_w")
                        point.energy_j = meas["nvml"].get("energy_j")
                except Exception:
                    pass

            results.append(point)

    return results


def print_sweep_results(results: List[DualRooflinePoint]):
    """Pretty-print sweep results."""
    print(f"{'Shape':>20s} {'Prec':>10s} {'SimAI':>7s} {'SimTF':>8s} "
          f"{'SimUs':>8s} {'Bound':>7s} {'MeasTF':>8s} {'MeasUs':>8s} {'Gap%':>6s}")
    print("-" * 100)
    for p in results:
        shape = f"{p.M}x{p.N}x{p.K}"
        meas_tf = f"{p.meas_tflops:.2f}" if p.meas_tflops else "-"
        meas_us = f"{p.meas_time_us:.1f}" if p.meas_time_us else "-"
        gap = f"{p.gap_pct:.1f}" if p.gap_pct is not None else "-"
        print(f"{shape:>20s} {p.precision:>10s} {p.sim_ai:>7.2f} {p.sim_tflops:>8.2f} "
              f"{p.sim_time_us:>8.1f} {p.sim_bottleneck:>7s} {meas_tf:>8s} {meas_us:>8s} {gap:>6s}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GEMM Roofline Sweep")
    parser.add_argument("--hardware", default="b10", help="Hardware key from registry")
    parser.add_argument("--no-measure", action="store_true", help="Skip actual benchmarks")
    parser.add_argument("--power", action="store_true", help="Enable NVML power tracking")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--precisions", nargs="+", help="Specific precisions to test")
    args = parser.parse_args()

    hw = get_hardware(args.hardware)
    precs = args.precisions

    print("=" * 100)
    print(f"GEMM Roofline Sweep — {hw.name}")
    print(f"BW: {hw.peak_bandwidth_gb_s} GB/s")
    print("=" * 100)

    results = run_gemm_sweep(
        hardware=hw,
        precisions=precs,
        run_measured=not args.no_measure,
        power_track=args.power,
    )

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        print_sweep_results(results)

        # Summary: best precision per shape
        print("\n" + "=" * 60)
        print("Best precision per shape (by predicted speedup):")
        print("=" * 60)
        shapes_seen = set()
        for p in results:
            shape = (p.M, p.N, p.K)
            if shape not in shapes_seen:
                shapes_seen.add(shape)
                # Find FP16 baseline for this shape
                fp16 = [r for r in results
                        if (r.M, r.N, r.K) == shape and r.precision == "FP16"]
                if fp16:
                    baseline_t = fp16[0].sim_time_us
                    shape_results = [r for r in results if (r.M, r.N, r.K) == shape]
                    best = min(shape_results, key=lambda r: r.sim_time_us)
                    speedup = baseline_t / best.sim_time_us if best.sim_time_us > 0 else 1.0
                    print(f"  {p.M:>5d}x{p.N}x{p.K}: {best.precision:>10s} "
                          f"({speedup:.1f}x vs FP16, {best.sim_bottleneck})")
