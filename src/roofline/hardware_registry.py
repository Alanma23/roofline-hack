"""
Hardware specification registry.

Supports Blackwell B10/B200, H100, A100, and custom ASICs.
Each HardwareSpec defines bandwidth + peak FLOPS per precision,
which is all the roofline model needs.
"""

from typing import Dict, List, Optional
from .calculator_shell import HardwareSpec

# ═══════════════════════════════════════════════
#  BLACKWELL PRESETS
# ═══════════════════════════════════════════════

# B200 (data center, 1000W, HBM3e)
BLACKWELL_B200 = HardwareSpec(
    name="NVIDIA B200",
    peak_bandwidth_gb_s=8000.0,
    peak_flops_tflops={
        "FP64": 45.0,
        "FP32": 90.0,
        "TF32": 2250.0,
        "BF16": 4500.0,
        "FP16": 4500.0,
        "FP8_E4M3": 9000.0,
        "FP8_E5M2": 9000.0,
        "NVFP4": 18000.0,
        "MXFP4": 18000.0,
        "MXFP8": 9000.0,
        "INT8": 9000.0,
        "INT4": 18000.0,
    },
)

# GB10 Grace Blackwell (ASUS Ascent GX10) — measured specs from ServeTheHome
# 128GB LPDDR5X @ 9400MT/s, 256-bit bus = 273-301 GB/s bandwidth
# 31 TFLOPS FP32, 1000 TFLOPS FP4 (dense), 1 PFLOP FP4 (sparse)
BLACKWELL_B10 = HardwareSpec(
    name="NVIDIA GB10 Grace Blackwell (GX10)",
    peak_bandwidth_gb_s=287.0,  # LPDDR5X 9400MT/s, 256-bit
    peak_flops_tflops={
        "FP64": 15.5,  # ~FP32/2
        "FP32": 31.0,  # measured
        "TF32": 62.0,  # ~2×FP32 (tensor cores)
        "BF16": 62.0,  # ~2×FP32
        "FP16": 62.0,  # ~2×FP32
        "FP8_E4M3": 124.0,  # ~2×FP16
        "FP8_E5M2": 124.0,  # ~2×FP16
        "NVFP4": 1000.0,  # measured dense FP4
        "MXFP4": 1000.0,  # similar to NVFP4
        "MXFP8": 124.0,  # similar to FP8
        "INT8": 124.0,  # ~2×FP16
        "INT4": 248.0,  # ~2×INT8
    },
)

# Other common GPUs for comparison
H100_SXM = HardwareSpec(
    name="NVIDIA H100 SXM",
    peak_bandwidth_gb_s=3350.0,
    peak_flops_tflops={
        "FP64": 67.0,
        "FP32": 67.0,
        "TF32": 989.0,
        "BF16": 1979.0,
        "FP16": 1979.0,
        "FP8_E4M3": 3958.0,
        "FP8_E5M2": 3958.0,
        "INT8": 3958.0,
    },
)

A100_SXM = HardwareSpec(
    name="NVIDIA A100 SXM",
    peak_bandwidth_gb_s=2039.0,
    peak_flops_tflops={
        "FP64": 19.5,
        "FP32": 19.5,
        "TF32": 156.0,
        "BF16": 312.0,
        "FP16": 312.0,
        "INT8": 624.0,
        "INT4": 1248.0,
    },
)

# ═══════════════════════════════════════════════
#  REGISTRY
# ═══════════════════════════════════════════════

HARDWARE_REGISTRY: Dict[str, HardwareSpec] = {
    "b10": BLACKWELL_B10,
    "b200": BLACKWELL_B200,
    "h100": H100_SXM,
    "a100": A100_SXM,
}


def get_hardware(key: str) -> HardwareSpec:
    """Lookup hardware by key. Raises KeyError if not found."""
    return HARDWARE_REGISTRY[key]


def list_hardware() -> List[str]:
    """Return all registered hardware keys."""
    return list(HARDWARE_REGISTRY.keys())


def register_hardware(key: str, spec: HardwareSpec):
    """Register a hardware spec (custom GPU or ASIC)."""
    HARDWARE_REGISTRY[key] = spec


def create_custom_asic(
    name: str,
    bandwidth_gb_s: float,
    flops_by_precision: Dict[str, float],
) -> HardwareSpec:
    """
    Create and register a custom ASIC/GPU spec.

    Example:
        create_custom_asic("MyChip", bandwidth_gb_s=500, flops_by_precision={"FP16": 100.0, "INT8": 200.0})
    """
    spec = HardwareSpec(
        name=name,
        peak_bandwidth_gb_s=bandwidth_gb_s,
        peak_flops_tflops=flops_by_precision,
    )
    key = name.lower().replace(" ", "_").replace("-", "_")
    register_hardware(key, spec)
    return spec


if __name__ == "__main__":
    print("=" * 60)
    print("Hardware Registry")
    print("=" * 60)
    for key in list_hardware():
        hw = get_hardware(key)
        precs = [k for k, v in hw.peak_flops_tflops.items() if v > 0]
        print(f"\n  {hw.name} ({key})")
        print(f"    BW: {hw.peak_bandwidth_gb_s} GB/s")
        print(f"    Precisions: {', '.join(precs)}")
        for p in precs:
            cai = hw.critical_ai(p)
            print(f"      {p:12s}: {hw.peak_flops_tflops[p]:>8.1f} TFLOPS  (critical AI = {cai:.1f})")
