"""Roofline calculator, hardware registry, and auto-quantization."""

from .roofline_math import (
    gemv_flops,
    gemv_bytes,
    gemm_flops,
    gemm_bytes,
    arithmetic_intensity,
    critical_ai,
    roofline_time,
)
from .calculator_shell import (
    RooflineCalculator,
    HardwareSpec,
    bytes_per_element,
)
from .hardware_registry import (
    BLACKWELL_B10,
    BLACKWELL_B200,
    H100_SXM,
    A100_SXM,
    HARDWARE_REGISTRY,
    get_hardware,
    list_hardware,
    register_hardware,
    create_custom_asic,
)
from .auto_quantize import recommend_quantization, QuantizationRecommendation

__all__ = [
    "gemv_flops",
    "gemv_bytes",
    "gemm_flops",
    "gemm_bytes",
    "arithmetic_intensity",
    "critical_ai",
    "roofline_time",
    "RooflineCalculator",
    "HardwareSpec",
    "BLACKWELL_B10",
    "BLACKWELL_B200",
    "H100_SXM",
    "A100_SXM",
    "HARDWARE_REGISTRY",
    "get_hardware",
    "list_hardware",
    "register_hardware",
    "create_custom_asic",
    "bytes_per_element",
    "recommend_quantization",
    "QuantizationRecommendation",
]
