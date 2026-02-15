"""Roofline calculator, hardware registry, and auto-quantization."""

from .calculator_shell import (
    RooflineCalculator,
    HardwareSpec,
    JETSON_ORIN_NANO,
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
    "RooflineCalculator",
    "HardwareSpec",
    "JETSON_ORIN_NANO",
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
