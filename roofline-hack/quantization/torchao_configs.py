"""
TorchAO (PyTorch Architecture Optimization) quantization configs.

TorchAO provides native PyTorch quantization primitives. This module wraps
quantize_(), int4_weight_only(), int8_weight_only() for transformer models.

Supported: INT4 (int4_weight_only), INT8 (int8_weight_only).
NVFP4/FP8 recommendations map to INT4/INT8 when torchao has no native path.
"""

from typing import Any, Optional

# Precision -> torchao mapping (NVFP4/FP8 recommendations use closest supported)
PRECISION_TO_TORCHAO = {
    "INT4": "INT4",
    "NF4": "INT4",
    "NVFP4": "INT4",
    "MXFP4": "INT4",
    "INT8": "INT8",
    "FP8_E4M3": "INT8",  # torchao has no FP8 weight-only; INT8 fallback
    "FP8_E5M2": "INT8",
}

# Lazy import to allow running without torchao
_torchao_quantize = None
_torchao_int4 = None
_torchao_int8 = None


def _ensure_torchao():
    global _torchao_quantize, _torchao_int4, _torchao_int8
    if _torchao_quantize is None:
        try:
            from torchao.quantization import quantize_
            from torchao.quantization import int4_weight_only, int8_weight_only
            _torchao_quantize = quantize_
            _torchao_int4 = int4_weight_only
            _torchao_int8 = int8_weight_only
        except ImportError:
            _torchao_quantize = False
    return _torchao_quantize


def get_quantize_config(
    precision: str = "INT4",
    group_size: int = 128,
) -> Any:
    """
    Get torchao quantization config for Jetson.

    Args:
        precision: INT4 or INT8
        group_size: for INT4 (default 128)

    Returns:
        Config object for quantize_(), or None if torchao not available
    """
    q = _ensure_torchao()
    if not q:
        return None
    if precision.upper() == "INT4":
        return _torchao_int4(group_size=group_size)
    if precision.upper() == "INT8":
        return _torchao_int8()
    return None


def map_precision_to_torchao(precision: str) -> Optional[str]:
    """Map roofline recommendation (NVFP4, FP8, etc.) to torchao-supported precision."""
    p = precision.strip().upper()
    return PRECISION_TO_TORCHAO.get(p, p if p in ("INT4", "INT8") else None)


def apply_quantization(model, precision: str = "INT4", group_size: int = 128) -> bool:
    """
    Apply weight-only quantization to model in-place.

    Args:
        model: PyTorch model with Linear layers
        precision: INT4, INT8, NVFP4, FP8, etc. (mapped to torchao-supported)
        group_size: for INT4

    Returns:
        True if quantization was applied, False if torchao unavailable
    """
    q = _ensure_torchao()
    if not q:
        return False
    mapped = map_precision_to_torchao(precision)
    if mapped is None:
        return False
    config = get_quantize_config(mapped, group_size)
    if config is None:
        return False
    q(model, config)
    return True
