"""
TorchAO quantization configs for Jetson Orin.

PTQ: int4_weight_only, int8_weight_only
AWQ: requires calibration data; use PTQ as fallback when no calibration.
"""

from typing import Any, Optional

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


def apply_quantization(model, precision: str = "INT4", group_size: int = 128) -> bool:
    """
    Apply weight-only quantization to model in-place.

    Args:
        model: PyTorch model with Linear layers
        precision: INT4 or INT8
        group_size: for INT4

    Returns:
        True if quantization was applied, False if torchao unavailable
    """
    q = _ensure_torchao()
    if not q:
        return False
    config = get_quantize_config(precision, group_size)
    if config is None:
        return False
    q(model, config)
    return True
