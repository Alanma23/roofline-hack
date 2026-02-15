"""
Jetson-aware scheduler: power mode + batch size + precision from roofline.

Uses lowlevel.py for hardware status, auto_quantize for precision.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Literal

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.jetson.lowlevel import (
    get_jetson_status,
    set_power_mode,
    scale_roofline_for_power,
)
from src.roofline.auto_quantize import recommend_quantization
from src.roofline.calculator_shell import HardwareSpec, RooflineCalculator


@dataclass
class SchedulerDecision:
    """Scheduler output."""

    power_mode_id: int  # 0=15W, 1=7W
    power_mode_name: str
    precision: str  # INT4, INT8, FP16
    method: str  # PTQ, AWQ
    batch_size: int
    target: Literal["latency", "throughput", "power"]
    reason: str


def recommend(
    target: Literal["latency", "throughput", "power"] = "latency",
    memory_limit_gb: float = 8.0,
) -> SchedulerDecision:
    """
    Recommend power mode, precision, and batch size.

    - latency: 15W, INT8/INT4, batch=1
    - throughput: 15W, INT4, batch=4+
    - power: 7W, INT4, batch=1
    """
    status = get_jetson_status()
    rec = recommend_quantization(memory_limit_gb=memory_limit_gb)

    if target == "power":
        return SchedulerDecision(
            power_mode_id=1,
            power_mode_name="7W",
            precision=rec.precision,
            method=rec.method,
            batch_size=1,
            target="power",
            reason="Power target → 7W, batch=1, " + rec.reason,
        )
    if target == "throughput":
        return SchedulerDecision(
            power_mode_id=0,
            power_mode_name="15W",
            precision=rec.precision,
            method=rec.method,
            batch_size=4,
            target="throughput",
            reason="Throughput target → 15W, batch=4, " + rec.reason,
        )
    # latency
    return SchedulerDecision(
        power_mode_id=0,
        power_mode_name="15W",
        precision=rec.precision,
        method=rec.method,
        batch_size=1,
        target="latency",
        reason="Latency target → 15W, batch=1, " + rec.reason,
    )


def apply_decision(decision: SchedulerDecision) -> bool:
    """Apply power mode. Returns True if successful."""
    return set_power_mode(decision.power_mode_id)


def get_scaled_hardware() -> Optional[HardwareSpec]:
    """Get HardwareSpec scaled for current power mode."""
    status = get_jetson_status()
    if not status.is_jetson:
        return None
    scaled = scale_roofline_for_power(60, 2.7, 5.5, status)
    return HardwareSpec(
        name=f"Jetson Orin Nano (scaled {status.power_mode_name or '?'})",
        peak_bandwidth_gb_s=scaled["peak_bandwidth_gb_s"],
        peak_flops_tflops=scaled["peak_flops_tflops"],
    )


if __name__ == "__main__":
    for t in ["latency", "throughput", "power"]:
        d = recommend(target=t)
        print(f"[{t}]", d.power_mode_name, d.precision, d.batch_size, "-", d.reason[:50] + "...")
