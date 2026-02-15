"""
NVML-based GPU monitoring.

Works on any NVIDIA GPU via pynvml — Blackwell, Hopper, Ampere, etc.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class GPUStatus:
    """Snapshot of GPU state from NVML."""
    device_name: str = ""
    gpu_clock_mhz: int = 0
    gpu_clock_max_mhz: int = 0
    mem_clock_mhz: int = 0
    mem_clock_max_mhz: int = 0
    sm_clock_mhz: int = 0
    power_draw_w: float = 0.0
    power_limit_w: float = 0.0
    temperature_c: int = 0
    temperature_throttle_c: int = 0
    mem_total_mb: int = 0
    mem_used_mb: int = 0
    mem_free_mb: int = 0
    gpu_utilization_pct: int = 0
    mem_utilization_pct: int = 0
    pcie_tx_kb_s: int = 0
    pcie_rx_kb_s: int = 0
    compute_capability: Tuple[int, int] = (0, 0)
    pcie_gen: int = 0
    pcie_width: int = 0
    errors: List[str] = field(default_factory=list)

    def is_blackwell(self) -> bool:
        return self.compute_capability[0] >= 10

    def is_hopper(self) -> bool:
        return self.compute_capability[0] == 9

    def to_dict(self) -> dict:
        return {
            "device_name": self.device_name,
            "gpu_clock_mhz": self.gpu_clock_mhz,
            "gpu_clock_max_mhz": self.gpu_clock_max_mhz,
            "mem_clock_mhz": self.mem_clock_mhz,
            "power_draw_w": round(self.power_draw_w, 1),
            "power_limit_w": round(self.power_limit_w, 1),
            "temperature_c": self.temperature_c,
            "mem_total_mb": self.mem_total_mb,
            "mem_used_mb": self.mem_used_mb,
            "mem_free_mb": self.mem_free_mb,
            "gpu_utilization_pct": self.gpu_utilization_pct,
            "mem_utilization_pct": self.mem_utilization_pct,
            "compute_capability": list(self.compute_capability),
        }


class NVMLMonitor:
    """
    NVML-based GPU monitor.

    Usage:
        mon = NVMLMonitor()
        mon.init()
        status = mon.sample()
        print(status.device_name, status.power_draw_w)
        mon.shutdown()
    """

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self._handle = None
        self._initialized = False

    def init(self):
        """Initialize NVML. Call once before sampling."""
        import pynvml
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        self._initialized = True

    def shutdown(self):
        """Shutdown NVML."""
        if self._initialized:
            import pynvml
            pynvml.nvmlShutdown()
            self._initialized = False

    def sample(self) -> GPUStatus:
        """Take a single NVML sample."""
        import pynvml
        if not self._initialized:
            self.init()

        h = self._handle
        s = GPUStatus()

        try:
            name = pynvml.nvmlDeviceGetName(h)
            s.device_name = name.decode() if isinstance(name, bytes) else name
        except pynvml.NVMLError as e:
            s.errors.append(f"name: {e}")

        # Clocks
        for attr, clock_type in [
            ("gpu_clock_mhz", pynvml.NVML_CLOCK_GRAPHICS),
            ("mem_clock_mhz", pynvml.NVML_CLOCK_MEM),
            ("sm_clock_mhz", pynvml.NVML_CLOCK_SM),
        ]:
            try:
                setattr(s, attr, pynvml.nvmlDeviceGetClockInfo(h, clock_type))
            except pynvml.NVMLError:
                pass

        for attr, clock_type in [
            ("gpu_clock_max_mhz", pynvml.NVML_CLOCK_GRAPHICS),
            ("mem_clock_max_mhz", pynvml.NVML_CLOCK_MEM),
        ]:
            try:
                setattr(s, attr, pynvml.nvmlDeviceGetMaxClockInfo(h, clock_type))
            except pynvml.NVMLError:
                pass

        # Power
        try:
            s.power_draw_w = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        except pynvml.NVMLError:
            pass
        try:
            s.power_limit_w = pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0
        except pynvml.NVMLError:
            pass

        # Temperature
        try:
            s.temperature_c = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        except pynvml.NVMLError:
            pass
        try:
            s.temperature_throttle_c = pynvml.nvmlDeviceGetTemperatureThreshold(
                h, pynvml.NVML_TEMPERATURE_THRESHOLD_THROTTLE
            )
        except pynvml.NVMLError:
            pass

        # Memory
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            s.mem_total_mb = mem.total // (1024 * 1024)
            s.mem_used_mb = mem.used // (1024 * 1024)
            s.mem_free_mb = mem.free // (1024 * 1024)
        except pynvml.NVMLError:
            pass

        # Utilization
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            s.gpu_utilization_pct = util.gpu
            s.mem_utilization_pct = util.memory
        except pynvml.NVMLError:
            pass

        # PCIe throughput
        try:
            s.pcie_tx_kb_s = pynvml.nvmlDeviceGetPcieThroughput(
                h, pynvml.NVML_PCIE_UTIL_TX_BYTES
            )
            s.pcie_rx_kb_s = pynvml.nvmlDeviceGetPcieThroughput(
                h, pynvml.NVML_PCIE_UTIL_RX_BYTES
            )
        except pynvml.NVMLError:
            pass

        # Compute capability
        try:
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(h)
            s.compute_capability = (major, minor)
        except pynvml.NVMLError:
            pass

        return s

    def detect_hardware_spec(self) -> Optional["HardwareSpec"]:
        """
        Auto-detect HardwareSpec from the actual GPU.
        Matches device name against the hardware registry.
        Returns None if no match found — user must provide specs.
        """
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from src.roofline.hardware_registry import HARDWARE_REGISTRY

        status = self.sample()
        name_lower = status.device_name.lower()
        for key, spec in HARDWARE_REGISTRY.items():
            if key in name_lower or spec.name.lower() in name_lower:
                return spec
        return None


if __name__ == "__main__":
    mon = NVMLMonitor()
    try:
        mon.init()
        status = mon.sample()
        print("=" * 60)
        print(f"GPU: {status.device_name}")
        print(f"Compute: {status.compute_capability[0]}.{status.compute_capability[1]}")
        print(f"Blackwell: {status.is_blackwell()}")
        print(f"Clock: {status.gpu_clock_mhz} / {status.gpu_clock_max_mhz} MHz")
        print(f"Memory: {status.mem_used_mb} / {status.mem_total_mb} MB")
        print(f"Power: {status.power_draw_w:.1f} / {status.power_limit_w:.1f} W")
        print(f"Temp: {status.temperature_c}°C")
        print(f"GPU util: {status.gpu_utilization_pct}%")
        print("=" * 60)

        spec = mon.detect_hardware_spec()
        if spec:
            print(f"Matched registry: {spec.name}")
        else:
            print("No registry match — provide custom HardwareSpec")
    except Exception as e:
        print(f"NVML not available: {e}")
    finally:
        mon.shutdown()
