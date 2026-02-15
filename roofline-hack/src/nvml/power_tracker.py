"""
Background NVML sampling thread.

Runs during kernel execution to capture power draw, temperature,
and clock frequencies at high frequency (default 10ms).
"""

import time
import threading
from dataclasses import dataclass
from typing import List, Optional

from .monitor import NVMLMonitor


@dataclass
class NVMLSample:
    """Single NVML sample point."""
    timestamp_s: float
    power_w: float
    temperature_c: int
    gpu_clock_mhz: int
    mem_clock_mhz: int
    gpu_util_pct: int
    mem_util_pct: int


class PowerTracker:
    """
    Samples NVML in a background thread during kernel execution.

    Usage:
        tracker = PowerTracker(monitor)
        tracker.start()
        # ... run kernel ...
        samples = tracker.stop()
        print(tracker.summary())
    """

    def __init__(self, monitor: NVMLMonitor, interval_ms: int = 10):
        self.monitor = monitor
        self.interval_s = interval_ms / 1000.0
        self.samples: List[NVMLSample] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start background sampling."""
        self.samples = []
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> List[NVMLSample]:
        """Stop sampling and return all collected samples."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        return self.samples

    def _loop(self):
        t0 = time.perf_counter()
        while self._running:
            try:
                s = self.monitor.sample()
                self.samples.append(NVMLSample(
                    timestamp_s=time.perf_counter() - t0,
                    power_w=s.power_draw_w,
                    temperature_c=s.temperature_c,
                    gpu_clock_mhz=s.gpu_clock_mhz,
                    mem_clock_mhz=s.mem_clock_mhz,
                    gpu_util_pct=s.gpu_utilization_pct,
                    mem_util_pct=s.mem_utilization_pct,
                ))
            except Exception:
                pass
            time.sleep(self.interval_s)

    def summary(self) -> dict:
        """Compute summary statistics from all samples."""
        if not self.samples:
            return {}
        powers = [s.power_w for s in self.samples]
        temps = [s.temperature_c for s in self.samples]
        clocks = [s.gpu_clock_mhz for s in self.samples]
        duration = self.samples[-1].timestamp_s if self.samples else 0
        return {
            "power_avg_w": sum(powers) / len(powers),
            "power_max_w": max(powers),
            "power_min_w": min(powers),
            "temp_avg_c": sum(temps) / len(temps),
            "temp_max_c": max(temps),
            "clock_avg_mhz": sum(clocks) / len(clocks),
            "clock_min_mhz": min(clocks),
            "clock_max_mhz": max(clocks),
            "duration_s": duration,
            "num_samples": len(self.samples),
            "energy_j": sum(s.power_w * self.interval_s for s in self.samples),
        }
