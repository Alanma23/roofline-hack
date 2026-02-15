"""
Low-level Jetson Orin systems integration.

Provides direct access to:
- Power mode (nvpmodel)
- GPU frequency (sysfs devfreq)
- EMC/memory frequency
- CPU frequency
- Memory stats (shared CPU+GPU)
- jetson_clocks status

Used by scheduler and power-aware roofline.
"""

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple

# Sysfs paths - Jetson Orin uses 17000000.gpu, older Tegra uses 57000000.gpu
GPU_DEVFREQ_PATHS = [
    "/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu",
    "/sys/devices/gpu.0/devfreq/57000000.gpu",
    "/sys/class/devfreq/17000000.gpu",
]

EMC_DEVFREQ_PATHS = [
    "/sys/devices/platform/17000000.emc/devfreq/17000000.emc",
    "/sys/devices/platform/57000000.emc/devfreq/57000000.emc",
]

CPU_FREQ_PATH = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
MEMINFO_PATH = "/proc/meminfo"
NVPMODEL_CONF = "/etc/nvpmodel.conf"


@dataclass
class JetsonStatus:
    """Current Jetson hardware status."""

    is_jetson: bool = False
    power_mode_id: Optional[int] = None
    power_mode_name: Optional[str] = None
    power_budget_w: Optional[float] = None
    gpu_freq_mhz: Optional[float] = None
    gpu_freq_max_mhz: Optional[float] = None
    emc_freq_mhz: Optional[float] = None
    cpu_freq_mhz: Optional[float] = None
    mem_total_mb: Optional[float] = None
    mem_available_mb: Optional[float] = None
    mem_used_mb: Optional[float] = None
    jetson_clocks_active: bool = False
    cuda_device_name: Optional[str] = None
    cuda_memory_gb: Optional[float] = None
    cuda_memory_allocated_gb: Optional[float] = None
    errors: List[str] = field(default_factory=list)


def _read_sysfs(path: str, default: Optional[str] = None) -> Optional[str]:
    """Read a sysfs file. Returns None on error."""
    try:
        p = Path(path)
        if p.exists():
            return p.read_text().strip()
    except (OSError, PermissionError):
        pass
    return default


def _read_int(path: str) -> Optional[int]:
    s = _read_sysfs(path)
    if s is not None:
        try:
            return int(s)
        except ValueError:
            pass
    return None


def _detect_jetson() -> bool:
    """Check if running on Jetson (Tegra)."""
    return Path("/etc/nv_tegra_release").exists() or Path("/etc/nvpmodel.conf").exists()


def get_gpu_freq() -> Tuple[Optional[float], Optional[float]]:
    """Return (current_mhz, max_mhz). Tries multiple sysfs paths."""
    for base in GPU_DEVFREQ_PATHS:
        cur = _read_int(f"{base}/cur_freq")
        if cur is not None:
            cur_mhz = cur / 1000.0  # kHz -> MHz
            max_val = _read_int(f"{base}/max_freq")
            max_mhz = (max_val / 1000.0) if max_val else None
            return cur_mhz, max_mhz
    return None, None


def get_emc_freq() -> Optional[float]:
    """Return EMC (memory controller) frequency in MHz."""
    for base in EMC_DEVFREQ_PATHS:
        cur = _read_int(f"{base}/cur_freq")
        if cur is not None:
            return cur / 1000.0
    return None


def get_cpu_freq() -> Optional[float]:
    """Return CPU0 frequency in MHz."""
    val = _read_int(CPU_FREQ_PATH)
    return (val / 1000.0) if val else None


def get_memory_stats() -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (total_mb, available_mb, used_mb) from /proc/meminfo."""
    try:
        data = Path(MEMINFO_PATH).read_text()
        total = re.search(r"MemTotal:\s+(\d+)\s+kB", data)
        avail = re.search(r"MemAvailable:\s+(\d+)\s+kB", data)
        total_mb = int(total.group(1)) / 1024 if total else None
        avail_mb = int(avail.group(1)) / 1024 if avail else None
        used_mb = (total_mb - avail_mb) if (total_mb and avail_mb) else None
        return total_mb, avail_mb, used_mb
    except (OSError, AttributeError):
        return None, None, None


def get_nvpmodel_status() -> Tuple[Optional[int], Optional[str], Optional[float]]:
    """
    Return (mode_id, mode_name, power_budget_w) via nvpmodel -q.
    Requires nvpmodel in PATH (JetPack).
    """
    try:
        out = subprocess.run(
            ["nvpmodel", "-q"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return None, None, None
        mode_id = None
        mode_name = None
        power_w = None
        for line in out.stdout.splitlines():
            if "NV Power Mode" in line or "Power Mode" in line:
                m = re.search(r"Mode\s*(\d+)", line, re.I)
                if m:
                    mode_id = int(m.group(1))
            if "Mode" in line and ":" in line:
                parts = line.split(":")
                if len(parts) >= 2:
                    mode_name = parts[-1].strip()
            if "W" in line and re.search(r"\d+\s*W", line):
                m = re.search(r"(\d+(?:\.\d+)?)\s*W", line)
                if m:
                    power_w = float(m.group(1))
        return mode_id, mode_name, power_w
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None, None, None


def get_jetson_clocks_status() -> bool:
    """Check if jetson_clocks (max clocks) is active."""
    # When jetson_clocks is active, scaling governor is often "performance"
    # and min_freq == max_freq. Simplified: check if we can read status.
    try:
        out = subprocess.run(
            ["jetson_clocks", "--show"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return out.returncode == 0 and "enabled" in out.stdout.lower()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_cuda_memory() -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """Return (device_name, total_gb, allocated_gb). Requires torch+cuda."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None, None, None
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / 1e9
        alloc_gb = torch.cuda.memory_allocated(0) / 1e9
        return props.name, total_gb, alloc_gb
    except ImportError:
        return None, None, None


def get_jetson_status() -> JetsonStatus:
    """Aggregate all low-level Jetson status."""
    s = JetsonStatus()
    s.is_jetson = _detect_jetson()

    if not s.is_jetson:
        s.errors.append("Not running on Jetson")
        return s

    # Power mode
    mode_id, mode_name, power_w = get_nvpmodel_status()
    s.power_mode_id = mode_id
    s.power_mode_name = mode_name
    s.power_budget_w = power_w

    # GPU
    gpu_cur, gpu_max = get_gpu_freq()
    s.gpu_freq_mhz = gpu_cur
    s.gpu_freq_max_mhz = gpu_max

    # EMC
    s.emc_freq_mhz = get_emc_freq()

    # CPU
    s.cpu_freq_mhz = get_cpu_freq()

    # Memory
    total_mb, avail_mb, used_mb = get_memory_stats()
    s.mem_total_mb = total_mb
    s.mem_available_mb = avail_mb
    s.mem_used_mb = used_mb

    # jetson_clocks
    s.jetson_clocks_active = get_jetson_clocks_status()

    # CUDA (optional)
    dev, cuda_total, cuda_alloc = get_cuda_memory()
    s.cuda_device_name = dev
    s.cuda_memory_gb = cuda_total
    s.cuda_memory_allocated_gb = cuda_alloc

    return s


def set_power_mode(mode_id: int) -> bool:
    """
    Set nvpmodel power mode. Requires sudo.
    mode_id: 0=15W (Orin Nano 8GB), 1=7W
    """
    try:
        subprocess.run(
            ["sudo", "nvpmodel", "-m", str(mode_id)],
            check=True,
            timeout=10,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def set_jetson_clocks(enable: bool) -> bool:
    """Enable or disable jetson_clocks (max performance). Requires sudo."""
    try:
        subprocess.run(
            ["sudo", "jetson_clocks"],
            check=True,
            timeout=30,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def scale_roofline_for_power(
    base_bw_gb_s: float,
    base_fp16_tflops: float,
    base_int8_tflops: float,
    status: JetsonStatus,
) -> dict:
    """
    Scale roofline BW and FLOPS based on current power mode / GPU frequency.

    Jetson Orin Nano 8GB: 15W GPU 625 MHz, 7W GPU 408 MHz.
    Scale factor ≈ current_freq / max_freq (when not at max).
    """
    scale = 1.0
    if status.gpu_freq_mhz and status.gpu_freq_max_mhz and status.gpu_freq_max_mhz > 0:
        scale = status.gpu_freq_mhz / status.gpu_freq_max_mhz
    elif status.power_budget_w:
        # Heuristic: 7W ~0.65x of 15W for GPU perf
        if status.power_budget_w <= 7:
            scale = 0.65
        elif status.power_budget_w <= 10:
            scale = 0.85

    return {
        "peak_bandwidth_gb_s": base_bw_gb_s * scale,
        "peak_flops_tflops": {
            "FP16": base_fp16_tflops * scale,
            "INT8": base_int8_tflops * scale,
            "INT4": 0.0,
        },
        "scale_factor": scale,
    }


if __name__ == "__main__":
    status = get_jetson_status()
    print("Jetson:", status.is_jetson)
    print("Power mode:", status.power_mode_id, status.power_mode_name, status.power_budget_w, "W")
    print("GPU freq:", status.gpu_freq_mhz, "MHz (max:", status.gpu_freq_max_mhz, ")")
    print("EMC freq:", status.emc_freq_mhz, "MHz")
    print("CPU freq:", status.cpu_freq_mhz, "MHz")
    print("Memory:", status.mem_total_mb, "MB total,", status.mem_available_mb, "MB avail")
    print("jetson_clocks:", status.jetson_clocks_active)
    print("CUDA:", status.cuda_device_name, status.cuda_memory_gb, "GB")
    if status.errors:
        print("Errors:", status.errors)

    if status.is_jetson:
        scaled = scale_roofline_for_power(60, 2.7, 5.5, status)
        print("\nScaled roofline:", scaled)
