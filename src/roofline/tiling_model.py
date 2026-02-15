"""
GEMM tiling analysis model.

Models how tile dimensions (Tm, Tn, Tk) affect:
- Shared memory usage (double-buffered A/B tiles)
- Wave quantization (tiles mapped across SMs)
- SM occupancy estimate
- L2 cache hit rate estimate

This is the "systems depth" differentiator — shows understanding
of GPU microarchitecture beyond just the roofline formula.
"""

import math
from dataclasses import dataclass
from typing import List

from .calculator_shell import bytes_per_element


@dataclass
class TilingAnalysis:
    """Result of analyzing a single tiling configuration."""
    tile_m: int
    tile_n: int
    tile_k: int
    shared_mem_bytes: int
    tiles_total: int
    tiles_per_sm: float
    waves: float                     # number of waves across all SMs
    wave_efficiency: float           # 0-1, 1.0 = perfect (no partial wave)
    sm_occupancy_pct: float          # estimated occupancy
    l2_hit_rate_estimate: float      # 0-1
    efficiency_score: float          # 0-1 composite

    def to_dict(self) -> dict:
        return {
            "tile_m": self.tile_m,
            "tile_n": self.tile_n,
            "tile_k": self.tile_k,
            "shared_mem_bytes": self.shared_mem_bytes,
            "tiles_total": self.tiles_total,
            "waves": round(self.waves, 2),
            "wave_efficiency": round(self.wave_efficiency, 3),
            "sm_occupancy_pct": round(self.sm_occupancy_pct, 1),
            "l2_hit_rate_estimate": round(self.l2_hit_rate_estimate, 3),
            "efficiency_score": round(self.efficiency_score, 3),
        }


# Blackwell defaults
BLACKWELL_SM_COUNT = 160
BLACKWELL_SHARED_MEM_PER_SM_KB = 228  # Blackwell: 228 KB shared memory per SM
BLACKWELL_L2_CACHE_MB = 96.0


def analyze_tiling(
    M: int, N: int, K: int,
    tile_m: int, tile_n: int, tile_k: int,
    precision: str,
    sm_count: int = BLACKWELL_SM_COUNT,
    shared_mem_per_sm_kb: int = BLACKWELL_SHARED_MEM_PER_SM_KB,
    l2_cache_mb: float = BLACKWELL_L2_CACHE_MB,
) -> TilingAnalysis:
    """
    Analyze a GEMM tiling configuration.

    Models:
    1. Shared memory: tile_A[Tm,Tk] + tile_B[Tk,Tn], double-buffered
    2. Wave quantization: how many full waves of tiles across SMs
    3. Occupancy: based on shared memory per block vs SM limit
    4. L2 hit rate: does the GEMM working set fit in L2?
    """
    bpe = bytes_per_element(precision)

    # Shared memory for A tile and B tile, double-buffered
    smem_a = tile_m * tile_k * bpe
    smem_b = tile_k * tile_n * bpe
    shared_mem_bytes = int(2 * (smem_a + smem_b))

    # Tile grid
    tiles_m = math.ceil(M / tile_m)
    tiles_n = math.ceil(N / tile_n)
    tiles_total = tiles_m * tiles_n
    tiles_per_sm = tiles_total / sm_count

    # Wave quantization
    # Ideal: tiles_total is exact multiple of sm_count
    waves = tiles_total / sm_count
    full_waves = math.ceil(waves)
    wave_efficiency = waves / full_waves if full_waves > 0 else 1.0

    # Occupancy estimate
    smem_limit = shared_mem_per_sm_kb * 1024
    if shared_mem_bytes > 0:
        concurrent_blocks = min(4, smem_limit // shared_mem_bytes)
    else:
        concurrent_blocks = 4
    sm_occupancy_pct = min(100.0, concurrent_blocks * 25.0)

    # L2 hit rate: ratio of L2 cache to working set
    working_set = M * K * bpe + K * N * bpe + M * N * bpe
    l2_bytes = l2_cache_mb * 1024 * 1024
    l2_hit_rate = min(1.0, l2_bytes / max(working_set, 1))

    # Composite efficiency score
    efficiency_score = (
        0.40 * wave_efficiency
        + 0.30 * (sm_occupancy_pct / 100.0)
        + 0.30 * l2_hit_rate
    )

    return TilingAnalysis(
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        shared_mem_bytes=shared_mem_bytes,
        tiles_total=tiles_total,
        tiles_per_sm=tiles_per_sm,
        waves=waves,
        wave_efficiency=wave_efficiency,
        sm_occupancy_pct=sm_occupancy_pct,
        l2_hit_rate_estimate=l2_hit_rate,
        efficiency_score=efficiency_score,
    )


# Common tile sizes used by cuBLAS / CUTLASS
STANDARD_TILINGS = [
    (64, 64, 32),
    (64, 128, 32),
    (128, 64, 32),
    (128, 128, 32),
    (128, 256, 32),
    (256, 128, 32),
    (256, 256, 32),
    (64, 64, 64),
    (64, 128, 64),
    (128, 128, 64),
    (128, 256, 64),
    (256, 128, 64),
]


def sweep_tilings(
    M: int, N: int, K: int,
    precision: str,
    sm_count: int = BLACKWELL_SM_COUNT,
    shared_mem_per_sm_kb: int = BLACKWELL_SHARED_MEM_PER_SM_KB,
    l2_cache_mb: float = BLACKWELL_L2_CACHE_MB,
) -> List[TilingAnalysis]:
    """
    Sweep standard tile sizes for a given GEMM shape.
    Returns sorted by efficiency score (best first).
    """
    results = []
    for tm, tn, tk in STANDARD_TILINGS:
        if tm > M and M > 0:
            continue
        if tn > N and N > 0:
            continue
        if tk > K and K > 0:
            continue
        analysis = analyze_tiling(
            M, N, K, tm, tn, tk, precision,
            sm_count=sm_count,
            shared_mem_per_sm_kb=shared_mem_per_sm_kb,
            l2_cache_mb=l2_cache_mb,
        )
        results.append(analysis)
    results.sort(key=lambda a: a.efficiency_score, reverse=True)
    return results


if __name__ == "__main__":
    M, N, K = 4096, 4096, 4096
    precision = "FP16"

    print("=" * 80)
    print(f"Tiling Analysis: GEMM {M}x{N}x{K} [{precision}]")
    print(f"Hardware: {BLACKWELL_SM_COUNT} SMs, {BLACKWELL_SHARED_MEM_PER_SM_KB} KB smem/SM, "
          f"{BLACKWELL_L2_CACHE_MB} MB L2")
    print("=" * 80)
    print(f"{'Tile':>16s} {'SMEM':>8s} {'Tiles':>6s} {'Waves':>7s} "
          f"{'WaveEff':>8s} {'Occup':>7s} {'L2Hit':>6s} {'Score':>7s}")
    print("-" * 80)

    for a in sweep_tilings(M, N, K, precision):
        print(f"  {a.tile_m:3d}x{a.tile_n:3d}x{a.tile_k:2d}  "
              f"{a.shared_mem_bytes:>7,d}  "
              f"{a.tiles_total:>5d}  "
              f"{a.waves:>6.1f}  "
              f"{a.wave_efficiency:>7.3f}  "
              f"{a.sm_occupancy_pct:>5.0f}%  "
              f"{a.l2_hit_rate_estimate:>5.3f}  "
              f"{a.efficiency_score:>6.3f}")
