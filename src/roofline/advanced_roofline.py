"""
Advanced Roofline Simulator with Hardware Realism.

Implements three critical complexity levels:
1. Hierarchical Memory (SRAM vs DRAM bandwidth/capacity)
2. Utilization/Padding (array dimension mismatches)
3. Mixed Precision (separate precision for inputs vs accumulation)

Based on realistic accelerator constraints that naive roofline models miss.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


# ═══════════════════════════════════════════════
#  HARDWARE SPECS - Advanced
# ═══════════════════════════════════════════════

@dataclass
class MemoryTier:
    """Memory hierarchy tier (SRAM, DRAM, etc.)."""
    name: str
    bandwidth_gbs: float  # GB/s
    capacity_bytes: int


@dataclass
class AdvancedAcceleratorSpec:
    """
    Advanced hardware specification with memory hierarchy and physical array dimensions.

    This models real accelerator constraints:
    - Memory hierarchy (SRAM vs DRAM)
    - Physical systolic array dimensions (for utilization calculation)
    - Frequency and precision support
    """
    name: str
    frequency_ghz: float

    # LEVEL 1: Memory Hierarchy
    sram: MemoryTier  # On-chip Global Buffer (L1/L2)
    dram: MemoryTier  # Off-chip HBM/DDR

    # LEVEL 2: Physical Array Dimensions (for Tiling/Padding)
    # Systolic arrays are typically square (Size x Size)
    array_dim_m: int  # Height of physical array
    array_dim_n: int  # Width of physical array
    array_dim_k: int = 1  # Systolic arrays process 1 K-step per cycle per PE

    # Additional metadata
    metadata: Dict = field(default_factory=dict)

    def get_peak_tops(self, precision_bits: int) -> float:
        """
        Calculate peak TOPS for given precision.

        Simplified model: Reducing precision increases effective array size
        due to SIMD packing (e.g., 4x INT8 vs 1x FP32 in same area).

        Args:
            precision_bits: Bit width (8, 16, 32, etc.)

        Returns:
            Peak TOPS (trillion ops/second)
        """
        # Packing factor: How many operations fit in a fixed logic area
        # e.g., 32-bit / 8-bit = 4x packing for INT8 vs FP32
        packing_factor = 32 // precision_bits

        # Total MACs = array size * packing factor
        total_macs = self.array_dim_m * self.array_dim_n * packing_factor

        # TOPS = (MACs * 2 ops/MAC * frequency) / 1000
        return (total_macs * 2 * self.frequency_ghz) / 1000.0


# ═══════════════════════════════════════════════
#  WORKLOAD SPECIFICATION
# ═══════════════════════════════════════════════

@dataclass
class LayerSpec:
    """Matrix multiplication workload: C[M,N] = A[M,K] @ B[K,N]"""
    name: str
    M: int  # Output rows
    N: int  # Output columns
    K: int  # Inner dimension

    def get_useful_ops(self) -> int:
        """Actual useful operations (2*M*N*K for matmul)."""
        return 2 * self.M * self.N * self.K

    def get_input_elements(self) -> int:
        """Total input elements (A + B matrices)."""
        return self.M * self.K + self.K * self.N

    def get_output_elements(self) -> int:
        """Total output elements (C matrix)."""
        return self.M * self.N


# ═══════════════════════════════════════════════
#  ADVANCED ROOFLINE SIMULATOR
# ═══════════════════════════════════════════════

class AdvancedRooflineSim:
    """
    Advanced roofline simulator with hardware realism.

    Models:
    1. Memory hierarchy (SRAM vs DRAM)
    2. Array utilization (padding penalty)
    3. Mixed precision (input vs accumulation precision)
    """

    def __init__(self, hw: AdvancedAcceleratorSpec):
        self.hw = hw

    def run_analysis(
        self,
        layer: LayerSpec,
        input_precision_bits: int = 8,
        accumulator_precision_bits: int = 32,
        verbose: bool = True
    ) -> Dict:
        """
        Run advanced roofline analysis on a layer.

        Args:
            layer: Matrix multiplication workload
            input_precision_bits: Bit width for inputs (weights/activations)
            accumulator_precision_bits: Bit width for partial sums/output
            verbose: Print detailed analysis

        Returns:
            Dict with performance metrics and bottleneck analysis
        """

        if verbose:
            print(f"\n{'='*70}")
            print(f"Advanced Roofline Analysis: {layer.name}")
            print(f"Input: {input_precision_bits}-bit | Accumulator: {accumulator_precision_bits}-bit")
            print(f"{'='*70}")

        # ═══════════════════════════════════════════════
        # LEVEL 2: Array Utilization (Padding Penalty)
        # ═══════════════════════════════════════════════
        # Hardware arrays have fixed dimensions. If workload doesn't match,
        # we must pad with zeros, wasting cycles.

        tiled_M = self._ceil_div(layer.M, self.hw.array_dim_m) * self.hw.array_dim_m
        tiled_N = self._ceil_div(layer.N, self.hw.array_dim_n) * self.hw.array_dim_n

        # Total operations including padded zeros
        total_hw_ops = 2 * tiled_M * tiled_N * layer.K
        useful_ops = layer.get_useful_ops()

        # Utilization: What fraction of hardware cycles do useful work?
        utilization = useful_ops / total_hw_ops if total_hw_ops > 0 else 0.0

        if verbose:
            print(f"\n[1] Array Utilization")
            print(f"  Workload:     M={layer.M}, N={layer.N}, K={layer.K}")
            print(f"  Padded to:    M={tiled_M}, N={tiled_N}")
            print(f"  Utilization:  {utilization*100:.1f}%")
            if utilization < 0.5:
                print(f"  ⚠️  WARNING: Low utilization! Padding wastes {(1-utilization)*100:.1f}% of cycles")

        # ═══════════════════════════════════════════════
        # LEVEL 3: Mixed Precision Traffic
        # ═══════════════════════════════════════════════
        # Inputs are low precision (INT8), but accumulation is high precision (INT32).
        # This creates asymmetric read/write bandwidth usage.

        # Input traffic (A and B matrices)
        input_bytes = (layer.M * layer.K + layer.K * layer.N) * (input_precision_bits / 8.0)

        # Output traffic (C matrix with high-precision partial sums)
        output_bytes = (layer.M * layer.N) * (accumulator_precision_bits / 8.0)

        total_data_bytes = input_bytes + output_bytes

        if verbose:
            print(f"\n[2] Mixed Precision Traffic")
            print(f"  Input:        {input_bytes / 1e6:.2f} MB ({input_precision_bits}-bit)")
            print(f"  Output:       {output_bytes / 1e6:.2f} MB ({accumulator_precision_bits}-bit)")
            print(f"  Total:        {total_data_bytes / 1e6:.2f} MB")

        # ═══════════════════════════════════════════════
        # LEVEL 1: Hierarchical Memory Check
        # ═══════════════════════════════════════════════
        # Does the entire working set fit in on-chip SRAM?
        # If yes, we use SRAM bandwidth (much faster).
        # If no, we're limited by DRAM bandwidth.

        if total_data_bytes <= self.hw.sram.capacity_bytes:
            active_mem = self.hw.sram
            mem_tier_name = "SRAM (On-Chip)"
        else:
            active_mem = self.hw.dram
            mem_tier_name = "DRAM (Off-Chip)"

        if verbose:
            print(f"\n[3] Memory Tier")
            print(f"  Active:       {mem_tier_name}")
            print(f"  Bandwidth:    {active_mem.bandwidth_gbs:.0f} GB/s")
            print(f"  Capacity:     {active_mem.capacity_bytes / 1e6:.0f} MB")
            if active_mem == self.hw.sram:
                print(f"  ✅ Cache hit! Using ultra-fast on-chip memory")
            else:
                print(f"  ⚠️  Off-chip access: Limited by DRAM bandwidth")

        # ═══════════════════════════════════════════════
        # ROOFLINE CALCULATION
        # ═══════════════════════════════════════════════

        # Peak compute (adjusted for input precision)
        peak_compute_tops = self.hw.get_peak_tops(input_precision_bits)

        # Effective compute (adjusted for utilization)
        effective_compute_tops = peak_compute_tops * utilization

        # Arithmetic intensity (ops per byte)
        arithmetic_intensity = useful_ops / total_data_bytes if total_data_bytes > 0 else 0

        # Memory-bound performance
        mem_bound_tops = (active_mem.bandwidth_gbs * arithmetic_intensity) / 1000.0

        # Actual performance is the minimum of compute and memory limits
        actual_tops = min(effective_compute_tops, mem_bound_tops)

        # Determine bottleneck
        if actual_tops == effective_compute_tops:
            bottleneck = "COMPUTE" if utilization > 0.9 else "UTILIZATION"
        else:
            bottleneck = "MEMORY"

        # Calculate latency
        latency_ms = (useful_ops / (actual_tops * 1e9)) if actual_tops > 0 else float('inf')

        if verbose:
            print(f"\n[4] Performance Analysis")
            print(f"  Peak Compute:      {peak_compute_tops:.2f} TOPS")
            print(f"  Effective Compute: {effective_compute_tops:.2f} TOPS (after {utilization*100:.1f}% util)")
            print(f"  Memory Bound:      {mem_bound_tops:.2f} TOPS")
            print(f"  Arithmetic Int:    {arithmetic_intensity:.2f} ops/byte")
            print(f"\n  ➤ Actual Performance: {actual_tops:.2f} TOPS")
            print(f"  ➤ Latency:            {latency_ms:.3f} ms")
            print(f"  ➤ Bottleneck:         {bottleneck}")

            if bottleneck == "UTILIZATION":
                print(f"\n  💡 Recommendation: Batch workloads or use smaller array dimensions")
            elif bottleneck == "MEMORY":
                print(f"\n  💡 Recommendation: Increase arithmetic intensity or use on-chip tiling")

        return {
            # Performance
            "actual_tops": actual_tops,
            "latency_ms": latency_ms,
            "latency_us": latency_ms * 1000,

            # Roofline components
            "peak_compute_tops": peak_compute_tops,
            "effective_compute_tops": effective_compute_tops,
            "mem_bound_tops": mem_bound_tops,
            "arithmetic_intensity": arithmetic_intensity,

            # Utilization
            "utilization": utilization,
            "padded_m": tiled_M,
            "padded_n": tiled_N,
            "useful_ops": useful_ops,
            "total_hw_ops": total_hw_ops,

            # Memory
            "mem_tier": mem_tier_name,
            "total_data_bytes": total_data_bytes,
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,

            # Bottleneck
            "bottleneck": bottleneck,

            # Hardware info
            "hardware": self.hw.name,
            "input_precision_bits": input_precision_bits,
            "accumulator_precision_bits": accumulator_precision_bits,
        }

    def _ceil_div(self, x: int, y: int) -> int:
        """Ceiling division: ⌈x/y⌉"""
        return (x + y - 1) // y

    def compare_batch_sizes(
        self,
        layer_template: LayerSpec,
        batch_sizes: List[int],
        input_precision_bits: int = 8,
        accumulator_precision_bits: int = 32
    ) -> List[Dict]:
        """
        Compare performance across different batch sizes.

        Useful for understanding how utilization changes with batch size.

        Args:
            layer_template: Base layer spec (will modify M dimension)
            batch_sizes: List of batch sizes to test
            input_precision_bits: Input precision
            accumulator_precision_bits: Accumulator precision

        Returns:
            List of analysis results for each batch size
        """
        results = []

        print(f"\n{'='*70}")
        print(f"Batch Size Sweep: {layer_template.name}")
        print(f"{'='*70}")

        for batch in batch_sizes:
            layer = LayerSpec(
                name=f"{layer_template.name}_B{batch}",
                M=batch,
                N=layer_template.N,
                K=layer_template.K
            )

            result = self.run_analysis(
                layer,
                input_precision_bits=input_precision_bits,
                accumulator_precision_bits=accumulator_precision_bits,
                verbose=False
            )

            results.append(result)

            print(f"  Batch={batch:4d}: {result['actual_tops']:6.2f} TOPS, "
                  f"Util={result['utilization']*100:5.1f}%, "
                  f"Latency={result['latency_ms']:7.3f}ms, "
                  f"Bottleneck={result['bottleneck']}")

        return results


# ═══════════════════════════════════════════════
#  HARDWARE PRESETS
# ═══════════════════════════════════════════════

def create_tpu_v3_like() -> AdvancedAcceleratorSpec:
    """
    TPU v3-like architecture.

    - 128x128 systolic array
    - 1 GHz frequency
    - 32 MB on-chip SRAM
    - 900 GB/s HBM bandwidth
    """
    return AdvancedAcceleratorSpec(
        name="TPU_v3_Like",
        frequency_ghz=1.0,
        array_dim_m=128,
        array_dim_n=128,
        sram=MemoryTier("SRAM", bandwidth_gbs=2000, capacity_bytes=32 * 1024 * 1024),
        dram=MemoryTier("HBM", bandwidth_gbs=900, capacity_bytes=16 * 1024 * 1024 * 1024),
    )


def create_blackwell_b10_advanced() -> AdvancedAcceleratorSpec:
    """
    Blackwell GB10 with estimated array dimensions.

    Based on:
    - Peak FP8: 164 TFLOPS
    - Bandwidth: 287 GB/s (LPDDR5X)
    - Estimated 64x64 array (rough guess)
    """
    return AdvancedAcceleratorSpec(
        name="Blackwell_GB10",
        frequency_ghz=1.2,  # Estimated
        array_dim_m=64,     # Estimated (164 TFLOPS / 1.2 GHz / 2 ops ≈ 68K MACs ≈ 64x64x16 packing)
        array_dim_n=64,
        sram=MemoryTier("L2", bandwidth_gbs=1500, capacity_bytes=16 * 1024 * 1024),  # Estimated 16 MB
        dram=MemoryTier("LPDDR5X", bandwidth_gbs=287, capacity_bytes=128 * 1024 * 1024 * 1024),
    )


# ═══════════════════════════════════════════════
#  COMMAND-LINE DEMO
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    print("="*70)
    print("Advanced Roofline Simulator - Demonstration")
    print("="*70)

    # Create hardware
    hw = create_tpu_v3_like()
    sim = AdvancedRooflineSim(hw)

    # Case A: Large batch (hardware friendly)
    print("\n" + "="*70)
    print("CASE A: Large Batch (M=1024) - Good Utilization")
    print("="*70)
    large_layer = LayerSpec("Large_GEMM", M=1024, N=1024, K=1024)
    sim.run_analysis(large_layer, input_precision_bits=8, accumulator_precision_bits=32)

    # Case B: Small batch (utilization killer)
    print("\n" + "="*70)
    print("CASE B: Small Batch (M=10) - Poor Utilization")
    print("="*70)
    small_layer = LayerSpec("Small_GEMM", M=10, N=1024, K=1024)
    sim.run_analysis(small_layer, input_precision_bits=8, accumulator_precision_bits=32)

    # Case C: Batch size sweep
    print("\n" + "="*70)
    print("CASE C: Batch Size Sweep")
    print("="*70)
    template = LayerSpec("Sweep", M=0, N=1024, K=1024)
    sim.compare_batch_sizes(
        template,
        batch_sizes=[1, 8, 16, 32, 64, 128, 256, 512, 1024],
        input_precision_bits=8,
        accumulator_precision_bits=32
    )
