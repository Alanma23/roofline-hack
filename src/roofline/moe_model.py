"""
Mixture of Experts (MoE) Performance Model.

Models expert routing, load balancing, and expert parallelism (EP) communication
for MoE architectures like DeepSeek-V3, Mixtral, and Grok.

Based on:
- Switch Transformer (Google 2021)
- DeepSpeed-MoE
- JAX ML Scaling Book
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

from .calculator_shell import HardwareSpec, RooflineCalculator, bytes_per_element


@dataclass
class MoEConfig:
    """MoE architecture configuration."""
    num_experts: int = 8
    experts_per_token: int = 2  # Top-K
    expert_ffn_dim: int = 14336
    capacity_factor: float = 1.25  # Load balancing buffer
    load_balance_loss_weight: float = 0.01
    expert_parallel: int = 1  # EP degree (expert parallelism)


class MoECalculator:
    """Roofline-based MoE performance predictor."""

    def __init__(self, hardware: HardwareSpec, use_efficiency: bool = True):
        """
        Initialize MoE calculator.

        Args:
            hardware: Hardware specification
            use_efficiency: Apply empirical efficiency factors (realistic vs ideal)
        """
        self.hardware = hardware
        self.base_calc = RooflineCalculator(hardware, use_efficiency)

    def predict_moe_layer(
        self,
        batch: int,
        seq_len: int,
        hidden_dim: int,
        num_experts: int,
        experts_per_token: int,
        expert_ffn_dim: int,
        precision: str,
        capacity_factor: float = 1.25,
        expert_parallel: int = 1,
        load_imbalance: float = 0.15,
        network_bw_gbs: float = 900.0,  # NVLink/ICI bandwidth
        network_latency_us: float = 3.0,  # Network latency
    ) -> Dict:
        """
        Predict MoE layer performance.

        Args:
            batch: Batch size
            seq_len: Sequence length
            hidden_dim: Hidden dimension
            num_experts: Total number of experts
            experts_per_token: Top-K experts activated per token
            expert_ffn_dim: Expert FFN intermediate dimension
            precision: Precision format (FP16, FP8, etc.)
            capacity_factor: Load balancing buffer multiplier
            expert_parallel: EP degree (experts distributed across devices)
            load_imbalance: Expected load imbalance (0-1 std/mean)
            network_bw_gbs: Network bandwidth for EP all-to-all (GB/s)
            network_latency_us: Network latency for EP all-to-all (μs)

        Returns:
            Dict with performance breakdown and bottleneck analysis
        """
        total_tokens = batch * seq_len
        bpe = bytes_per_element(precision)

        # ═══════════════════════════════════════════════
        # 1. ROUTER OVERHEAD
        # ═══════════════════════════════════════════════
        # Router is typically a small MLP: hidden_dim -> num_experts
        router_flops = 2 * total_tokens * hidden_dim * num_experts
        router_bytes = total_tokens * hidden_dim * bpe  # Input
        router_bytes += num_experts * hidden_dim * bpe  # Router weights
        router_bytes += total_tokens * num_experts * 4  # Router logits (FP32)

        # Top-K sorting overhead (assume O(N log K) comparisons)
        topk_sorting_flops = total_tokens * num_experts * math.log2(experts_per_token)
        router_flops += topk_sorting_flops

        # Predict router time
        router_pred = self.base_calc.predict_gemm(
            M=total_tokens,
            N=num_experts,
            K=hidden_dim,
            precision=precision
        )
        router_time_us = router_pred["predicted_time_us"]
        router_tflops = router_pred["predicted_tflops"]

        # ═══════════════════════════════════════════════
        # 2. EXPERT COMPUTATION
        # ═══════════════════════════════════════════════
        # Tokens per expert (with capacity factor for load balancing)
        avg_tokens_per_expert = (total_tokens * experts_per_token) / num_experts
        tokens_per_expert = avg_tokens_per_expert * capacity_factor

        # Expert FFN: SwiGLU (gate + up + down)
        # gate: [tokens, hidden] @ [hidden, ffn] = [tokens, ffn]
        # up: [tokens, hidden] @ [hidden, ffn] = [tokens, ffn]
        # down: [tokens, ffn] @ [ffn, hidden] = [tokens, hidden]
        # Total: 6 * tokens * hidden * ffn (3 matmuls, each 2*M*N*K)
        expert_flops_per_expert = 6 * tokens_per_expert * hidden_dim * expert_ffn_dim

        # Total FLOPs across all experts (parallelized across devices)
        total_expert_flops = expert_flops_per_expert * num_experts

        # Predict expert computation time (parallelized across EP devices)
        # With EP, each device handles num_experts / EP experts
        experts_per_device = num_experts / expert_parallel
        expert_flops_per_device = expert_flops_per_expert * experts_per_device

        # Predict expert time (one expert's workload)
        expert_pred = self.base_calc.predict_gemm(
            M=int(tokens_per_expert),
            N=expert_ffn_dim,
            K=hidden_dim,
            precision=precision
        )
        expert_time_per_expert_us = expert_pred["predicted_time_us"]

        # Time for all experts on one device (sequential)
        expert_time_per_device_us = expert_time_per_expert_us * experts_per_device

        # Apply load imbalance penalty
        # If load is uneven, some devices idle while others finish
        imbalance_penalty = 1 + load_imbalance
        expert_time_us = expert_time_per_device_us * imbalance_penalty

        # ═══════════════════════════════════════════════
        # 3. EXPERT PARALLELISM COMMUNICATION (All-to-All)
        # ═══════════════════════════════════════════════
        # With EP, tokens need to be routed to experts on different devices
        # All-to-all scatter/gather: each device sends/receives tokens
        # Bytes transferred = total_tokens * hidden_dim * bpe * 2 * (EP - 1) / EP
        all_to_all_bytes = 0
        all_to_all_time_us = 0

        if expert_parallel > 1:
            # Each device sends/receives (total_tokens / EP) tokens to (EP - 1) other devices
            tokens_per_device = total_tokens / expert_parallel
            all_to_all_bytes = tokens_per_device * hidden_dim * bpe * 2 * (expert_parallel - 1)

            # All-to-all time: bandwidth + latency
            # Bandwidth time: bytes / (bandwidth * 1e9) * 1e6 = μs
            bandwidth_time_us = all_to_all_bytes / (network_bw_gbs * 1e9) * 1e6

            # Latency time: log2(EP) hops for all-to-all
            latency_time_us = network_latency_us * math.log2(expert_parallel)

            all_to_all_time_us = bandwidth_time_us + latency_time_us

        # ═══════════════════════════════════════════════
        # 4. TOTAL TIME AND BOTTLENECK
        # ═══════════════════════════════════════════════
        total_time_us = router_time_us + expert_time_us + all_to_all_time_us

        # Determine bottleneck
        bottleneck = "compute"
        if all_to_all_time_us > max(router_time_us, expert_time_us):
            bottleneck = "communication"
        elif load_imbalance > 0.2:
            bottleneck = "load_imbalance"

        # ═══════════════════════════════════════════════
        # 5. EXPERT UTILIZATION (Simulated)
        # ═══════════════════════════════════════════════
        # Simulate per-expert token counts with load imbalance
        # Assume Gaussian distribution: mean = avg_tokens_per_expert, std = load_imbalance * mean
        import random
        random.seed(42)  # Reproducible for demo
        expert_utilization = []
        for _ in range(min(num_experts, 64)):  # Limit to 64 for visualization
            # Simulate token count with imbalance
            tokens = max(0, random.gauss(avg_tokens_per_expert, load_imbalance * avg_tokens_per_expert))
            util_pct = min(100.0, (tokens / (avg_tokens_per_expert * capacity_factor)) * 100)
            expert_utilization.append(round(util_pct, 1))

        # ═══════════════════════════════════════════════
        # 6. RECOMMENDATIONS
        # ═══════════════════════════════════════════════
        recommendations = []

        if load_imbalance > 0.2:
            recommendations.append(
                f"High load imbalance ({load_imbalance:.1%}). Consider auxiliary load balancing loss or expert dropout."
            )

        if all_to_all_time_us > 0 and all_to_all_time_us / total_time_us > 0.15:
            comm_overhead_pct = (all_to_all_time_us / total_time_us) * 100
            recommendations.append(
                f"Communication overhead: {comm_overhead_pct:.1f}%. Consider reducing EP degree or using expert replication."
            )

        if experts_per_token / num_experts < 0.1:
            sparsity_pct = (1 - experts_per_token / num_experts) * 100
            recommendations.append(
                f"High sparsity ({sparsity_pct:.1f}%). Expect {experts_per_token / num_experts:.2%} of dense compute cost."
            )

        if capacity_factor > 1.5:
            recommendations.append(
                f"High capacity factor ({capacity_factor:.2f}). May waste compute on dropped tokens."
            )

        # ═══════════════════════════════════════════════
        # RETURN RESULTS
        # ═══════════════════════════════════════════════
        sparsity = 1 - (experts_per_token / num_experts)

        return {
            # Router
            "router_flops": router_flops,
            "router_time_us": router_time_us,
            "router_tflops": router_tflops,
            "routing_overhead_pct": (router_time_us / total_time_us) * 100,

            # Experts
            "expert_flops": total_expert_flops,
            "expert_time_us": expert_time_us,
            "expert_tflops": (total_expert_flops / expert_time_us) / 1e6 if expert_time_us > 0 else 0,

            # Communication
            "all_to_all_bytes": all_to_all_bytes,
            "all_to_all_time_us": all_to_all_time_us,
            "communication_overhead_pct": (all_to_all_time_us / total_time_us) * 100 if total_time_us > 0 else 0,

            # Totals
            "total_flops": router_flops + total_expert_flops,
            "total_time_us": total_time_us,
            "total_time_ms": total_time_us / 1000,

            # Load balancing
            "load_imbalance": load_imbalance,
            "capacity_factor": capacity_factor,
            "avg_tokens_per_expert": avg_tokens_per_expert,
            "expert_utilization": expert_utilization,

            # Sparsity
            "sparsity": sparsity,
            "active_experts_pct": (1 - sparsity) * 100,

            # Bottleneck
            "bottleneck": bottleneck,
            "recommendations": recommendations,

            # Debugging
            "debug": {
                "tokens_per_expert": tokens_per_expert,
                "experts_per_device": experts_per_device,
                "expert_time_per_expert_us": expert_time_per_expert_us,
                "imbalance_penalty": imbalance_penalty,
            },
        }


if __name__ == "__main__":
    # Example: DeepSeek-V3 MoE layer
    from .hardware_registry import get_hardware

    hw = get_hardware("b10")
    calc = MoECalculator(hw)

    print("=" * 80)
    print("MoE Performance Model - DeepSeek-V3 Example")
    print("=" * 80)

    result = calc.predict_moe_layer(
        batch=1,
        seq_len=4096,
        hidden_dim=7168,
        num_experts=256,
        experts_per_token=8,
        expert_ffn_dim=1536,
        precision="FP8_E4M3",
        capacity_factor=1.3,
        expert_parallel=8,
        load_imbalance=0.12,
    )

    print(f"\n[Router]")
    print(f"  Time: {result['router_time_us']:.1f} μs ({result['routing_overhead_pct']:.1f}%)")
    print(f"  FLOPs: {result['router_flops'] / 1e9:.2f} GFLOPs")

    print(f"\n[Experts]")
    print(f"  Time: {result['expert_time_us']:.1f} μs")
    print(f"  FLOPs: {result['expert_flops'] / 1e12:.2f} TFLOPs")
    print(f"  Throughput: {result['expert_tflops']:.1f} TFLOPS")
    print(f"  Sparsity: {result['sparsity'] * 100:.1f}% ({result['active_experts_pct']:.1f}% active)")

    print(f"\n[Communication (EP={8})]")
    print(f"  All-to-All Bytes: {result['all_to_all_bytes'] / 1e6:.2f} MB")
    print(f"  All-to-All Time: {result['all_to_all_time_us']:.1f} μs ({result['communication_overhead_pct']:.1f}%)")

    print(f"\n[Total]")
    print(f"  Time: {result['total_time_ms']:.2f} ms")
    print(f"  Bottleneck: {result['bottleneck']}")

    print(f"\n[Recommendations]")
    for rec in result['recommendations']:
        print(f"  • {rec}")
