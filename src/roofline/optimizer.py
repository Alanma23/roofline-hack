"""
Target-driven hardware optimizer for MoE/GEMM inference workloads.

Given a workload and performance targets, sweeps TP/PP/EP/precision combinations
to find configurations that meet latency/throughput goals, ranked by efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .inference_sizing import compute_inference_sizing


@dataclass
class OptimizationTarget:
    """Performance targets for the optimizer search."""
    latency_ms: Optional[float] = None           # Max allowed latency
    throughput_tok_s: Optional[float] = None     # Min required tokens/s
    max_nodes: int = 8
    max_asics: int = 64


@dataclass
class RankedConfig:
    """A ranked hardware+parallelism configuration."""
    tp: int
    pp: int
    ep: int
    nodes: int
    precision: str
    predicted_latency_ms: float
    predicted_throughput_tok_s: float
    bottleneck: str
    pareto_optimal: bool
    score: float                    # composite: lower is better (latency * cost)
    ep_uses_internode: bool = False
    network_time_ms: float = 0.0
    ep_time_ms: float = 0.0
    note: str = ""


def _flag_pareto(configs: List[RankedConfig]) -> List[RankedConfig]:
    """Mark Pareto-optimal configs: not dominated on (latency, asics)."""
    for i, cfg in enumerate(configs):
        cfg_asics = cfg.tp * cfg.pp
        dominated = any(
            other.predicted_latency_ms <= cfg.predicted_latency_ms
            and (other.tp * other.pp) <= cfg_asics
            and (
                other.predicted_latency_ms < cfg.predicted_latency_ms
                or (other.tp * other.pp) < cfg_asics
            )
            for j, other in enumerate(configs)
            if j != i
        )
        cfg.pareto_optimal = not dominated
    return configs


def optimize_hardware(
    workload: Dict,
    target: OptimizationTarget,
    config_space: Optional[Dict] = None,
) -> List[RankedConfig]:
    """
    Grid-sweep TP/PP/EP/precision space and return configs meeting the target,
    ranked by composite score (latency_closeness * cost_efficiency).

    Args:
        workload: WorkloadSpec dict (model, precision, phase, batch, seq_len, moe?)
        target: OptimizationTarget constraints
        config_space: Override candidate sets. Keys: tp, pp, ep, precision, hardware

    Returns:
        List of RankedConfig, sorted by score (ascending = better).
    """
    cs = config_space or {}
    tp_candidates: List[int] = cs.get("tp", [1, 2, 4, 8])
    pp_candidates: List[int] = cs.get("pp", [1, 2, 4])
    ep_candidates: List[int] = cs.get("ep", [1, 2, 4, 8])
    precision_candidates: List[str] = cs.get("precision", [workload.get("precision", {}).get("computeAs", "FP16")])
    hardware: Dict = cs.get("hardware", {})

    gpus_per_node: int = max(1, int(cs.get("gpus_per_node", 8)))

    results: List[RankedConfig] = []

    for tp in tp_candidates:
        for pp in pp_candidates:
            for ep in ep_candidates:
                asics = tp * pp
                if asics > target.max_asics:
                    continue
                nodes = max(1, (asics + gpus_per_node - 1) // gpus_per_node)
                if nodes > target.max_nodes:
                    continue

                for prec in precision_candidates:
                    # Build request with this parallelism
                    wl = dict(workload)
                    wl_prec = dict(wl.get("precision", {}))
                    wl_prec["computeAs"] = prec
                    wl = {**wl, "precision": wl_prec}

                    req = {
                        "workload": wl,
                        "hardware": hardware,
                        "parallel": {
                            "tp": tp,
                            "pp": pp,
                            "ep": ep,
                            "max_asics": target.max_asics,
                        },
                        "network": cs.get("network", {}),
                    }

                    try:
                        r = compute_inference_sizing(req, include_recommendations=False)
                    except Exception:
                        continue

                    lat_ms = r["time"]["end_to_end_ms"]
                    toks = r["time"].get("tokens_per_s", 0.0)
                    bottleneck = r["bottleneck"]
                    net_ms = r["time"].get("network_ms", 0.0)
                    ep_ms = r.get("collective_breakdown", {}).get("ep_time_ms", 0.0)

                    # Filter: must meet targets if specified
                    if target.latency_ms is not None and lat_ms > target.latency_ms:
                        continue
                    if target.throughput_tok_s is not None and toks < target.throughput_tok_s:
                        continue

                    # Score: penalize high latency and high ASIC count
                    # Normalize latency to target (or 1 if no target)
                    lat_norm = lat_ms / target.latency_ms if target.latency_ms else lat_ms
                    cost_norm = asics / target.max_asics
                    score = lat_norm * (0.7) + cost_norm * (0.3)

                    results.append(RankedConfig(
                        tp=tp,
                        pp=pp,
                        ep=ep,
                        nodes=nodes,
                        precision=prec,
                        predicted_latency_ms=lat_ms,
                        predicted_throughput_tok_s=toks,
                        bottleneck=bottleneck,
                        pareto_optimal=False,   # filled below
                        score=score,
                        ep_uses_internode=ep > gpus_per_node,
                        network_time_ms=net_ms,
                        ep_time_ms=ep_ms,
                    ))

    results.sort(key=lambda c: c.score)
    results = _flag_pareto(results)
    return results


def suggest_next_step(run_history: List[Dict]) -> Dict:
    """
    Analyze bottleneck distribution across run history and recommend the next
    configuration or hardware change to try.

    Args:
        run_history: List of run dicts, each with keys:
            - bottleneck: "compute" | "memory" | "network"
            - tp, pp, ep, latency_ms, network_time_ms, ep_time_ms, ep_uses_internode

    Returns:
        Suggestion dict with: action, reason, suggested_change
    """
    if not run_history:
        return {
            "action": "start_sweep",
            "reason": "No run history yet",
            "suggested_change": "Run an EP sweep: ep ∈ {1,2,4,8} with intranode NVLink",
        }

    bottlenecks = [r.get("bottleneck", "unknown") for r in run_history]
    last = run_history[-1]

    # Tally bottleneck types
    network_count = bottlenecks.count("network")
    memory_count = bottlenecks.count("memory")
    compute_count = bottlenecks.count("compute")

    ep_uses_internode = last.get("ep_uses_internode", False)
    ep = last.get("ep", 1)
    tp = last.get("tp", 1)

    if network_count >= len(run_history) // 2 + 1:
        # Network is dominant bottleneck
        if ep_uses_internode:
            return {
                "action": "reduce_ep_or_upgrade_ib",
                "reason": f"EP={ep} crosses node boundary — all-to-all uses IB which is slow",
                "suggested_change": "Try EP≤GPUs_per_node to keep all-to-all on NVLink, or upgrade IB to ≥100 GB/s",
            }
        if ep > 1:
            return {
                "action": "reduce_ep",
                "reason": "EP all-to-all is dominating network time",
                "suggested_change": f"Reduce EP from {ep} to {max(1, ep // 2)}, or increase intranode BW",
            }
        return {
            "action": "reduce_tp_or_upgrade_link",
            "reason": "TP all-reduce is network bottleneck",
            "suggested_change": f"Reduce TP from {tp} to {max(1, tp // 2)}, or increase link BW",
        }

    if memory_count >= len(run_history) // 2 + 1:
        return {
            "action": "quantize_or_increase_bw",
            "reason": "Memory bandwidth is the dominant bottleneck",
            "suggested_change": "Try FP8 or INT4 quantization, or design ASIC with higher HBM BW (>4 TB/s)",
        }

    if compute_count >= len(run_history) // 2 + 1:
        return {
            "action": "reduce_batch_or_increase_flops",
            "reason": "Compute is the dominant bottleneck",
            "suggested_change": "Reduce batch size to shift to memory-bound regime, or use higher TFLOPS ASIC",
        }

    return {
        "action": "explore_ep_sweep",
        "reason": "Mixed bottlenecks — EP sweep recommended",
        "suggested_change": "Run EP ∈ {1,2,4,8} × TP ∈ {1,2} to find the network-compute sweet spot",
    }
