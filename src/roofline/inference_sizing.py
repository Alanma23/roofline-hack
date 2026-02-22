"""
Inference roofline sizing model (frontend parity).

Models:
- Megatron-style prefill/decode per-layer FLOPs/bytes
- TP all-reduce and PP boundary send communication
- Compute/memory/network three-way bottleneck
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .calculator_shell import bytes_per_element


def _pos_int(value, fallback: int) -> int:
    try:
        n = int(round(float(value)))
    except Exception:
        return fallback
    return max(1, n)


def _pos_float(value, fallback: float) -> float:
    try:
        n = float(value)
    except Exception:
        return fallback
    return n if n > 0 else fallback


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _resolve_memory_model(hardware: Dict) -> Dict:
    single_bw = _pos_float(hardware.get("mem_bw_gbs", hardware.get("bw", 1.0)), 1.0)
    model = hardware.get("memory_model", {}) or {}
    mode = "two_tier" if model.get("mode") == "two_tier" else "single"
    if mode != "two_tier":
        return {"mode": "single", "effective_bw_gbs": single_bw}

    hit = _clamp(float(model.get("onchip_hit_rate", 0.8)), 0.0, 1.0)
    onchip = _pos_float(model.get("onchip_bw_gbs", single_bw), single_bw)
    offchip = _pos_float(model.get("offchip_bw_gbs", single_bw), single_bw)
    denom = (hit / onchip) + ((1.0 - hit) / offchip)
    effective = (1.0 / denom) if denom > 0 else single_bw
    return {
        "mode": "two_tier",
        "effective_bw_gbs": effective,
        "onchip_bw_gbs": onchip,
        "offchip_bw_gbs": offchip,
        "onchip_hit_rate": hit,
    }


def _resolve_tokens(workload: Dict) -> int:
    seq_len = _pos_int(workload.get("seq_len", 1), 1)
    phase = workload.get("phase", "decode")
    if phase == "prefill":
        return _pos_int(workload.get("prefill_tokens", seq_len), seq_len)
    return _pos_int(workload.get("decode_tokens", 1), 1)


def _partition_layers(layer_count: int, pp: int) -> Tuple[List[int], List[Dict]]:
    pp = _pos_int(pp, 1)
    L = max(0, int(layer_count))
    base = L // pp
    rem = L % pp
    stage_of_layer: List[int] = [0] * L
    stages: List[Dict] = []
    cursor = 0
    for stage in range(pp):
        count = base + (1 if stage < rem else 0)
        start = cursor
        end = cursor + count - 1 if count > 0 else cursor - 1
        stages.append({"stage": stage, "start": start, "end": end, "count": count})
        for i in range(start, end + 1):
            if 0 <= i < L:
                stage_of_layer[i] = stage
        cursor += count
    return stage_of_layer, stages


def _ring_ar_bytes(tensor_bytes: float, tp: int) -> float:
    tp = _pos_int(tp, 1)
    if tp <= 1:
        return 0.0
    return 2.0 * (tp - 1) / tp * max(0.0, float(tensor_bytes))


def _ring_ar_time_s(tensor_bytes: float, tp: int, bw_gbs: float, lat_us: float) -> Tuple[float, float, float, float]:
    tp = _pos_int(tp, 1)
    if tp <= 1:
        return 0.0, 0.0, 0.0, 0.0
    bytes_sent = _ring_ar_bytes(tensor_bytes, tp)
    lat_s = 2.0 * (tp - 1) * max(0.0, lat_us) * 1e-6
    bw_s = bytes_sent / (_pos_float(bw_gbs, 1.0) * 1e9)
    return lat_s + bw_s, bytes_sent, lat_s, bw_s


def _p2p_time_s(tensor_bytes: float, bw_gbs: float, lat_us: float) -> Tuple[float, float, float]:
    bytes_sent = max(0.0, float(tensor_bytes))
    if bytes_sent <= 0:
        return 0.0, 0.0, 0.0
    lat_s = max(0.0, lat_us) * 1e-6
    bw_s = bytes_sent / (_pos_float(bw_gbs, 1.0) * 1e9)
    return lat_s + bw_s, lat_s, bw_s


def _compute_workload(workload: Dict) -> Dict:
    model = workload.get("model", {})
    precision = workload.get("precision", {})

    L = _pos_int(model.get("L", 1), 1)
    H = _pos_int(model.get("H", 1), 1)
    nh = _pos_int(model.get("nh", 1), 1)
    nkv = _pos_int(model.get("nkv", 1), 1)
    dh = _pos_int(model.get("dh", max(1, H // nh)), max(1, H // nh))
    dff = _pos_int(model.get("dff", max(1, (8 * H) // 3)), max(1, (8 * H) // 3))
    V = _pos_int(model.get("V", H), H)
    gate = bool(model.get("gate", True))

    B = _pos_int(workload.get("batch", 1), 1)
    S = _pos_int(workload.get("seq_len", 1), 1)
    T = _resolve_tokens(workload)

    w = precision.get("w", "FP16")
    a = precision.get("a", "FP16")
    kv = precision.get("kv", "FP16")
    cp = precision.get("computeAs", "FP16")

    w_b = float(bytes_per_element(w))
    a_b = float(bytes_per_element(a))
    kv_b = float(bytes_per_element(kv))
    o_b = a_b

    dkv = nkv * dh
    token_tensor_elems = B * T * H
    token_tensor_bytes = token_tensor_elems * a_b

    ops: List[Dict] = []
    layers: List[Dict] = []

    for layer in range(L):
        layer_weight = 0.0

        def add(name: str, flops: float, bytes_used: float, kind: str, weight_bytes: float = 0.0):
            ops.append(
                {
                    "name": name,
                    "flops": flops,
                    "bytes": bytes_used,
                    "ai": flops / bytes_used if bytes_used > 0 else 0.0,
                    "type": kind,
                    "cp": cp,
                    "layer": layer,
                    "weight_bytes": weight_bytes,
                }
            )

        def add_gemm(name: str, M: int, N: int, K: int):
            nonlocal layer_weight
            flops = 2.0 * M * N * K
            bytes_used = M * K * a_b + K * N * w_b + M * N * o_b
            weight_bytes = K * N * w_b
            layer_weight += weight_bytes
            add(name, flops, bytes_used, "gemm", weight_bytes)

        def add_attn(name: str, Sq: int, Skv: int, score: bool):
            flops = 2.0 * B * nh * Sq * Skv * dh
            if score:
                bytes_used = B * nh * Sq * dh * a_b + B * nkv * Skv * dh * kv_b + B * nh * Sq * Skv * a_b
            else:
                bytes_used = B * nh * Sq * Skv * a_b + B * nkv * Skv * dh * kv_b + B * nh * Sq * dh * o_b
            add(name, flops, bytes_used, "attention")

        def add_elem(name: str, flops: float, bytes_used: float):
            add(name, flops, bytes_used, "elementwise")

        add_gemm("q_proj", B * T, H, H)
        add_gemm("k_proj", B * T, dkv, H)
        add_gemm("v_proj", B * T, dkv, H)
        add_attn("qk_score", T, S, True)
        add_elem("softmax", 5.0 * B * nh * T * S, 2.0 * B * nh * T * S * a_b)
        add_attn("sv_prod", T, S, False)
        add_gemm("o_proj", B * T, H, H)
        add_elem("rmsnorm", 5.0 * B * T * H, 2.0 * B * T * H * a_b)

        if gate:
            add_gemm("gate_proj", B * T, dff, H)
            add_gemm("up_proj", B * T, dff, H)
            add_elem("silu_mul", 3.0 * B * T * dff, 3.0 * B * T * dff * a_b)
        else:
            add_gemm("up_proj", B * T, dff, H)

        add_gemm("down_proj", B * T, H, dff)
        add_elem("residual", 2.0 * B * T * H, 6.0 * B * T * H * a_b)

        layers.append(
            {
                "layer": layer,
                "input_bytes": token_tensor_bytes,
                "output_bytes": token_tensor_bytes,
                "weight_bytes": layer_weight,
                "tp_sync_points": [
                    {"name": "attn_out", "tensor_bytes": token_tensor_bytes},
                    {"name": "mlp_down", "tensor_bytes": token_tensor_bytes},
                ],
            }
        )

    logit_flops = 2.0 * B * T * V * H
    logit_bytes = B * T * H * a_b + H * V * w_b + B * T * V * o_b
    ops.append(
        {
            "name": "logit_proj",
            "flops": logit_flops,
            "bytes": logit_bytes,
            "ai": logit_flops / logit_bytes if logit_bytes > 0 else 0.0,
            "type": "gemm",
            "cp": cp,
            "layer": -1,
            "weight_bytes": H * V * w_b,
        }
    )

    total_flops = sum(op["flops"] for op in ops)
    total_bytes = sum(op["bytes"] for op in ops)

    return {
        "phase": workload.get("phase", "decode"),
        "token_count": T,
        "ops": ops,
        "layers": layers,
        "totals": {
            "flops": total_flops,
            "bytes": total_bytes,
            "ai": total_flops / total_bytes if total_bytes > 0 else 0.0,
        },
    }


def compute_inference_sizing(payload: Dict, include_recommendations: bool = True) -> Dict:
    workload = payload.get("workload", {})
    hardware = payload.get("hardware", {})
    parallel = payload.get("parallel", {})
    network = payload.get("network", {})

    data = _compute_workload(workload)

    peak_map = hardware.get("peak_tflops", {})
    memory_model = _resolve_memory_model(hardware)
    mem_bw_gbs = _pos_float(memory_model.get("effective_bw_gbs", hardware.get("mem_bw_gbs", 1.0)), 1.0)
    compute_as = workload.get("precision", {}).get("computeAs", "FP16")
    peak_tflops = _pos_float(
        peak_map.get(compute_as, peak_map.get("FP16", 1.0)),
        1.0,
    )

    compute_s = 0.0
    memory_s = 0.0
    kernel_s = 0.0
    for op in data["ops"]:
        t_comp = op["flops"] / (peak_tflops * 1e12)
        t_mem = op["bytes"] / (mem_bw_gbs * 1e9)
        compute_s += t_comp
        memory_s += t_mem
        kernel_s += max(t_comp, t_mem)

    tp = _pos_int(parallel.get("tp", 1), 1)
    pp = _pos_int(parallel.get("pp", 1), 1)
    max_asics = _pos_int(parallel.get("max_asics", 16), 16)

    tp_bw = _pos_float(network.get("tp_link_bw_gbs", 900), 900)
    tp_lat = max(0.0, float(network.get("tp_link_latency_us", 3)))
    pp_bw = _pos_float(network.get("pp_link_bw_gbs", tp_bw), tp_bw)
    pp_lat = max(0.0, float(network.get("pp_link_latency_us", tp_lat)))
    overlap = _clamp(float(network.get("overlap_fraction", 0.0)), 0.0, 1.0)

    stage_of_layer, stages = _partition_layers(len(data["layers"]), pp)
    boundaries = {s["end"] for s in stages[:-1] if s["count"] > 0}

    tp_count = 0
    tp_bytes = 0.0
    tp_time_s = 0.0
    tp_lat_s = 0.0

    pp_count = 0
    pp_bytes = 0.0
    pp_time_s = 0.0
    pp_lat_s = 0.0

    layer_io = []
    for idx, layer in enumerate(data["layers"]):
        layer_tp_bytes = 0.0
        layer_tp_time_s = 0.0
        for sync in layer["tp_sync_points"]:
            t_s, b_s, lat_s, _ = _ring_ar_time_s(sync["tensor_bytes"], tp, tp_bw, tp_lat)
            layer_tp_bytes += b_s
            layer_tp_time_s += t_s
            tp_lat_s += lat_s
            tp_count += 1

        tp_bytes += layer_tp_bytes
        tp_time_s += layer_tp_time_s

        layer_pp_bytes = 0.0
        if idx in boundaries and pp > 1:
            t_s, lat_s, _ = _p2p_time_s(layer["output_bytes"], pp_bw, pp_lat)
            layer_pp_bytes = layer["output_bytes"]
            pp_bytes += layer_pp_bytes
            pp_time_s += t_s
            pp_lat_s += lat_s
            pp_count += 1

        layer_io.append(
            {
                "layer": idx + 1,
                "stage": stage_of_layer[idx] + 1,
                "input_bytes": layer["input_bytes"],
                "output_bytes": layer["output_bytes"],
                "weight_bytes": layer["weight_bytes"],
                "tp_sync_bytes": layer_tp_bytes,
                "pp_boundary_send_bytes": layer_pp_bytes,
            }
        )

    network_s = tp_time_s + pp_time_s
    end_to_end_s = kernel_s + (1.0 - overlap) * network_s

    bottleneck = max(
        [("compute", compute_s), ("memory", memory_s), ("network", network_s)],
        key=lambda x: x[1],
    )[0]

    required: Dict[str, float] = {}
    if bottleneck == "network":
        target_s = max(compute_s, memory_s)
        variable = tp_bytes + pp_bytes
        if variable > 0 and target_s > (tp_lat_s + pp_lat_s):
            required["network_bw_gbs"] = variable / (target_s - tp_lat_s - pp_lat_s) / 1e9
    elif bottleneck == "memory":
        target_s = max(compute_s, network_s)
        if target_s > 0:
            required["mem_bw_gbs"] = data["totals"]["bytes"] / target_s / 1e9
            if memory_model.get("mode") == "two_tier":
                hit = _clamp(float(memory_model.get("onchip_hit_rate", 0.0)), 0.0, 1.0)
                one_minus = 1.0 - hit
                onchip = _pos_float(memory_model.get("onchip_bw_gbs", required["mem_bw_gbs"]), required["mem_bw_gbs"])
                coeff = (target_s * 1e9) / data["totals"]["bytes"]
                remain = coeff - (hit / onchip)
                if one_minus > 0 and remain > 0:
                    required["offchip_bw_gbs"] = one_minus / remain
    else:
        target_s = max(memory_s, network_s)
        if target_s > 0:
            required["peak_tflops"] = data["totals"]["flops"] / target_s / 1e12

    recommendations: List[Dict] = []
    if include_recommendations:
        tp_candidates = [1, 2, 4, 8]
        pp_candidates = [1, 2, 4]
        for tp_c in tp_candidates:
            for pp_c in pp_candidates:
                asics = tp_c * pp_c
                if asics > max_asics:
                    continue
                trial = {
                    **payload,
                    "parallel": {**parallel, "tp": tp_c, "pp": pp_c, "max_asics": max_asics},
                }
                r = compute_inference_sizing(trial, include_recommendations=False) if (tp_c, pp_c) != (tp, pp) else None
                if r is None:
                    latency_ms = end_to_end_s * 1e3
                    b = bottleneck
                else:
                    latency_ms = r["time"]["end_to_end_ms"]
                    b = r["bottleneck"]
                note = f"{b}-bound"
                recommendations.append(
                    {
                        "tp": tp_c,
                        "pp": pp_c,
                        "asics": asics,
                        "latency_ms": latency_ms,
                        "bottleneck": b,
                        "note": note,
                    }
                )

        recommendations.sort(key=lambda x: (x["latency_ms"], x["asics"]))
        recommendations = recommendations[:3]

    return {
        "totals": data["totals"],
        "collective": {
            "tp_allreduce_count": tp_count,
            "tp_allreduce_bytes": tp_bytes,
            "pp_send_count": pp_count,
            "pp_send_bytes": pp_bytes,
        },
        "time": {
            "compute_ms": compute_s * 1e3,
            "memory_ms": memory_s * 1e3,
            "network_ms": network_s * 1e3,
            "kernel_ms": kernel_s * 1e3,
            "end_to_end_ms": end_to_end_s * 1e3,
        },
        "bottleneck": bottleneck,
        "layer_io": layer_io,
        "recommendations": recommendations,
        "required_to_debottleneck": required,
    }


def sweep_inference_sizing(payload: Dict) -> Dict:
    req = dict(payload)
    tp_candidates = req.pop("tp_candidates", [1, 2, 4, 8])
    pp_candidates = req.pop("pp_candidates", [1, 2, 4])
    max_asics = _pos_int(req.get("parallel", {}).get("max_asics", 16), 16)

    base = compute_inference_sizing(req)
    candidates = []
    for tp in tp_candidates:
        for pp in pp_candidates:
            tp_i = _pos_int(tp, 1)
            pp_i = _pos_int(pp, 1)
            if tp_i * pp_i > max_asics:
                continue
            trial = {
                **req,
                "parallel": {**req.get("parallel", {}), "tp": tp_i, "pp": pp_i, "max_asics": max_asics},
            }
            r = compute_inference_sizing(trial, include_recommendations=False)
            candidates.append(
                {
                    "tp": tp_i,
                    "pp": pp_i,
                    "asics": tp_i * pp_i,
                    "latency_ms": r["time"]["end_to_end_ms"],
                    "bottleneck": r["bottleneck"],
                    "note": f"{r['bottleneck']}-bound",
                }
            )

    candidates.sort(key=lambda x: (x["latency_ms"], x["asics"]))
    return {"base": base, "candidates": candidates[:10]}
