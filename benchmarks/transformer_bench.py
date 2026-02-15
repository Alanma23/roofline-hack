"""
Full transformer inference benchmark for roofline validation.

Benchmarks TinyLlama/Phi-2 at FP16, INT8, INT4 and reports tokens/sec.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time
from typing import Optional, Dict, List


def benchmark_model(
    model_name: str = "TinyLlama-1.1B-Chat-v1.0",
    precisions: Optional[List[str]] = None,
    num_tokens: int = 20,
    warmup_tokens: int = 2,
) -> Dict[str, Dict]:
    """
    Benchmark model at different precisions.

    Args:
        model_name: HuggingFace model id
        precisions: ["FP16", "INT8", "INT4"] or None for all
        num_tokens: tokens to generate for timing
        warmup_tokens: warmup generation length

    Returns:
        {precision: {tokens_per_sec, time_s, ...}}
    """
    precisions = precisions or ["FP16", "INT8", "INT4"]
    results = {}

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        return {"error": str(e)}

    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for prec in precisions:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if prec == "FP16" else torch.float32,
                low_cpu_mem_usage=True,
            )

            if prec in ("INT8", "INT4"):
                try:
                    from quantization.torchao_configs import apply_quantization
                    apply_quantization(model, precision=prec)
                except Exception:
                    results[prec] = {"error": "torchao quantization failed"}
                    continue

            model = model.to(device)
            model.eval()

            prompt = "The meaning of life is"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Warmup
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=warmup_tokens)
            torch.cuda.synchronize()

            # Benchmark
            start = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=num_tokens, do_sample=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            n_gen = out.shape[1] - inputs["input_ids"].shape[1]
            results[prec] = {
                "tokens_per_sec": n_gen / elapsed,
                "time_s": elapsed,
                "tokens_generated": n_gen,
            }
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            results[prec] = {"error": str(e)}

    return results


def run_comparison(model_name: str = "TinyLlama-1.1B-Chat-v1.0"):
    """Run FP16 vs INT8 vs INT4 comparison and print results."""
    print("=" * 60)
    print(f"Transformer Benchmark: {model_name}")
    print("=" * 60)

    results = benchmark_model(model_name, num_tokens=20)
    if "error" in results and isinstance(results["error"], str):
        print("Error:", results["error"])
        return

    baseline_tps = None
    for prec, r in results.items():
        if "error" in r:
            print(f"\n[{prec}] Error: {r['error']}")
            continue
        tps = r["tokens_per_sec"]
        if baseline_tps is None:
            baseline_tps = tps
        speedup = tps / baseline_tps if baseline_tps else 1.0
        print(f"\n[{prec}]")
        print(f"  Tokens/sec: {tps:.1f}")
        print(f"  Time: {r['time_s']:.2f}s for {r['tokens_generated']} tokens")
        print(f"  Speedup vs baseline: {speedup:.2f}x")


if __name__ == "__main__":
    run_comparison()
