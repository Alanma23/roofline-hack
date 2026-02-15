"""
End-to-end pipeline: load model → roofline analysis → auto quantize → benchmark.

Usage:
    from quantization.pipeline import run_pipeline
    run_pipeline(model_name="TinyLlama-1.1B-Chat-v1.0", memory_limit_gb=8)
"""

import sys
from pathlib import Path

# Add project root for imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Optional, Dict, Any
import time


def run_pipeline(
    model_name: str = "TinyLlama-1.1B-Chat-v1.0",
    memory_limit_gb: Optional[float] = 8.0,
    phase: str = "decode",
    num_tokens: int = 10,
) -> Dict[str, Any]:
    """
    Load model, run roofline analysis, auto-select quantization, apply, benchmark.

    Returns:
        dict with recommendation, applied config, tokens_per_sec, etc.
    """
    from src.roofline.auto_quantize import recommend_quantization
    from quantization.torchao_configs import apply_quantization

    result = {"recommendation": None, "quantized": False, "tokens_per_sec": None, "error": None}

    # 1. Roofline recommendation
    rec = recommend_quantization(
        memory_limit_gb=memory_limit_gb,
        phase=phase,
    )
    result["recommendation"] = {
        "precision": rec.precision,
        "method": rec.method,
        "reason": rec.reason,
        "predicted_speedup": rec.predicted_speedup,
    }

    # 2. Load model (optional - requires transformers)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        result["error"] = "transformers not installed"
        return result

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        result["error"] = str(e)
        return result

    # 3. Apply quantization
    applied = apply_quantization(model, precision=rec.precision)
    result["quantized"] = applied

    # 4. Benchmark (simple forward pass timing)
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        prompt = "Hello, how are you?"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Warmup
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=1)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Time generation
        start = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=num_tokens, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        result["tokens_per_sec"] = num_tokens / elapsed if elapsed > 0 else 0
    except Exception as e:
        result["error"] = result.get("error") or str(e)

    return result


if __name__ == "__main__":
    r = run_pipeline(model_name="TinyLlama-1.1B-Chat-v1.0", memory_limit_gb=8)
    print("Recommendation:", r.get("recommendation"))
    print("Quantized:", r.get("quantized"))
    print("Tokens/sec:", r.get("tokens_per_sec"))
    if r.get("error"):
        print("Error:", r["error"])
