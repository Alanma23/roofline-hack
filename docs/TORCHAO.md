# TorchAO Deep Dive

**TorchAO** = PyTorch Architecture Optimization (https://github.com/pytorch/ao). It is **not** "torchAL" — the project uses **torchao** for quantization.

## What TorchAO Does Here

TorchAO provides native PyTorch quantization primitives. This project uses it to apply **weight-only quantization** (PTQ) to transformer models:

| Roofline Recommendation | TorchAO Config | Notes |
|-------------------------|----------------|-------|
| INT4, NVFP4, NF4, MXFP4 | `int4_weight_only(group_size=128)` | Grouped quantization |
| INT8, FP8_E4M3, FP8_E5M2 | `int8_weight_only()` | FP8 maps to INT8 fallback |
| FP16, FP32 | — | No quantization |

## Integration Flow

```
recommend_quantization()  →  precision (e.g. NVFP4)
         ↓
map_precision_to_torchao()  →  INT4 or INT8
         ↓
get_quantize_config(INT4, group_size=128)  →  config object
         ↓
quantize_(model, config)  →  model modified in-place
```

## API Usage

```python
from quantization.torchao_configs import get_quantize_config, apply_quantization, map_precision_to_torchao

# Map roofline output to torchao-supported precision
mapped = map_precision_to_torchao("NVFP4")  # → "INT4"

# Apply to model
success = apply_quantization(model, precision="INT4", group_size=128)
```

## Lazy Import

TorchAO is optional. If not installed, `apply_quantization` returns `False` and the pipeline continues without quantization. This allows running roofline analysis and benchmarks on systems without torchao.

## TorchAO Primitives Used

- `quantize_(model, config)` — in-place quantization of Linear layers
- `int4_weight_only(group_size=128)` — 4-bit grouped quantization
- `int8_weight_only()` — 8-bit per-tensor quantization

## Limitations

- **FP8 weight-only**: TorchAO does not expose FP8 weight quantization; FP8 recommendations map to INT8.
- **AWQ**: Requires calibration data; this project uses PTQ as default.
- **KV cache quantization**: Not handled by torchao weight-only; separate mechanism needed.
