# Roofline Performance Toolkit

Predict and optimize transformer inference performance using the roofline model. Automatic quantization recommendations for NVIDIA GB10 Grace Blackwell and other GPUs.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Predict performance (no GPU needed)
python src/roofline/calculator_shell.py

# Get quantization recommendation
python -c "from src.roofline.quantization_integration import IntegratedQuantizationEngine; \
from src.roofline.hardware_registry import BLACKWELL_B10; \
engine = IntegratedQuantizationEngine(BLACKWELL_B10); \
rec = engine.recommend({'H': 4096, 'L': 32, 'nh': 32, 'nkv': 8, 'dh': 128, 'dff': 14336}, phase='decode', seq_len=4096); \
print(f'Strategy: {rec.config.strategy_name}, Speedup: {rec.predicted_speedup:.2f}x')"

# Run tests
python run_tests.py
```

## Why This Matters

Transformer inference is **memory-bound**: the bottleneck isn't compute, it's moving data from memory.

- GB10 has 124 TFLOPS (FP8) but only 287 GB/s bandwidth
- Decode (GEMV) has AI ~2 FLOP/byte vs critical AI of 216 → **memory-bound**
- **Solution**: Lower precision = less data = proportional speedup

**Speedup = bytes reduction ratio** (for memory-bound ops)

## Core Concept: Roofline Model

```python
time = max(Bytes/Bandwidth, FLOPs/PeakFLOPs)

AI = FLOPs / Bytes  # Arithmetic Intensity

if AI < Critical_AI: "memory-bound"   # Most transformer ops
if AI > Critical_AI: "compute-bound"  # Large batch prefill
```

## Hardware Support

| Hardware | Bandwidth | FP16 | FP8 | NVFP4 | INT4 |
|----------|-----------|------|-----|-------|------|
| **GB10** (primary) | 287 GB/s | 62 | 124 | **1000** | 248 TFLOPS |
| B200 | 8000 GB/s | 180 | 4500 | 9000 | 9000 TFLOPS |
| H100 | 3350 GB/s | 134 | 1979 | — | 3958 TFLOPS |
| A100 | 2039 GB/s | 312 | — | — | 624 TFLOPS |

## Precision Formats

| Format | Bytes | GB10 | Speedup | Quality Loss | Use Case |
|--------|-------|------|---------|--------------|----------|
| FP16 | 2.0 | ✓ | 1.0× | None | Baseline |
| FP8 E4M3 | 1.0 | ✓✓ | 2.0× | <1% | Balanced |
| INT4 | 0.5 | ✓ | 4.0× | 2-5% | Max speed |
| **NVFP4** | 0.5 | **✓✓** | **4.0×** | **<1%** | **Best (1000 TFLOPS!)** |

## Usage

### 1. Basic Roofline Prediction

```python
from src.roofline import RooflineCalculator, get_hardware

calc = RooflineCalculator(get_hardware("b10"))
result = calc.predict_gemv(N=4096, K=4096, precision="FP16")

print(f"Time: {result['predicted_time_us']:.1f} µs")
print(f"Bottleneck: {result['bottleneck']}")  # memory/compute
print(f"AI: {result['ai']:.2f} FLOP/byte")
```

### 2. Advanced Quantization (Recommended)

```python
from src.roofline.quantization_integration import IntegratedQuantizationEngine
from src.roofline.hardware_registry import BLACKWELL_B10

engine = IntegratedQuantizationEngine(BLACKWELL_B10)

# Llama-3 8B decode @ 4K context
rec = engine.recommend(
    model_config={"H": 4096, "L": 32, "nh": 32, "nkv": 8, "dh": 128, "dff": 14336},
    phase="decode",
    seq_len=4096
)

# Output: Strategy: nvfp4_blackwell, FlashAttention: FA3, Speedup: 2.59x
```

**Features:**
- **Mixed Precision**: Per-operator precision (17 operator types, 6 strategies)
- **Dynamic Quantization**: Runtime adaptation with outlier detection
- **FlashAttention**: FA1/2/3/4 support with FP8 and sparsity
- **Context-Aware**: Automatic selection for decode/prefill/long-context

### 3. Benchmark on GPU

```python
from benchmarks.kernel_shell import GEMMKernel

kernel = GEMMKernel(M=1, N=4096, K=4096, precision="NVFP4")
result = kernel.benchmark()

print(f"Measured: {result['measured_time_us']:.1f} µs")
print(f"Bandwidth: {result['measured_bandwidth_gb_s']:.1f} GB/s")
```

## Performance Results (GB10)

**Llama-3 8B Decode (4K context):**

| Configuration | Time | Speedup | Quality | Memory |
|---------------|------|---------|---------|--------|
| FP16 Baseline | 45.8 ms | 1.00× | — | 0% |
| FP8 Balanced | 23.7 ms | 1.93× | <1% | 50% |
| **NVFP4 Blackwell** | **17.7 ms** | **2.59×** | **<1%** | **72%** |

**Long Context (32K tokens):**

| Configuration | KV Cache | Time | Savings |
|---------------|----------|------|---------|
| FP16 | 4096 MB | 91.6 ms | 0% |
| **NVFP4 KV** | **1175 MB** | **37.4 ms** | **71%** |

**FlashAttention:**

| Version | Features | Time | Speedup |
|---------|----------|------|---------|
| FA1 | Tiled attention | 33.8 ms | 1.00× |
| FA2 | Better parallelism | 29.4 ms | 1.15× |
| **FA3** | **FP8 native** | **15.4 ms** | **2.20×** |
| FA4 | Block-sparse | 15.4 ms | 2.20× |

## Architecture

```
src/roofline/
├── calculator_shell.py          # Core roofline formulas
├── hardware_registry.py         # Hardware specs
├── mixed_precision.py           # Per-operator strategies
├── dynamic_quant.py             # Runtime adaptive quant
├── flash_attention.py           # FA1/2/3/4 models
└── quantization_integration.py  # Unified engine

benchmarks/
├── kernel_shell.py              # GEMV/GEMM benchmarking
└── transformer_bench.py         # Full model benchmarks

tests/
└── test_*.py                    # 72 unit tests

examples/
└── quantization_demo.py         # Complete demo
```

## Testing

```bash
python run_tests.py              # Run all 72 tests
python run_tests.py -v           # Verbose
python run_tests.py mixed_precision  # Specific module
```

### Single-server deployment (app + API for anyone on the site)

Serve the frontend and API from one server so anyone visiting gets both:

```bash
cd frontend && npm run build && cd ..
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Then visit `http://localhost:8000` for the app. API at `/api/*`, public (no GPU) API at `/api/public/*`, docs at `/docs`.

Endpoints:
- `POST /api/analyze` - Analyze single GEMM (theory + measurement)
- `POST /api/sweep` - Sweep across shapes and precisions
- `POST /api/recommend` - Get quantization recommendation
- `GET /api/nvml/status` - Live GPU monitoring
- `GET /api/nvml/stream` - Real-time SSE stream

All tests pass in <0.01s. See **[tests/README.md](tests/README.md)**.

## Documentation

- **[docs/QUANTIZATION.md](docs/QUANTIZATION.md)** - Advanced quantization system
- **[THEORY.md](THEORY.md)** - Roofline math and precision formats
- **[GUIDE.md](GUIDE.md)** - Implementation guide
- **[CHANGELOG.md](CHANGELOG.md)** - Version history

## Key Insights

1. **Memory-bound dominance**: Most ops have AI < 10 vs critical AI ~216 → bandwidth limited
2. **KV cache critical**: At >1.6K tokens (FP16), KV cache reading exceeds weight loading
3. **FFN is 2/3 of compute**: Quantizing FFN weights to NVFP4 = huge impact
4. **FlashAttention + FP8**: 2.2× speedup over baseline (FA1 FP16 → FA3 FP8)
5. **NVFP4 on Blackwell**: Best accuracy/speed (4× speedup, <1% loss, 1000 TFLOPS)

## References

- [Roofline Model (Williams et al.)](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf)
- [ASUS Ascent GX10 Review](https://www.servethehome.com/asus-ascent-gx10-review-a-new-nvidia-gb10-solution/)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [TorchAO](https://github.com/pytorch/ao)

## License

MIT License
