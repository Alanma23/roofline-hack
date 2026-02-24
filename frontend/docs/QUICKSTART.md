# Quick Start Guide

Get the roofline calculator running in 5 minutes.

## 1. Install Dependencies

### Backend
```bash
pip install fastapi uvicorn pydantic numpy
```

### Frontend
```bash
cd frontend
npm install
```

## 2. Start Services

### Option A: Development Mode (Backend + Frontend Separate)

**Terminal 1 - Backend:**
```bash
uvicorn api.server:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

Access:
- Frontend UI: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Option B: Production Mode (Single Server)

```bash
# Build frontend
cd frontend && npm run build && cd ..

# Run everything from one server
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Access everything at: http://localhost:8000

## 3. Try It Out

### Web UI

1. **Select Hardware**: Choose "GB10 Blackwell" from dropdown
2. **Select Model**: Pick "Llama-3 8B" or "DeepSeek-V3 671B (MoE)"
3. **Choose Precision**: Select "FP8 E4M3" for fast inference
4. **View Results**: See roofline plot, bottleneck analysis, throughput

### Python CLI

```bash
# Basic roofline
python -m src.roofline.calculator_shell

# MoE analysis
python -m src.roofline.moe_model

# Advanced roofline
python -m src.roofline.advanced_roofline
```

### REST API

```bash
# Test with curl
curl -X POST 'http://localhost:8000/api/analyze?hardware_key=b10' \
  -H 'Content-Type: application/json' \
  -d '{
    "M": 4096,
    "N": 4096,
    "K": 4096,
    "precision": "FP16",
    "use_efficiency": true
  }' | python -m json.tool
```

Expected output:
```json
{
  "hardware": "NVIDIA GB10 Grace Blackwell (GX10)",
  "simulated": [{
    "predicted_time_us": 14339.2,
    "bottleneck": "memory",
    "efficiency_factor": 0.155,
    ...
  }],
  ...
}
```

## 4. Common Tasks

### Compare Configurations

1. Configure workload (model, precision, batch)
2. Click "💾 Save current configuration"
3. Change precision (e.g., FP16 → FP8)
4. Save again
5. Select both runs with checkboxes
6. Click "📊 Compare (2 selected)"
7. View delta percentages (green = faster)

### Analyze MoE Model

1. Select "DeepSeek-V3 671B (MoE)" from model dropdown
2. MoE configuration panel appears automatically
3. Review: 256 experts, 8 active, 96.9% sparsity
4. Click "🔬 Analyze MoE Performance"
5. View:
   - Time breakdown (router/experts/all-to-all)
   - Expert utilization heatmap
   - Bottleneck identification
   - Recommendations

### Test Advanced Roofline

```python
from src.roofline.advanced_roofline import AdvancedRooflineSim, LayerSpec, create_tpu_v3_like

hw = create_tpu_v3_like()
sim = AdvancedRooflineSim(hw)

# Single token inference (M=1) - Shows utilization penalty
layer = LayerSpec("Inference", M=1, N=4096, K=4096)
result = sim.run_analysis(layer, input_precision_bits=8, accumulator_precision_bits=32)

# Output:
# Utilization: 0.8% (padding wastes 99.2%!)
# Performance: 1.02 TOPS (vs 131 TOPS at full utilization)
# Bottleneck: UTILIZATION
```

## 5. Troubleshooting

### Port Already in Use
```bash
# Change port
uvicorn api.server:app --port 8001

# Or kill existing process
lsof -ti:8000 | xargs kill -9
```

### Module Not Found
```bash
# Ensure you're in project root
cd /path/to/roofline-hack

# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Frontend Not Loading
```bash
# Clear Vite cache
cd frontend
rm -rf node_modules/.vite
npm run dev
```

### CUDA Benchmarks Not Working
CUDA benchmarks require NVIDIA GPU. If not available:
- Simulation-only mode works fine
- Use `run_all_precisions=false` in API calls
- Frontend shows "GPU status: NVML unavailable"

## 6. Next Steps

- Read [API Reference](API_REFERENCE.md) for all endpoints
- Explore [Advanced Roofline](ADVANCED_ROOFLINE.md) for inference optimization
- Check [Architecture](ARCHITECTURE.md) to understand internals
- See [Development Guide](DEVELOPMENT.md) to extend functionality

## 7. Example Workflows

### Workflow 1: Compare Quantization Strategies
```bash
# Save baseline (FP16)
# Save W4A16 (NVFP4)
# Save W4A8 (NVFP4 + FP8)
# Compare → See memory savings vs accuracy trade-off
```

### Workflow 2: Optimize MoE Configuration
```bash
# Try EP=1 (no parallelism) → Compute-bound
# Try EP=8 (high parallelism) → Communication-bound
# Find sweet spot where bottleneck transitions
```

### Workflow 3: Batch Size Analysis
```python
# Use advanced roofline batch sweep
from src.roofline.advanced_roofline import AdvancedRooflineSim, LayerSpec, create_blackwell_b10_advanced

hw = create_blackwell_b10_advanced()
sim = AdvancedRooflineSim(hw)
layer = LayerSpec("Test", M=0, N=4096, K=4096)

results = sim.compare_batch_sizes(layer, batch_sizes=[1, 8, 16, 32, 64, 128, 256])
# Find minimum batch for >90% utilization
```

---

**You're ready to go!** Start with the web UI, then explore Python API and advanced features.
