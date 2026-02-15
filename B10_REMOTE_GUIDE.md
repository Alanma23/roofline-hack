# Running Roofline Toolkit on a Remote B10 via SSH

This guide covers running the auto quantizer, benchmarks, and connecting the React frontend to real-time B10 data.

---

## 1) Running the Auto Quantizer & Code on B10 via SSH

The **B10 must run the Python code locally** — NVML and CUDA require direct GPU access. Use SSH to run everything on the B10 machine.

### Step 1: SSH into the B10

```bash
ssh user@your-b10-host
# e.g. ssh nvidia@192.168.1.100
```

### Step 2: Clone and Install (on B10)

```bash
cd ~
git clone <your-repo-url> roofline-hack
cd roofline-hack

# Create venv (recommended)
python3 -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

pip install -r requirements.txt
```

### Step 3: Run the Auto Quantizer

```bash
python -c "
from src.roofline.auto_quantize import recommend_quantization
from src.roofline.hardware_registry import get_hardware

hw = get_hardware('b10')
rec = recommend_quantization(hardware=hw, M=1, N=4096, K=4096, phase='decode')
print(f'Recommended: {rec.precision} ({rec.method})')
print(f'Reason: {rec.reason}')
print(f'Expected speedup: {rec.predicted_speedup:.2f}x vs FP16')
"
```

### Step 4: Run Benchmarks & Compare Theory vs Reality

```bash
# Kernel benchmarks (requires CUDA)
python benchmarks/kernel_shell.py

# Full theory vs measured comparison
python compare_shell.py
```

### Step 5: Run the API Server on B10

The API must run **on the B10** so it can access NVML and CUDA:

```bash
# On the B10 (bind to 0.0.0.0 so it's reachable over the network)
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

---

## 2) Getting Realtime Data from B10 & Comparing in React

### Architecture

- **API (B10):** Runs on the B10, exposes `/api/nvml/status` and `/api/nvml/stream` for real-time GPU data.
- **Frontend (your laptop):** React app that fetches from the API and overlays predicted vs measured points on the roofline plot.

### Option A: SSH Port Forwarding (simplest)

Forward the B10's API port to your laptop:

```bash
# On your laptop (in a separate terminal)
ssh -L 8000:localhost:8000 user@your-b10-host
```

Then on the B10, start the API:

```bash
uvicorn api.server:app --host 127.0.0.1 --port 8000
```

The frontend's Vite proxy already sends `/api` → `http://localhost:8000`, so no code changes needed. Start the frontend:

```bash
cd frontend && npm run dev
```

### Option B: Direct API URL (no tunnel)

If the B10 is reachable on your network (e.g. `192.168.1.100`):

1. Run the API on B10 with `--host 0.0.0.0`.
2. Set the API base in the frontend. Create `frontend/.env`:

```bash
VITE_API_BASE=http://192.168.1.100:8000
```

Then update `roofline-calc-v2.jsx` to use it:

```javascript
const API_BASE = import.meta.env.VITE_API_BASE ?? "";
```

### Realtime Data Flow

| Endpoint | Purpose |
|----------|---------|
| `GET /api/nvml/status` | One-shot GPU status (power, temp, clocks, utilization) — polled every 2s |
| `GET /api/nvml/stream` | SSE stream for continuous updates (500ms interval) |
| `POST /api/analyze` | Returns **simulated** (predicted) + **measured** (real) roofline points |

### How Predicted vs Measured Compare in the React App

1. **Predicted rooflines:** Drawn from hardware specs (bandwidth, peak TFLOPS). The diagonal and ceilings come from `HW_PRESETS` / `hw.flops`.
2. **Simulated points:** From `POST /api/analyze` — roofline theory predictions.
3. **Measured points:** From `POST /api/analyze` when a GPU is present — actual benchmark results.

In the **Kernel Spot Check** panel:
- Enter M, N, K, precision.
- Click **Analyze GEMM**.
- The API runs a real benchmark on the B10 and returns both simulated and measured.
- Measured points are overlaid on the roofline plot as **green hollow circles** with an "M" label.
- Predicted vs measured times are shown side by side in the Analysis Result panel.

### Enabling NVML Realtime Stream (optional)

The frontend currently polls `/api/nvml/status` every 2 seconds. To use the SSE stream instead for lower latency, you'd add an `EventSource` to `GET /api/nvml/stream` in the React code.

---

## Quick Reference

| Task | Command |
|------|---------|
| SSH to B10 | `ssh user@b10-host` |
| Install deps | `pip install -r requirements.txt` |
| Auto quantizer | `python -c "from src.roofline.auto_quantize import recommend_quantization; print(recommend_quantization(M=1,N=4096,K=4096))"` |
| Run API on B10 | `uvicorn api.server:app --host 0.0.0.0 --port 8000` |
| Port forward (laptop) | `ssh -L 8000:localhost:8000 user@b10-host` |
| Start frontend | `cd frontend && npm run dev` |
