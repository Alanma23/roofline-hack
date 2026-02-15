# Run on M3 Pro (no GPU)

## Quick start

```bash
# 1. Install Python deps (no pynvml)
pip install -r requirements-mac.txt

# 2. Install frontend deps
cd frontend && npm install && cd ..

# 3. Start backend (terminal 1)
uvicorn api.server:app --reload --port 8000

# 4. Start frontend (terminal 2)
cd frontend && npm run dev
```

Then open **http://localhost:5173** in your browser.

## What works without GPU

- **Roofline plot** — model ops, hardware presets, precision configs
- **GEMM Analyzer** — simulated predictions (no measured)
- **API** — `/api/analyze`, `/api/recommend`, `/api/hardware`, etc.
- **Auto-quantize** — `python -m src.roofline.auto_quantize`

## What doesn't work

- NVML status (503)
- Measured GEMM times (simulated only)
- `compare_shell.py`, `transformer_bench.py` (need CUDA)
