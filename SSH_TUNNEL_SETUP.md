# SSH Tunnel Setup Guide

Run benchmarks on GB10 (remote GPU) from Mac (local frontend).

## Architecture

```
Mac (localhost:5173)
  ↓ Vite proxy
Mac (localhost:8000)
  ↓ SSH tunnel
GB10 (localhost:8000)
  ↓ API server
GB10 GPU (CUDA benchmarks)
```

## One-Time Setup

### 1. Start API Server on GB10

SSH into GB10 and run:

```bash
cd ~/roofline-hack
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

Leave this running in Terminal 1.

### 2. Create SSH Tunnel from Mac

On your Mac, open a new terminal and run:

```bash
ssh -L 8000:localhost:8000 user@gb10-hostname
```

Replace `user@gb10-hostname` with your actual SSH credentials.

**Important:** Keep this terminal window open. If the connection drops, the tunnel breaks.

### 3. Test Connection

On Mac, verify the tunnel works:

```bash
curl http://localhost:8000/api/nvml/status
```

You should see GB10 GPU info in the response.

### 4. Start Frontend on Mac

```bash
cd ~/Downloads/roofline-hack/frontend
npm run dev
```

Open http://localhost:5173 in your browser.

## Usage

### Regular Workflow

1. Enter GEMM config: M=4096, N=4096, K=4096
2. **Optional:** Check "Run all precisions" checkbox
3. Click "Analyze GEMM"
4. Wait for results (~5-10s per precision)
5. Results appear as colored hollow circles on roofline plot

### Precision Colors

- **Orange** (#f97316): FP16
- **Green** (#22c55e): FP8_E4M3
- **Purple** (#a855f7): NVFP4
- **Cyan** (#06b6d4): INT8
- **Blue** (#3b82f6): INT4

### Multi-Precision Sweep

When "Run all precisions" is checked:
- Runs 5 benchmarks sequentially (not concurrent)
- Same M×N×K shape, different precisions
- All results plotted together for comparison
- ~30-60 seconds total (5 × 30 iterations each)

## Troubleshooting

### "API offline" Error

**Cause:** Tunnel or API server not running.

**Fix:**
1. Check API server on GB10 (Terminal 1)
2. Check SSH tunnel on Mac (Terminal 2)
3. Test with `curl http://localhost:8000/api/nvml/status`

### SSH Tunnel Keeps Dropping

**Solution 1:** Use `autossh` for auto-reconnect:
```bash
brew install autossh
autossh -M 0 -L 8000:localhost:8000 user@gb10-hostname
```

**Solution 2:** Add keep-alive to SSH config:
```bash
# ~/.ssh/config
Host gb10-hostname
  ServerAliveInterval 60
  ServerAliveCountMax 3
```

### GPU Contention (2 kernels running)

**Important:** The user has 2 notebooks running concurrently. Precision sweep runs sequentially to avoid GPU memory issues.

If you see CUDA OOM errors:
1. Stop other GPU processes
2. Reduce batch size in benchmarks
3. Run single precision mode instead of sweep

### Results Don't Persist

Current implementation stores results in React state (lost on reload).

To persist results across sessions (future enhancement):
- Add localStorage save/load
- Export results to CSV
- Use browser IndexedDB

## Advanced: Auto-Reconnect Script

Create `~/bin/tunnel-gb10.sh`:

```bash
#!/bin/bash
while true; do
  echo "Starting SSH tunnel to GB10..."
  ssh -L 8000:localhost:8000 user@gb10-hostname
  echo "Tunnel disconnected. Reconnecting in 5 seconds..."
  sleep 5
done
```

Make executable and run:
```bash
chmod +x ~/bin/tunnel-gb10.sh
~/bin/tunnel-gb10.sh
```

## Architecture Notes

### Why SSH Tunnel?

- Mac has no CUDA → can't run benchmarks locally
- GB10 has CUDA → runs benchmarks on real hardware
- Tunnel proxies API calls from Mac to GB10
- Frontend stays on Mac (hot reload, fast dev)

### Security

- Tunnel only exposes localhost:8000 (not public)
- API server binds to 0.0.0.0:8000 but only accepts local connections
- CORS enabled for frontend development

### Performance

- Minimal latency (LAN speeds)
- Benchmark time dominated by GPU compute, not network
- Measured points cached in React state for instant re-render
