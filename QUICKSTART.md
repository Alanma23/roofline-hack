# Quick Start: Precision Sweep

**3-minute setup to run benchmarks on GB10 from your Mac**

---

## Terminal Setup (3 terminals)

### Terminal 1: GB10 API Server
```bash
ssh user@gb10-hostname
cd ~/roofline-hack
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```
*Keep running*

### Terminal 2: Mac SSH Tunnel
```bash
ssh -L 8000:localhost:8000 user@gb10-hostname
```
*Keep running*

### Terminal 3: Mac Frontend
```bash
cd ~/Downloads/roofline-hack/frontend
npm run dev
```
*Open http://localhost:5173*

---

## Usage

### Single Precision (5 seconds)
1. Enter: M=4096, N=4096, K=4096
2. Select: FP16
3. Click: **Analyze GEMM**
4. Result: 1 orange circle on plot

### Multi-Precision Sweep (60 seconds)
1. Enter: M=4096, N=4096, K=4096
2. Check: ☑ **Run all precisions**
3. Click: **Analyze GEMM**
4. Result: 5 colored circles (orange, green, purple, cyan, blue)

---

## Color Legend

- 🟠 **Orange** = FP16
- 🟢 **Green** = FP8_E4M3
- 🟣 **Purple** = NVFP4
- 🔵 **Cyan** = INT8
- 🔷 **Blue** = INT4

---

## Troubleshooting

**"API offline" warning?**
→ Check Terminal 1 (API server) and Terminal 2 (SSH tunnel)

**Tunnel keeps dropping?**
→ Use: `autossh -M 0 -L 8000:localhost:8000 user@gb10-hostname`

**CUDA OOM error?**
→ Reduce M/N/K or stop other GPU processes

---

## What's Happening?

```
Your Mac → SSH Tunnel → GB10 API → GB10 GPU
           (8000)      (8000)     (CUDA)
```

Results appear instantly on your local browser!

---

**Full docs:** See `SSH_TUNNEL_SETUP.md` and `IMPLEMENTATION_SUMMARY.md`
