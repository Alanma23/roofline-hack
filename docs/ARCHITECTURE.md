# System Architecture

Comprehensive overview of the roofline calculator's design and components.

## High-Level Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (React)                      │
│  ┌───────────────────────────────────────────────────┐  │
│  │  UI Components                                     │  │
│  │  - Roofline Plot (D3.js)                          │  │
│  │  - MoE Configuration Panel                        │  │
│  │  - Run Tracking Workspace                         │  │
│  │  - Comparison Tables                              │  │
│  └───────────────────────────────────────────────────┘  │
│                         │                                │
│                         │ HTTP/REST                      │
│                         ▼                                │
├─────────────────────────────────────────────────────────┤
│                   Backend (FastAPI)                      │
│  ┌───────────────────────────────────────────────────┐  │
│  │  API Server (14 endpoints)                        │  │
│  │  - Basic Roofline                                 │  │
│  │  - MoE Analysis                                   │  │
│  │  - Advanced Roofline                              │  │
│  │  - Run Tracking                                   │  │
│  └───────────────────────────────────────────────────┘  │
│                         │                                │
│                         │                                │
│                         ▼                                │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Core Roofline Models                             │  │
│  │  ┌─────────────────┬─────────────────┬──────────┐ │  │
│  │  │ Basic Roofline  │ MoE Model       │ Advanced │ │  │
│  │  │ + Efficiency    │ + Expert Routing│ Roofline │ │  │
│  │  └─────────────────┴─────────────────┴──────────┘ │  │
│  └───────────────────────────────────────────────────┘  │
│                         │                                │
│                         │                                │
│                         ▼                                │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Hardware Registry                                │  │
│  │  - GB10, B200, H100, A100                        │  │
│  │  - Efficiency Factors                             │  │
│  │  - Memory Hierarchy                               │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Backend Components

### 1. Core Roofline Calculators

#### a) Basic Roofline (`calculator_shell.py`)
- **Purpose**: Classic roofline model with empirical efficiency factors
- **Key Classes**:
  - `HardwareSpec`: Hardware specification (bandwidth, FLOPS, efficiency)
  - `RooflineCalculator`: Predicts GEMV/GEMM/Attention performance
- **Efficiency Factors**: Per-kernel, per-precision corrections (e.g., FP16 GEMM: 0.155)
- **Formula**: 
  ```
  Time = max(FLOPs / (Peak_FLOPS * Efficiency), Bytes / Bandwidth)
  ```

#### b) MoE Model (`moe_model.py`)
- **Purpose**: Model Mixture of Experts architectures
- **Components**:
  1. **Router**: MLP + top-K sorting overhead
  2. **Expert Computation**: Parallelized SwiGLU FFN
  3. **All-to-All Communication**: Expert parallelism (EP) scatter/gather
  4. **Load Balancing**: Imbalance penalties and utilization
- **Key Metrics**: Sparsity, expert utilization, communication overhead
- **Bottleneck Detection**: Compute / Communication / Load_Imbalance

#### c) Advanced Roofline (`advanced_roofline.py`)
- **Purpose**: 3-level hardware realism for inference workloads
- **Level 1 - Memory Hierarchy**:
  - SRAM (on-chip, ultra-fast) vs DRAM (off-chip)
  - Automatic tier detection based on working set size
- **Level 2 - Array Utilization**:
  - Padding penalty when workload ≠ array dimensions
  - Example: M=10 → padded to 128 = 7.8% utilization
- **Level 3 - Mixed Precision**:
  - Separate precision for inputs vs accumulation
  - Asymmetric read/write bandwidth
- **Key Insight**: Captures 33× error in naive roofline for single-token inference

### 2. Hardware Registry (`hardware_registry.py`)
- **Presets**: GB10, B200, H100, A100
- **Efficiency Factors**: Empirical corrections per hardware/kernel/precision
- **Custom Hardware**: Support for registering ASICs

### 3. Supporting Modules
- `roofline_math.py`: FLOP/byte calculations, arithmetic intensity
- `inference_sizing.py`: Multi-chip TP/PP sizing
- `tiling_model.py`: Tile size optimization
- `auto_quantize.py`: Quantization recommendations

## Frontend Architecture

### Technology Stack
- **Framework**: React 18
- **Visualization**: D3.js for roofline plots
- **State Management**: React useState/useEffect
- **Persistence**: localStorage for run tracking
- **Build Tool**: Vite

### Key Components

#### 1. Main App (`roofline-calc-v2.jsx`)
- **State Management**: Hardware, workload, precision, saved runs
- **Panels**:
  - Hardware configuration
  - Workload selection (dense + MoE models)
  - Precision configurations
  - MoE configuration (when MoE model selected)
  - Saved runs with comparison
- **Roofline Plot**: Log-log plot with multiple roofline ceilings

#### 2. Run Storage (`RunStorage`)
- **Persistence**: localStorage API wrapper
- **Operations**: save, list, load, delete, export, import
- **Format**: JSON with metadata (timestamp, tags, config, results)

#### 3. Comparison Workspace
- **Features**:
  - Checkbox selection of runs
  - Side-by-side comparison table
  - Delta percentages (green = better, red = worse)
  - Export/import JSON

## Data Flow

### Analysis Request Flow

```
User Input (UI)
    │
    ▼
Frontend State Update
    │
    ▼
API Request (POST /api/analyze)
    │
    ▼
FastAPI Server
    │
    ├─→ Hardware Registry (get specs)
    │
    ├─→ Roofline Calculator
    │   ├─→ Apply efficiency factors
    │   ├─→ Calculate AI, roofline time
    │   └─→ Determine bottleneck
    │
    └─→ Response (JSON)
        │
        ▼
Frontend Update (state + plot)
```

### MoE Analysis Flow

```
Select MoE Model (e.g., DeepSeek-V3)
    │
    ▼
MoE Config Panel Appears
    │
    ▼
Click "🔬 Analyze MoE Performance"
    │
    ▼
API Request (POST /api/moe/analyze)
    │
    ▼
MoECalculator
    ├─→ Router overhead
    ├─→ Expert computation (parallelized)
    ├─→ All-to-all communication
    ├─→ Load imbalance penalty
    └─→ Bottleneck identification
        │
        ▼
Response with:
    - Time breakdown
    - Expert utilization heatmap
    - Recommendations
```

## API Design

### Endpoint Organization

**Basic Roofline**:
- `/api/analyze` - Single GEMM analysis
- `/api/sweep` - Shape/precision sweep

**Run Tracking**:
- `/api/runs/save` - Save configuration
- `/api/runs/list` - List saved runs
- `/api/runs/{run_id}` - Load/delete specific run
- `/api/runs/compare` - Compare multiple runs
- `/api/runs/export` - Export as JSON
- `/api/runs/import` - Import from JSON

**MoE**:
- `/api/moe/analyze` - MoE layer analysis

**Advanced**:
- `/api/advanced/analyze` - Single layer with 3-level realism
- `/api/advanced/batch-sweep` - Batch size sweep

**Utilities**:
- `/api/hardware` - Hardware management
- `/api/recommend` - Quantization recommendations
- `/api/nvml/status` - Live GPU status

### Schema Design (Pydantic)

**Key Models**:
- `GEMMSpec`: Matrix multiply specification
- `RunMetadata`: Saved run with full snapshot
- `MoEWorkloadSpec`: MoE layer configuration
- `AdvancedLayerSpec`: Layer with precision bits
- `AdvancedRooflineResult`: Results with utilization/memory tier

## Database & Storage

### Current Implementation
- **Run Tracking**: In-memory dict (`_run_storage`)
- **Frontend Persistence**: localStorage (JSON)

### Migration Path (Future)
```python
# Easy migration to PostgreSQL/SQLite
class RunStorage:
    def __init__(self, db_conn):
        self.db = db_conn
    
    def save(self, run: RunMetadata):
        self.db.execute(
            "INSERT INTO runs (...) VALUES (...)",
            run.dict()
        )
```

## Performance Considerations

### Backend Optimizations
1. **Caching**: Hardware specs loaded once
2. **Efficiency Factors**: Pre-computed lookup tables
3. **Batch Operations**: Vectorized numpy operations
4. **API Response Size**: Limit roofline points in response

### Frontend Optimizations
1. **Lazy Loading**: D3 plots rendered on demand
2. **Debouncing**: Slider updates throttled
3. **localStorage**: Async save/load
4. **Memoization**: `useMemo` for expensive computations

## Security Considerations

### Current Status
- **No Authentication**: Public deployment mode
- **No Rate Limiting**: Add for production
- **Input Validation**: Pydantic schemas
- **No SQL Injection**: No database (in-memory storage)

### Production Hardening Checklist
- [ ] Add authentication (OAuth, API keys)
- [ ] Add rate limiting (per-IP, per-user)
- [ ] Add CORS restrictions
- [ ] Add request size limits
- [ ] Add logging and monitoring
- [ ] Add HTTPS/TLS
- [ ] Add input sanitization for custom hardware

## Extensibility

### Adding New Hardware

```python
# hardware_registry.py
NEW_CHIP = HardwareSpec(
    name="New Chip",
    peak_bandwidth_gb_s=1000.0,
    peak_flops_tflops={
        "FP16": 200.0,
        "INT8": 400.0,
    },
    kernel_efficiency={
        "gemv": {"FP16": 0.6, "INT8": 0.5},
        "gemm": {"FP16": 0.8, "INT8": 0.7},
    }
)

HARDWARE_REGISTRY["new_chip"] = NEW_CHIP
```

### Adding New Precision Format

```python
# calculator_shell.py
def bytes_per_element(precision: str) -> float:
    formats = {
        # Existing formats...
        "NEW_FORMAT": 0.5,  # 4 bits
    }
    return formats.get(precision, 2.0)

# hardware_registry.py
hw.peak_flops_tflops["NEW_FORMAT"] = 500.0
```

### Adding New Analysis Mode

```python
# New file: src/roofline/custom_analysis.py
class CustomAnalyzer:
    def __init__(self, hw: HardwareSpec):
        self.hw = hw
    
    def analyze(self, workload):
        # Custom logic
        return results

# server.py
@app.post("/api/custom/analyze")
def custom_analyze(spec):
    analyzer = CustomAnalyzer(hw)
    return analyzer.analyze(spec)
```

## Testing Strategy

### Backend Tests
```bash
# Unit tests
pytest src/roofline/test_*.py

# Integration tests
pytest api/test_endpoints.py

# Smoke tests
python -m src.roofline.calculator_shell
python -m src.roofline.moe_model
python -m src.roofline.advanced_roofline
```

### Frontend Tests
```bash
cd frontend
npm run test        # Vitest unit tests
npm run test:e2e    # Playwright E2E tests
```

### API Tests
```bash
# Start server
uvicorn api.server:app --reload

# Run curl tests
bash scripts/test_api.sh
```

## Deployment Options

### Option 1: Development (Hot Reload)
```bash
# Terminal 1
uvicorn api.server:app --reload --port 8000

# Terminal 2
cd frontend && npm run dev
```

### Option 2: Single-Server Production
```bash
cd frontend && npm run build && cd ..
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### Option 3: Docker (Future)
```dockerfile
FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN cd frontend && npm install && npm run build
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Option 4: Kubernetes (Future)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: roofline-calculator
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: roofline-calculator:latest
        ports:
        - containerPort: 8000
```

## Monitoring & Observability

### Logging (Future Enhancement)
```python
import logging

logger = logging.getLogger("roofline")
logger.info(f"Analyzed {M}x{N}x{K} @ {precision}")
logger.warning(f"Low utilization: {util*100:.1f}%")
```

### Metrics (Future Enhancement)
```python
from prometheus_client import Counter, Histogram

api_requests = Counter("api_requests_total", "Total API requests")
analysis_duration = Histogram("analysis_duration_seconds", "Analysis time")

@api_requests.count_exceptions()
@analysis_duration.time()
def analyze(...):
    ...
```

## Future Enhancements

### Planned Features
1. **Sparsity Support**: Skip zero multiplications in advanced roofline
2. **Multi-Chip Scaling**: Model all-reduce for data parallelism
3. **Dynamic Precision**: Per-layer precision schedules
4. **Kernel Fusion**: Model fused operations (GEMM → ReLU → GEMM)
5. **Power Modeling**: Watts per operation
6. **Cost Modeling**: $/TFLOP-hour for cloud deployment

### Architecture Changes
1. **Database Migration**: PostgreSQL for run tracking
2. **Caching Layer**: Redis for frequently accessed hardware specs
3. **Message Queue**: Celery for long-running batch sweeps
4. **WebSocket**: Real-time updates for streaming benchmarks

---

**Last Updated**: 2024-02-23

**Architecture Version**: 2.0 (includes MoE + Advanced Roofline)
