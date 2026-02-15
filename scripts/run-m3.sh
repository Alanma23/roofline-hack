#!/bin/bash
# Run full stack on M3 Pro (no GPU)
# Backend: port 8000 | Frontend: port 5173

set -e
cd "$(dirname "$0")/.."

echo "=== Roofline Toolkit — M3 Pro (no GPU) ==="
echo ""

# Check backend deps
if ! python -c "import fastapi" 2>/dev/null; then
  echo "Installing Python deps..."
  pip install -r requirements-mac.txt
fi

# Check frontend deps
if [ ! -d "frontend/node_modules" ]; then
  echo "Installing frontend deps..."
  cd frontend && npm install && cd ..
fi

echo ""
echo "Starting backend (port 8000)..."
python -m uvicorn api.server:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

sleep 2
echo "Starting frontend (port 5173)..."
cd frontend && npm run dev &
FRONTEND_PID=$!

echo ""
echo "✓ Backend:  http://localhost:8000"
echo "✓ Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both."
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
