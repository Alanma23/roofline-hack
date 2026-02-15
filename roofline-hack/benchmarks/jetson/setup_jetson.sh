#!/bin/bash
# Quick setup script for Jetson Orin Nano validation

echo "🤖 Jetson Orin Nano Roofline Validation Setup"
echo "=============================================="
echo ""

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "⚠️  WARNING: Not running on Jetson device"
    echo "This script is optimized for Jetson Orin Nano"
    echo ""
fi

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo "❌ CUDA not found. Install JetPack SDK first:"
    echo "   https://developer.nvidia.com/embedded/jetpack"
    exit 1
fi

echo "✅ CUDA detected:"
nvcc --version | grep release
echo ""

# Check current power mode
echo "📊 Current Power Mode:"
sudo nvpmodel -q | grep "NV Power Mode"
echo ""

# Python deps
echo "📦 Installing Python dependencies..."
pip3 install torch pandas matplotlib --user

# Verify PyTorch + CUDA
echo ""
echo "🔍 Verifying PyTorch CUDA support..."
python3 << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
EOF

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Set maximum performance mode:"
echo "   sudo nvpmodel -m 0"
echo "   sudo jetson_clocks"
echo ""
echo "2. Run validation:"
echo "   python3 benchmarks/jetson/validate_jetson.py"
echo ""
echo "3. Compare power modes:"
echo "   sudo nvpmodel -m 1  # 7W mode"
echo "   python3 benchmarks/jetson/validate_jetson.py"
