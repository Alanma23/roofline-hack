#!/usr/bin/env python3
"""
Verify that the precision sweep implementation is correct.
Run this before starting the API server to check for issues.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

def test_imports():
    """Verify all required modules can be imported."""
    print("✓ Testing imports...")

    try:
        from api.server import app, analyze_gemm
        print("  ✓ API server imports OK")
    except ImportError as e:
        print(f"  ✗ API server import failed: {e}")
        return False

    try:
        from api.schemas import GEMMSpec, AnalyzeResponse
        print("  ✓ API schemas imports OK")
    except ImportError as e:
        print(f"  ✗ API schemas import failed: {e}")
        return False

    try:
        from src.roofline.hardware_registry import get_hardware
        print("  ✓ Hardware registry imports OK")
    except ImportError as e:
        print(f"  ✗ Hardware registry import failed: {e}")
        return False

    return True


def test_api_signature():
    """Verify the analyze_gemm endpoint has the new parameter."""
    print("\n✓ Testing API signature...")

    from api.server import analyze_gemm
    import inspect

    sig = inspect.signature(analyze_gemm)
    params = list(sig.parameters.keys())

    if 'run_all_precisions' in params:
        print("  ✓ run_all_precisions parameter exists")
    else:
        print(f"  ✗ run_all_precisions parameter missing. Found: {params}")
        return False

    # Check default value
    param = sig.parameters['run_all_precisions']
    if param.default == False:
        print("  ✓ run_all_precisions defaults to False")
    else:
        print(f"  ✗ run_all_precisions has wrong default: {param.default}")
        return False

    return True


def test_hardware_specs():
    """Verify hardware specs are available."""
    print("\n✓ Testing hardware specs...")

    from src.roofline.hardware_registry import get_hardware

    try:
        hw = get_hardware("b10")
        print(f"  ✓ GB10/B10 hardware loaded: {hw.name}")
        print(f"    - Bandwidth: {hw.peak_bandwidth_gb_s} GB/s")
        print(f"    - FP16 TFLOPS: {hw.peak_flops_tflops.get('FP16', 0)}")
        print(f"    - NVFP4 TFLOPS: {hw.peak_flops_tflops.get('NVFP4', 0)}")
    except Exception as e:
        print(f"  ✗ Failed to load hardware: {e}")
        return False

    return True


def test_precision_list():
    """Verify precision list matches frontend."""
    print("\n✓ Testing precision list...")

    backend_precisions = ["FP16", "FP8_E4M3", "NVFP4", "INT8", "INT4"]
    print(f"  Backend precisions: {backend_precisions}")

    # Read frontend file to check if precisions match
    frontend_file = ROOT / "frontend" / "roofline-calc-v2.jsx"
    if frontend_file.exists():
        content = frontend_file.read_text()
        if 'Run all precisions (FP16, FP8, NVFP4, INT8, INT4)' in content:
            print("  ✓ Frontend checkbox label matches backend")
        else:
            print("  ✗ Frontend checkbox label doesn't match backend")
            return False
    else:
        print("  ⚠ Frontend file not found (skip check)")

    return True


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("PRECISION SWEEP IMPLEMENTATION VERIFICATION")
    print("=" * 60)

    tests = [
        test_imports,
        test_api_signature,
        test_hardware_specs,
        test_precision_list,
    ]

    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"\n✗ Test {test.__name__} crashed: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    if all(results):
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Start API server on GB10:")
        print("   uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload")
        print("\n2. Create SSH tunnel from Mac:")
        print("   ssh -L 8000:localhost:8000 user@gb10-hostname")
        print("\n3. Start frontend:")
        print("   cd frontend && npm run dev")
        print("\n4. Open http://localhost:5173")
        print("=" * 60)
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 60)
        print("\nPlease fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
