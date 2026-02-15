#!/usr/bin/env python
"""
Test runner for roofline quantization system.

Usage:
    python run_tests.py                  # Run all tests
    python run_tests.py -v               # Verbose output
    python run_tests.py mixed_precision  # Run specific test file
"""

import sys
import unittest

def main():
    """Run tests with optional filtering."""
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    verbosity = 2 if verbose else 1

    # Filter arguments
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

    if args:
        # Run specific test file
        test_module = f"tests.test_{args[0]}"
        loader = unittest.TestLoader()
        try:
            suite = loader.loadTestsFromName(test_module)
        except Exception as e:
            print(f"Error loading test module '{test_module}': {e}")
            print("\nAvailable test modules:")
            print("  - mixed_precision")
            print("  - dynamic_quant")
            print("  - flash_attention")
            print("  - quantization_integration")
            return 1
    else:
        # Run all tests
        loader = unittest.TestLoader()
        suite = loader.discover("tests", pattern="test_*.py")

    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
