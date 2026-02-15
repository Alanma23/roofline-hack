# Unit Tests

Comprehensive unit tests for the roofline quantization system.

## Running Tests

```bash
# Run all tests
python -m unittest discover -s tests -p "test_*.py"

# Run with verbose output
python -m unittest discover -s tests -p "test_*.py" -v

# Run specific test file
python -m unittest tests.test_mixed_precision -v

# Run specific test class
python -m unittest tests.test_mixed_precision.TestMixedPrecisionConfig -v

# Run specific test
python -m unittest tests.test_mixed_precision.TestMixedPrecisionConfig.test_get_precision_default -v
```

## Test Coverage

### test_mixed_precision.py (18 tests)
- **TestMixedPrecisionConfig**: Configuration and precision retrieval
- **TestPresetStrategies**: All preset strategies (FP16, FP8, NVFP4, etc.)
- **TestStrategyRegistry**: Strategy registry operations
- **TestOperatorType**: Operator type definitions

### test_dynamic_quant.py (17 tests)
- **TestActivationStats**: Activation statistics and outlier detection
- **TestDynamicQuantConfig**: Configuration options
- **TestDynamicQuantizer**: Scale computation and fallback behavior
- **TestAdaptivePrecisionSelector**: Context-aware precision selection
- **TestAdaptiveQuantConfig**: Adaptive configuration

### test_flash_attention.py (19 tests)
- **TestFlashAttentionVersion**: FA version definitions
- **TestFlashAttentionConfig**: Configuration for FA1/2/3/4
- **TestFlashAttentionRoofline**: Performance predictions
- **TestRecommendFAPrecision**: Precision recommendations
- **TestFlashAttentionMetrics**: Metrics validation

### test_quantization_integration.py (18 tests)
- **TestIntegratedQuantConfig**: Integrated configuration
- **TestIntegratedQuantizationEngine**: Complete recommendation engine
- **TestQuantizationRecommendation**: Recommendation structure
- **TestDynamicQuantizationIntegration**: Dynamic quant integration
- **TestAdaptivePrecisionIntegration**: Adaptive precision integration

**Total: 72 tests**

## What's Tested

### Mixed Precision
✓ Operator-level precision configuration
✓ Layer-wise quantization overrides
✓ All preset strategies (FP16, FP8, W4A16, NVFP4, etc.)
✓ KV cache precision handling
✓ Strategy registry operations

### Dynamic Quantization
✓ Activation statistics computation
✓ Outlier detection and ratio calculation
✓ Scale computation (absmax, percentile, MSE-optimal)
✓ Automatic FP16 fallback
✓ Scale smoothing across batches
✓ Adaptive precision selection
✓ Context-aware recommendations

### FlashAttention
✓ FA1/2/3/4 version support
✓ FP8 native support (FA3)
✓ Block-sparse attention (FA4)
✓ Performance predictions
✓ Memory vs compute bottleneck detection
✓ Precision recommendations for different hardware
✓ Prefill vs decode optimizations

### Integration
✓ Complete recommendation pipeline
✓ Context-aware strategy selection
✓ Decode, prefill, and long context scenarios
✓ Hardware-specific optimizations
✓ Justification generation
✓ Confidence levels
✓ Bottleneck analysis
✓ Memory savings calculation

## Test Principles

1. **Unit tests only**: Each test focuses on a single function or method
2. **No GPU required**: Tests use theoretical calculations only
3. **Fast execution**: All 72 tests run in < 0.01 seconds
4. **Clear assertions**: Each test has explicit expected values
5. **Good coverage**: Tests cover normal cases, edge cases, and error conditions

## Adding New Tests

When adding new features, add corresponding tests:

```python
import unittest
from src.roofline.your_module import YourClass

class TestYourClass(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.obj = YourClass()

    def test_your_feature(self):
        """Test your feature."""
        result = self.obj.your_method()
        self.assertEqual(result, expected_value)

if __name__ == "__main__":
    unittest.main()
```

## Continuous Integration

To run tests automatically:

```bash
# Add to .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python -m unittest discover -s tests -p "test_*.py"
```
