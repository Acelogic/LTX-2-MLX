# LTX-2-MLX Test Suite

This directory contains the test suite for the LTX-2-MLX project, organized into unit tests and integration tests.

## Test Structure

### Unit Tests (Fast, No Weights Required)

These tests validate individual components without requiring model weights:

- **test_scheduler.py** (22 tests) - Sigma schedulers and Euler diffusion step
  - LTX2Scheduler sigma schedule generation
  - Distilled sigma value validation
  - EulerDiffusionStep numerical stability and correctness

- **test_conditioning.py** (15 tests) - Conditioning logic for video generation
  - VideoConditionByLatentIndex (token replacement)
  - VideoConditionByKeyframeIndex (keyframe appending)
  - VideoLatentTools utilities

- **test_upscalers.py** (23 tests) - Spatial and temporal upscaling components
  - GroupNorm, Conv3d, ResBlock3d building blocks
  - SpatialPixelShuffle and TemporalPixelShuffle
  - Full SpatialUpscaler and TemporalUpscaler pipelines
  - Numerical stability verification (prevents res-block explosion)

**Total unit tests: 60**
**Execution time: ~2 seconds**

### Integration Tests (Slow, Requires Weights)

These tests validate end-to-end video generation pipelines:

- **test_video_generation.py** - Full pipeline tests with model weights
  - Text-to-video generation
  - Image-to-video generation
  - Two-stage pipeline with spatial upscaling
  - FP8 quantized model testing

## Running Tests

### With `uv` (Recommended - Fast and Isolated)

`uv` automatically creates an isolated environment and runs tests:

```bash
# Run all unit tests (fast, ~2 seconds)
uv run pytest tests/ -m unit -v

# Run specific test file
uv run pytest tests/test_scheduler.py -v

# Run all tests
uv run pytest tests/ -v
```

### With `pytest` Directly

If you have dependencies installed:

```bash
# Run all unit tests
PYTHONPATH=. pytest tests/ -m unit -v

# Or if installed in editable mode
pytest tests/ -m unit -v
```

### Run All Unit Tests (Fast)
```bash
uv run pytest tests/ -m unit -v
```

### Run Specific Test File
```bash
# Scheduler tests only
pytest tests/test_scheduler.py -v

# Conditioning tests only
pytest tests/test_conditioning.py -v

# Upscaler tests only
pytest tests/test_upscalers.py -v
```

### Run Integration Tests (Requires Weights)
```bash
pytest tests/ -m integration -v
```

### Run All Tests
```bash
pytest tests/ -v
```

### Skip Slow Tests
```bash
pytest tests/ -m "not slow" -v
```

### Skip Tests Requiring Weights
```bash
pytest tests/ -m "not requires_weights" -v
```

## Test Markers

Tests are automatically marked based on their characteristics:

- `@pytest.mark.unit` - Fast unit tests without external dependencies
- `@pytest.mark.integration` - End-to-end integration tests
- `@pytest.mark.slow` - Tests that take >10 seconds
- `@pytest.mark.requires_weights` - Tests requiring model weights

## Configuration

Test configuration is managed in:

- **conftest.py** - Pytest configuration, fixtures, and shared utilities
- **pyproject.toml** - Test settings under `[tool.pytest.ini_options]`

## Fixtures

Shared fixtures available to all tests:

- `test_logger` - Verbose test logging with timing
- `temp_output_dir` - Temporary directory for test outputs
- `weights_dir` - Path to model weights directory
- `examples_dir` - Path to example images directory

## Test Coverage

Current test coverage by component:

| Component | Tests | Status |
|-----------|-------|--------|
| Schedulers & Diffusion | 22 | ✅ Complete |
| Conditioning Logic | 15 | ✅ Complete |
| Upscalers (Spatial/Temporal) | 23 | ✅ Complete |
| VAE Encoder/Decoder | 0 | ⏳ Pending |
| Transformer Blocks | 0 | ⏳ Pending |
| Text Encoder | 0 | ⏳ Pending |
| Full Pipelines | 0 | ⏳ Pending |

## Writing New Tests

### Unit Test Template

```python
"""Unit tests for <component name>.

Tests <component> without requiring model weights.

Run with: pytest tests/test_<component>.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from LTX_2_MLX.components import YourComponent


class TestYourComponent:
    """Test YourComponent functionality."""

    def test_basic_functionality(self):
        """Test basic component operation."""
        component = YourComponent()
        input_data = mx.random.normal((1, 128, 8, 8))

        output = component(input_data)

        assert output.shape == input_data.shape
        assert np.all(np.isfinite(np.array(output)))
```

### Integration Test Template

```python
@pytest.mark.integration
@pytest.mark.requires_weights
@pytest.mark.slow
def test_full_pipeline(test_logger):
    """Test complete video generation pipeline."""
    test_logger.log("Starting full pipeline test")

    # Your test implementation

    test_logger.log("Pipeline test completed")
```

## CI/CD Integration

For CI/CD pipelines, run only fast unit tests:

```bash
# Run unit tests only (no model weights needed)
pytest tests/ -m "unit and not slow" -v --maxfail=3
```

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError: No module named 'LTX_2_MLX'`:

```bash
# Set PYTHONPATH before running tests
PYTHONPATH=/path/to/LTX-2-MLX pytest tests/ -v

# Or install the package in development mode
pip install -e .
```

### Test Collection Errors

If pytest fails to collect tests:

1. Check that conftest.py exists in tests/
2. Verify all test files start with `test_`
3. Ensure all test functions start with `test_`

### Slow Test Execution

Unit tests should complete in ~2 seconds. If slower:

1. Check you're not running integration tests by accident
2. Use `-m unit` to run only unit tests
3. Verify test isn't loading model weights

## Future Work

Planned additions to test suite:

1. **VAE Tests** - Encoder/decoder roundtrip validation
2. **Transformer Tests** - Attention block correctness
3. **Text Encoder Tests** - Gemma embedding generation
4. **Pipeline Tests** - End-to-end pipeline integration tests
5. **Performance Benchmarks** - Speed and memory usage tracking
