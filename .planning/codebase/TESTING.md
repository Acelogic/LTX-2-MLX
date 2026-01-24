# Testing Patterns

**Analysis Date:** 2026-01-23

## Test Framework

**Runner:**
- pytest 7.4.0+
- Config: `pyproject.toml` [tool.pytest.ini_options]
- Additional plugin: `pytest-timeout==2.1.0` for test timeout handling

**Assertion Library:**
- Built-in pytest assertions: `assert`, `assert_allclose`, `assert_array_equal`
- NumPy assertions for numerical testing: `np.testing.assert_allclose()`
- MLX/NumPy comparison assertions

**Run Commands:**
```bash
# Run all tests
pytest tests/

# Run all tests with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_scheduler.py -v

# Run specific test class
pytest tests/test_scheduler.py::TestLTX2Scheduler -v

# Run specific test method
pytest tests/test_scheduler.py::TestLTX2Scheduler::test_sigma_schedule_length -v

# Run with markers (unit tests only)
pytest -m unit

# Run all except slow tests
pytest -m "not slow"

# Run with timeout (requires pytest-timeout)
pytest --timeout=300

# Watch mode (requires pytest-watch, not in deps but can be installed)
pytest-watch tests/
```

## Test File Organization

**Location:**
- All tests in `tests/` directory at project root
- `tests/conftest.py` for shared configuration and fixtures
- `tests/fixtures/` directory for test data/fixtures
- `tests/outputs/` directory for test output artifacts

**Naming:**
- Test files: `test_*.py` (e.g., `test_scheduler.py`, `test_pipelines.py`)
- Test classes: `Test*` (e.g., `TestLTX2Scheduler`, `TestGenerationConfig`)
- Test methods: `test_*` (e.g., `test_sigma_schedule_length`)

**Structure:**
```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_scheduler.py        # Scheduler tests
├── test_conditioning.py     # Conditioning tests
├── test_upscalers.py        # Upscaler component tests
├── test_pipelines.py        # Pipeline config and utility tests
├── test_loaders.py          # Weight loader tests
├── test_parity.py           # PyTorch/MLX parity tests (integration)
├── test_spatial_upscaler_parity.py   # Spatial upscaler parity
├── test_upscaler_full_parity.py      # Full upscaler parity
├── fixtures/                # Test data
└── outputs/                 # Test output directory
```

## Test Structure

**Suite Organization:**
```python
# From test_scheduler.py
class TestLTX2Scheduler:
    """Test LTX2Scheduler sigma schedule generation."""

    def test_scheduler_basic_creation(self):
        """Test that scheduler can be created with default parameters."""
        scheduler = LTX2Scheduler()
        assert scheduler is not None

    def test_sigma_schedule_length(self):
        """Test that sigma schedule has correct length."""
        steps = 25
        scheduler = LTX2Scheduler()
        sigmas = scheduler.execute(steps=steps)

        # Scheduler returns steps+1 values (including final 0)
        assert len(sigmas) == steps + 1
        assert sigmas.shape == (steps + 1,)
```

**Patterns:**
- One test class per logical component
- Docstrings for class and each test method
- Arrange-Act-Assert pattern within test methods
- Descriptive method names that explain what is being tested
- Comments inline for non-obvious test logic

**Setup/Teardown:**
- pytest fixtures via `@pytest.fixture` decorator
- Scope options: `function` (default), `class`, `module`, `session`
- Fixtures use generator pattern with `yield`: `yield result` then cleanup after
- conftest.py for shared fixtures across multiple test files

**Example from conftest.py:**
```python
@pytest.fixture
def test_logger(request) -> Generator[VerboseTestLogger, None, None]:
    """Fixture providing a verbose test logger."""
    logger = VerboseTestLogger(request.node.name)
    logger.start()
    yield logger
    logger.end()

@pytest.fixture
def temp_output_dir(tmp_path) -> Generator[Path, None, None]:
    """Fixture providing a temporary output directory."""
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir(exist_ok=True)
    yield output_dir

@pytest.fixture(scope="session")
def weights_dir() -> Path:
    """Fixture providing path to weights directory."""
    return Path(__file__).parent.parent / "weights"
```

## Mocking

**Framework:** `unittest.mock` from Python standard library

**Patterns:**
```python
# From test_pipelines.py
from unittest.mock import MagicMock

@pytest.fixture
def mock_pipeline(self):
    """Create a pipeline with mock components."""
    mock_transformer = MagicMock()
    mock_decoder = MagicMock()
    mock_patchifier = MagicMock()
    mock_guider = CFGGuider(scale=7.5)

    return TextToVideoPipeline(
        transformer=mock_transformer,
        decoder=mock_decoder,
        patchifier=mock_patchifier,
        guider=mock_guider,
    )
```

**What to Mock:**
- Heavy external dependencies (model loading, weight files)
- Components with expensive initialization
- External I/O operations in isolation tests
- Return specific values from mocked methods: `mock.return_value = expected_value`

**What NOT to Mock:**
- Core logic being tested (the unit under test)
- Configuration objects (use real instances)
- Type conversions and data transformations
- Data structures and simple utilities

## Fixtures and Factories

**Test Data:**
```python
# From test_pipelines.py - creating test configs
def test_custom_config(self):
    """Test custom configuration values."""
    config = GenerationConfig(
        height=720,
        width=1280,
        num_frames=65,
        num_inference_steps=30,
        cfg_scale=5.0,
        seed=42,
    )
    assert config.height == 720
    # ...

# From test_conditioning.py - creating test tensors
def test_basic_conditioning(self):
    """Test basic latent index conditioning."""
    target_shape = VideoLatentShape(
        batch=1,
        channels=128,
        frames=9,
        height=8,
        width=8,
    )
    cond_latent = mx.random.normal((1, 128, 3, 8, 8))
    conditioning = VideoConditionByLatentIndex(
        latent=cond_latent,
        strength=0.5,
        latent_idx=0,
    )
```

**Location:**
- Fixtures in `tests/conftest.py` for shared fixtures
- Test class fixtures defined as methods with `@pytest.fixture`
- Test data structures created inline in test methods (simple) or via fixtures (complex/reusable)
- `tests/fixtures/` directory for static test files

## Coverage

**Requirements:** None enforced (no coverage target specified)

**View Coverage:**
```bash
# Install coverage if not present
pip install pytest-cov

# Generate coverage report
pytest --cov=LTX_2_MLX --cov-report=html tests/

# View HTML report
open htmlcov/index.html
```

## Test Types

**Unit Tests:**
- Scope: Single class or function in isolation
- Scope markers: `@pytest.mark.unit` auto-applied to scheduler, conditioning, upscaler tests
- Examples: `test_scheduler.py`, `test_conditioning.py`, `test_upscalers.py`
- No external weights or heavy I/O required
- Run with: `pytest -m unit`

**Integration Tests:**
- Scope: Multiple components working together (pipeline execution, weight loading)
- Scope markers: `@pytest.mark.integration`, `@pytest.mark.requires_weights`, `@pytest.mark.slow`
- May require model weights and significant computation time
- Examples: `test_parity.py`, `test_pipelines.py` (pipeline creation tests), upscaler parity tests
- Run with: `pytest -m integration` or `pytest -m requires_weights`

**Parity Tests:**
- Special integration tests comparing MLX output to PyTorch reference
- Files: `test_parity.py`, `test_spatial_upscaler_parity.py`, `test_upscaler_full_parity.py`
- Marked with `@pytest.mark.slow` and `@pytest.mark.requires_weights`
- Use correlation metric >= 0.95 for comparing numerical outputs
- Compare arrays element-wise and compute correlation coefficients

## Markers and Test Organization

**Available Markers:** (Defined in pyproject.toml and conftest.py)
- `@pytest.mark.unit`: Unit tests without external dependencies
- `@pytest.mark.integration`: Integration tests with multiple components
- `@pytest.mark.requires_weights`: Tests that require model weight files
- `@pytest.mark.slow`: Tests with long execution time (> 30 seconds)

**Auto-Marking:** (Applied in conftest.py via `pytest_collection_modifyitems`)
- `test_scheduler*` → unit
- `test_conditioning*` → unit
- `test_upscalers*` → unit
- `test_video_generation*` → integration + requires_weights + slow

**Selection Examples:**
```bash
# Run only unit tests (fast)
pytest -m unit

# Run integration tests
pytest -m integration

# Run excluding slow tests
pytest -m "not slow"

# Run tests requiring weights only
pytest -m requires_weights
```

## Common Patterns

**Async Testing:**
Not used (synchronous only; this is a non-async codebase)

**Error Testing:**
```python
# From test_pipelines.py
def test_invalid_frame_count_raises(self):
    """Test invalid frame counts raise ValueError."""
    invalid_frames = [2, 8, 10, 16, 24, 32, 100, 120]
    for frames in invalid_frames:
        with pytest.raises(ValueError) as exc_info:
            GenerationConfig(num_frames=frames)
        assert "8*k + 1" in str(exc_info.value)

# From test_conditioning.py
def test_conditioning_raises_on_invalid_shape(self):
    """Test that conditioning raises on invalid shape."""
    with pytest.raises((ValueError, ConditioningError)):
        # Test code that should raise
        pass
```

**Parametrization:**
```python
# From test_scheduler.py - testing multiple values
def test_different_num_steps(self):
    """Test scheduler with different numbers of steps."""
    scheduler = LTX2Scheduler()
    for steps in [10, 20, 30, 50]:
        sigmas = scheduler.execute(steps=steps)
        assert len(sigmas) == steps + 1
        assert float(sigmas[0]) > 0
        assert float(sigmas[-1]) < 1e-6

# Not heavily used: @pytest.mark.parametrize decorator could be used but inline loops are preferred
```

**Numerical Assertions:**
```python
# From test_scheduler.py
def test_scheduler_reproducibility(self):
    """Test that scheduler produces same sigmas with same parameters."""
    scheduler1 = LTX2Scheduler()
    scheduler2 = LTX2Scheduler()

    sigmas1 = scheduler1.execute(steps=30)
    sigmas2 = scheduler2.execute(steps=30)

    np.testing.assert_allclose(
        np.array(sigmas1),
        np.array(sigmas2),
        rtol=1e-6,
        err_msg="Schedulers with same params should produce same sigmas"
    )

# From test_upscalers.py - comparing ranges and variances
def test_resblock_stability(self):
    """Test that ResBlock doesn't cause exponential amplification."""
    block1 = ResBlock3d(channels=512, num_groups=32)
    x = mx.random.normal((1, 512, 4, 8, 8))

    initial_var = float(mx.var(x))
    x1 = block1(x)
    var1 = float(mx.var(x1))

    assert var1 < initial_var * 10, f"Variance exploded: {initial_var} -> {var1}"
```

**Parity Comparison:**
```python
# From test_parity.py
def compute_correlation(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    flat1 = arr1.flatten().astype(np.float64)
    flat2 = arr2.flatten().astype(np.float64)
    if np.std(flat1) < 1e-8 or np.std(flat2) < 1e-8:
        return 1.0 if np.allclose(flat1, flat2) else 0.0
    return float(np.corrcoef(flat1, flat2)[0, 1])

def compare_arrays(name: str, mlx_arr: np.ndarray, pytorch_arr: np.ndarray) -> dict:
    """Compare two arrays and return metrics."""
    if mlx_arr.shape != pytorch_arr.shape:
        return {
            "name": name,
            "status": "SHAPE_MISMATCH",
            "mlx_shape": mlx_arr.shape,
            "pytorch_shape": pytorch_arr.shape,
            "correlation": 0.0,
        }

    corr = compute_correlation(mlx_arr, pytorch_arr)
    diff = np.abs(mlx_arr - pytorch_arr)

    return {
        "name": name,
        "status": "PASS" if corr >= CORRELATION_THRESHOLD else "FAIL",
        "correlation": corr,
        "max_diff": float(diff.max()),
        "mean_diff": float(diff.mean()),
    }
```

## Test Configuration

**pytest.ini Options:**
- `minversion = "7.0"`: Minimum pytest version
- `addopts = "-ra -q --strict-markers"`: Quiet output, report all outcomes, strict marker checking
- `testpaths = ["tests"]`: Search for tests in tests/ directory
- `python_files = "test_*.py"`: Test file pattern
- `python_classes = "Test*"`: Test class pattern
- `python_functions = "test_*"`: Test function pattern
- `pythonpath = "."`: Set PYTHONPATH so imports work correctly

## Example Test File Structure

```python
"""Unit tests for LTX-2 schedulers.

Tests the sigma schedule generation and Euler diffusion step logic
without requiring model weights.

Run with: pytest tests/test_scheduler.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from LTX_2_MLX.components.schedulers import LTX2Scheduler


class TestLTX2Scheduler:
    """Test LTX2Scheduler sigma schedule generation."""

    def test_scheduler_basic_creation(self):
        """Test that scheduler can be created with default parameters."""
        scheduler = LTX2Scheduler()
        assert scheduler is not None

    def test_sigma_schedule_length(self):
        """Test that sigma schedule has correct length."""
        steps = 25
        scheduler = LTX2Scheduler()
        sigmas = scheduler.execute(steps=steps)

        # Scheduler returns steps+1 values (including final 0)
        assert len(sigmas) == steps + 1
        assert sigmas.shape == (steps + 1,)

    def test_sigma_schedule_monotonic_decreasing(self):
        """Test that sigma values decrease monotonically."""
        scheduler = LTX2Scheduler()
        sigmas = scheduler.execute(steps=30)

        # Convert to numpy for easier comparison
        sigmas_np = np.array(sigmas)

        # Check monotonically decreasing
        assert np.all(
            sigmas_np[:-1] >= sigmas_np[1:]
        ), "Sigmas should decrease monotonically"
```

---

*Testing analysis: 2026-01-23*
