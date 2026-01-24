# Coding Conventions

**Analysis Date:** 2026-01-23

## Naming Patterns

**Files:**
- Module files use `snake_case`: `schedulers.py`, `diffusion_steps.py`, `video_vae.py`
- Test files follow pytest convention: `test_*.py` (e.g., `test_scheduler.py`, `test_upscalers.py`)
- Package directories use `snake_case`: `conditioning/`, `pipelines/`, `model/`
- Special marker for main package directory uses SCREAMING_SNAKE_CASE: `LTX_2_MLX/`

**Functions:**
- Use `snake_case` for all functions: `rms_norm()`, `load_image_tensor()`, `to_velocity()`
- Utility functions are lowercase: `check_config_value()`, `compute_correlation()`
- Factory functions follow pattern `create_*`: `create_image_conditionings()`, `create_pipeline()`, `create_ic_lora_pipeline()`

**Variables:**
- Use `snake_case` for all variables: `sigma_shift`, `image_path`, `denoise_mask`
- Constants in SCREAMING_SNAKE_CASE: `BASE_SHIFT_ANCHOR`, `MAX_SHIFT_ANCHOR`, `VIDEO_SCALE_FACTORS`, `CORRELATION_THRESHOLD`
- Tensor/array variables often have descriptive suffixes: `sigmas_np`, `img_mx`, `initial_state`
- State variables use descriptive names: `LatentState`, `PipelineState`, `initial_state`

**Types/Classes:**
- Classes use `PascalCase`: `VideoPixelShape`, `SpatioTemporalScaleFactors`, `LatentState`, `VideoConditionByLatentIndex`
- Exception classes use `PascalCase` with `Error` suffix: `ConditioningError`
- Enum classes use `PascalCase`: `ControlType`, `Modality`
- Test classes use `Test` prefix: `TestLTX2Scheduler`, `TestVideoConditionByLatentIndex`, `TestGenerationConfig`

**Protocols/Interfaces:**
- Protocol classes follow interface convention with `Protocol` suffix or name: `SchedulerProtocol`, `ConditioningItem`

## Code Style

**Formatting:**
- Line length: 100 characters (enforced by Black)
- Tool: `black==23.0.0` for code formatting
- Target Python versions: 3.10, 3.11, 3.12
- Indentation: 4 spaces (enforced by Black)

**Linting:**
- Tool: `ruff>=0.1.0`
- Enabled rules: E (errors), W (warnings), F (pyflakes), I (isort), B (flake8-bugbear), C4 (comprehensions), UP (pyupgrade)
- Ignored rules: E501 (line length, handled by Black), B008 (function calls in defaults), C901 (complexity), W191 (tabs)
- File-specific: `__init__.py` ignores F401 (unused imports allowed for re-exports)

**Type Checking:**
- Tool: `pyright>=1.1.0`
- Target Python: 3.10
- Platform: Darwin
- `reportMissingImports = true`
- `reportMissingTypeStubs = false`

## Import Organization

**Order:**
1. Standard library imports (`sys`, `os`, `math`, `time`, `json`, `dataclasses`, `typing`, `pathlib`)
2. Third-party imports (`mlx`, `numpy`, `PIL`, `torch`, `transformers`, `safetensors`, `tqdm`)
3. Local imports (relative imports from package using `from ..` or `from .`)

**Examples from codebase:**
```python
# From core_utils.py
from typing import Any, Union
import mlx.core as mx
```

```python
# From pipelines/common.py
import os
from dataclasses import dataclass
from typing import List
import mlx.core as mx
import numpy as np
from PIL import Image
from ..conditioning.item import ConditioningItem
from ..types import LatentState
```

**Path Aliases:**
- Relative imports from package root: `from LTX_2_MLX.types import VideoPixelShape`
- Relative imports within package: `from ..types import LatentState` (parent) or `from .item import ConditioningItem` (same)
- Prefer relative imports within the package

**Import Handling:**
- Conditional/optional imports wrapped in try/except for ImportError: Used in VAE modules for optional dependencies
- Lazy imports within functions only when necessary for optional features
- Explicit re-exports in `__init__.py` files for public API (F401 ignored)

## Error Handling

**Patterns:**
- Raise specific exceptions with descriptive messages: `ValueError`, `FileNotFoundError`, `ImportError`
- Include context in error messages: `f"Config value {key} is {actual}, expected {expected}"`
- Validate inputs at function entry: Check shape, type, value ranges
- Use try/except for external library integration (e.g., importing optional modules)

**Common Validations:**
- Shape validation with clear error messages: `raise ValueError(f"Invalid input shape: {x.shape}, expected 4D or 5D")`
- File existence checks: `if not os.path.exists(image_path): raise FileNotFoundError(f"Image not found: {image_path}")`
- Configuration validation: `if actual != expected: raise ValueError(message)`
- Custom exceptions for domain-specific errors: `ConditioningError` for conditioning issues

**Example from core_utils.py:**
```python
def to_velocity(sample: mx.array, sigma: Union[float, mx.array], denoised_sample: mx.array) -> mx.array:
    # Convert sigma to scalar if it's an array
    if isinstance(sigma, mx.array):
        sigma = float(sigma.item())

    if sigma == 0:
        raise ValueError("Sigma can't be 0.0")

    # Process...
```

**Example from pipelines/common.py:**
```python
def load_image_tensor(image_path: str, height: int, width: int, dtype: mx.Dtype = mx.float32) -> mx.array:
    # Validate file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        img = Image.open(image_path)
    except Exception as e:
        raise ValueError(f"Failed to open image {image_path}: {e}")

    # Validate format
    if img.mode not in ['RGB', 'RGBA', 'L']:
        raise ValueError(f"Unsupported image format: {img.mode}. Supported formats: RGB, RGBA, L")
```

## Logging

**Framework:** `print()` statements via pytest's capturing mechanism

**Patterns:**
- Verbose logging via custom `VerboseTestLogger` class in tests
- Test logging includes timestamps: `[HH:MM:SS] [LEVEL] test_name: message`
- Log levels: INFO, STEP, WARN (capitalized)
- No structured logging library; use formatted strings with clear prefixes
- Tests can access logger via `test_logger` fixture from conftest

**Example from conftest.py:**
```python
def log(self, message: str, level: str = "INFO"):
    """Log a message with timestamp."""
    import time
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {self.test_name}: {message}")
```

## Comments

**When to Comment:**
- Document complex mathematical operations: Explain transformations, formulas
- Clarify non-obvious design decisions: Why this approach was chosen
- Mark important preconditions or invariants
- Note implementation details that differ from typical patterns
- Avoid obvious comments (e.g., `# increment counter`)

**Pattern Examples:**
```python
# From core_utils.py - explain sigma conversion
# Convert sigma to scalar if it's an array
if isinstance(sigma, mx.array):
    sigma = float(sigma.item())

# From schedulers.py - explain complex math
# Linear spacing from 1.0 to 0.0
sigmas = mx.linspace(1.0, 0.0, steps + 1)

# Compute shift based on token count (linear interpolation)
x1 = BASE_SHIFT_ANCHOR
x2 = MAX_SHIFT_ANCHOR
mm = (max_shift - base_shift) / (x2 - x1)
b = base_shift - mm * x1
sigma_shift = tokens * mm + b

# Avoid division by zero for sigmas == 0
# sigmas_transformed = exp_shift / (exp_shift + (1/sigmas - 1)^power)
sigmas_transformed = mx.where(sigmas != 0, ..., mx.zeros_like(sigmas))
```

## Docstrings

**Format:** Google-style docstrings for all public functions and classes

**Components:**
- One-line summary (present tense): "Generate sigma schedule", "Convert sample to velocity"
- Blank line, then detailed description if needed
- `Args:` section with type and description for each parameter
- `Returns:` section with type and description
- `Raises:` section for exceptions that may be raised
- Optional `Examples:` section for complex usage

**Examples from codebase:**

```python
# From types.py
def from_shape(shape: Tuple[int, ...]) -> "VideoLatentShape":
    """Convert from tuple shape to VideoLatentShape."""
    return VideoLatentShape(...)

# From core_utils.py
def rms_norm(
    x: mx.array, weight: mx.array | None = None, eps: float = 1e-6
) -> mx.array:
    """
    Root-mean-square (RMS) normalize `x` over its last dimension.

    Uses optimized mx.fast.rms_norm Metal kernel for efficiency.

    Args:
        x: Input tensor to normalize.
        weight: Optional learnable scale parameter.
        eps: Small constant for numerical stability.

    Returns:
        RMS normalized tensor.
    """
    return mx.fast.rms_norm(x, weight, eps)

# From pipelines/common.py
def load_image_tensor(
    image_path: str,
    height: int,
    width: int,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Load an image and prepare for VAE encoding.

    Args:
        image_path: Path to the image file
        height: Target height in pixels
        width: Target width in pixels
        dtype: Output data type

    Returns:
        Image tensor of shape (1, C, 1, H, W) normalized to [-1, 1]

    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the image format is unsupported or loading fails
    """
```

## Function Design

**Size:** Functions are typically 10-50 lines; complex operations broken into smaller functions
- Utility/core functions: 5-20 lines
- Pipeline methods: 20-50 lines
- Factory functions: 30-80 lines (they compose multiple components)

**Parameters:**
- Max 5-7 parameters for regular functions
- Use dataclasses for configuration: `GenerationConfig`, `DistilledConfig`, `ImageCondition`
- Optional parameters have defaults: `strength: float = 0.95`, `dtype: mx.Dtype = mx.float32`
- Use keyword-only arguments with `*` separator when many parameters: Not heavily used but available

**Return Values:**
- Single return value per function (no multiple returns)
- Return type explicitly annotated: `-> mx.array`, `-> List[ConditioningItem]`
- Dataclasses for multiple related returns
- Generators used: `-> Generator[T, None, None]` for fixtures

**Type Annotations:**
- All function parameters and returns have type hints
- Union types: `Union[float, mx.array]` or `float | mx.array` (Python 3.10+)
- Optional types: `Optional[mx.array]` (same as `mx.array | None`)
- Protocol types for abstractions: `ConditioningItem` protocol
- Generic types from typing: `List`, `Tuple`, `Dict`, `Generator`

## Module Design

**Exports:**
- Explicit in `__init__.py` files with clear imports
- Unused imports in `__init__.py` allowed (F401 ignored) for re-exporting
- Example from `LTX_2_MLX/model/video_vae/__init__.py`: Imports and re-exports convolution, ops, resnet classes

**Barrel Files:**
- Used throughout: `__init__.py` collects related classes/functions
- Example: `pipelines/__init__.py` exports pipeline creation functions
- Purpose: Simplify imports for users of the package

**Module Responsibilities:**
- `conditioning/`: Conditioning items and state manipulation
- `pipelines/`: End-to-end generation pipelines and configurations
- `components/`: Diffusion-specific components (schedulers, noise schedulers, patchifiers)
- `model/`: Neural network layers and modules (transformer, VAE, upscalers)
- `loader/`: Weight loading and model registry
- `utils/`: Helper utilities (model tracking, prompt enhancement)

## Dataclass Patterns

**Usage:**
- Configuration objects: `GenerationConfig`, `DistilledConfig`, `TwoStageCFGConfig`
- State containers: `LatentState`, `PipelineState`
- Data wrappers: `ImageCondition`, `VideoCondition`, `AudioLatentShape`
- Type definitions: `VideoPixelShape`, `SpatioTemporalScaleFactors`

**Conventions:**
- Use `@dataclass` decorator for immutable data: `@dataclass(frozen=True)` for `LatentState`
- Use standard `@dataclass` for mutable configs
- NamedTuple for simple type definitions: `VideoPixelShape`, `SpatioTemporalScaleFactors`
- Field defaults for optional parameters: `strength: float = 0.95`

---

*Convention analysis: 2026-01-23*
