# Technology Stack

**Analysis Date:** 2026-01-23

## Languages

**Primary:**
- Python 3.10+ - Core implementation language for all ML models and pipelines
- Python 3.11 - Tested and supported
- Python 3.12 - Tested and supported

**Secondary:**
- None - Pure Python implementation with compiled acceleration via MLX

## Runtime

**Environment:**
- macOS with Apple Silicon (M1/M2/M3/M4 required)
- Python 3.10 or higher

**Package Manager:**
- uv (recommended) - Modern, fast Python package manager
- pip - Standard Python package manager (alternative)
- Lockfile: `uv.lock` present for reproducible installations

## Frameworks

**Core ML Framework:**
- MLX >=0.20.0 - Apple Silicon-optimized ML library, primary compute backend
- Location: `LTX_2_MLX/model/`

**Model Components:**
- Transformers >=4.57.3 - For text encoder tokenization and utilities
- safetensors >=0.4.0 - Weight serialization and loading format
- einops >=0.7.0 - Tensor manipulation and reshaping utilities
- numpy >=1.24.0 - Numerical computing

**Data & Media:**
- Pillow >=10.0.0 - Image processing for frame generation
- tqdm >=4.65.0 - Progress bars for long-running operations
- ffmpeg (system requirement via `brew install ffmpeg`) - Video encoding/decoding

**Text Encoding:**
- sentencepiece >=0.2.1 - Tokenization for Gemma 3 text encoder
- protobuf >=6.33.3 - Protocol buffer support for model serialization

**Testing:**
- pytest >=7.4.0 - Test framework
- pytest-timeout >=2.1.0 - Test timeout management
- Config: `pyproject.toml` contains pytest configuration

**Development & Code Quality:**
- black >=23.0.0 - Code formatter (line-length: 100)
- ruff >=0.1.0 - Linter (Python 3.10 target)
- pyright >=1.1.0 - Static type checker

## Key Dependencies

**Critical (Production):**
- mlx >=0.20.0 - Why it matters: Core compute engine for all inference on Apple Silicon; enables GPU acceleration
- transformers >=4.57.3 - Why it matters: Provides AutoTokenizer for Gemma 3 text encoding
- safetensors >=0.4.0 - Why it matters: Loads pre-trained model weights (.safetensors format)
- numpy >=1.24.0 - Why it matters: Numerical operations for tensor manipulations
- torch >=2.9.1 - Why it matters: Used for PyTorch weight conversion (optional, for development)

**Infrastructure:**
- huggingface_hub - Used only in `scripts/download_weights.py` for downloading model weights; not a production dependency
- rich - Used in `scripts/download_weights.py` for formatted terminal output; not a production dependency

## Configuration

**Environment:**
- Python version: Enforced via `requires-python = ">=3.10"` in `pyproject.toml`
- Platform: macOS (darwin) with ARM64 architecture assumed

**Build:**
- Build system: hatchling (backend)
- Build config: `pyproject.toml` lines 64-65
- Package name: `ltx-2-mlx`
- Version: 0.1.0

**Linting & Formatting:**
- Formatter: black with line-length 100
- Linter: ruff with rules: E, W, F, I, B, C4, UP (see `pyproject.toml` lines 107-122)
- Type checker: pyright targeting Python 3.10 on Darwin platform

**Testing Configuration:**
- Test directory: `tests/`
- Markers defined: `slow`, `requires_weights`, `integration`, `unit`
- PYTHONPATH: `.` (current directory)
- Test discovery: `test_*.py` files

## Platform Requirements

**Development:**
- macOS with Apple Silicon (M1/M2/M3/M4)
- ~25GB RAM minimum (128GB recommended for high resolution)
- ffmpeg installed via Homebrew: `brew install ffmpeg`
- Python 3.10+ installation
- uv package manager installed: `curl -LsSf https://astral.sh/uv/install.sh | sh`

**Production:**
- Same macOS + Apple Silicon requirements
- Same ffmpeg requirement
- ~25GB free disk space for model weights (43GB+ for full model suite)

## Optional Dependencies

**Development Only:**
- pytest, pytest-timeout for running test suite
- black for code formatting
- ruff for linting
- pyright for type checking

**Script-Only (not required for core inference):**
- huggingface_hub - Download model weights from HuggingFace Hub
- rich - Terminal UI for download script

---

*Stack analysis: 2026-01-23*
