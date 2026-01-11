# LTX-2-MLX Project Structure

This document describes the organization of the LTX-2-MLX codebase.

## Directory Layout

```
LTX-2-MLX/
├── LTX_2_MLX/              # Main package (MLX implementation)
│   ├── components/         # Reusable diffusion components
│   │   ├── diffusion_steps.py   # Euler diffusion step
│   │   ├── guiders.py           # CFG guidance
│   │   ├── noisers.py           # Gaussian noise injection
│   │   ├── patchifiers.py       # Video latent patchification
│   │   ├── perturbations.py     # Latent perturbations
│   │   └── schedulers.py        # Sigma schedulers
│   │
│   ├── conditioning/       # Conditioning systems
│   │   ├── item.py              # ConditioningItem protocol
│   │   ├── keyframe.py          # Keyframe conditioning
│   │   ├── latent.py            # Latent-index conditioning
│   │   └── tools.py             # VideoLatentTools utilities
│   │
│   ├── loader/             # Weight loading and LoRA
│   │   ├── fp8_loader.py        # FP8 quantized weight loading
│   │   ├── lora_loader.py       # LoRA weight fusion
│   │   ├── registry.py          # Weight registry
│   │   └── weight_converter.py  # PyTorch → MLX conversion
│   │
│   ├── model/              # Neural network models
│   │   ├── audio_vae/           # Audio VAE (encoder, decoder, vocoder)
│   │   ├── text_encoder/        # Gemma-3-12B text encoder
│   │   │   ├── connector.py         # Text → latent projection
│   │   │   ├── encoder.py           # Feature extraction pipeline
│   │   │   ├── feature_extractor.py # Gemma layer processing
│   │   │   └── gemma3.py            # Gemma model implementation
│   │   ├── transformer/         # LTX-2 transformer blocks
│   │   │   ├── attention.py         # Self/cross attention
│   │   │   ├── feed_forward.py      # MLP layers
│   │   │   ├── model.py             # Model wrapper (X0Model)
│   │   │   ├── rope.py              # RoPE positional encoding
│   │   │   ├── timestep_embedding.py # AdaLN conditioning
│   │   │   └── transformer.py       # Main transformer
│   │   ├── upscaler/            # Video upscaling (2x spatial, 2x temporal)
│   │   │   ├── spatial.py           # SpatialUpscaler (H,W → H*2,W*2)
│   │   │   └── temporal.py          # TemporalUpscaler (F → F*2)
│   │   └── video_vae/           # Video VAE (latent ↔ pixel)
│   │       ├── encoder.py           # VideoEncoder
│   │       ├── decoder.py           # VideoDecoder
│   │       ├── simple_encoder.py    # SimpleVideoEncoder wrapper
│   │       ├── simple_decoder.py    # SimpleVideoDecoder wrapper
│   │       ├── convolution.py       # 3D convolutions
│   │       ├── ops.py               # VAE operations
│   │       ├── resnet.py            # ResNet blocks
│   │       ├── sampling.py          # Up/downsampling
│   │       └── tiling.py            # Tiled decoding for memory
│   │
│   ├── pipelines/          # High-level generation pipelines
│   │   ├── common.py            # Shared pipeline utilities
│   │   ├── distilled.py         # Distilled 6-step pipeline
│   │   ├── ic_lora.py           # IC-LoRA image-to-video
│   │   ├── keyframe_interpolation.py # Keyframe interpolation
│   │   ├── one_stage.py         # Single-stage CFG pipeline
│   │   ├── text_to_video.py     # Alias for distilled pipeline
│   │   └── two_stage.py         # Two-stage CFG + upsampling
│   │
│   ├── utils/              # Utilities
│   │   ├── model_ledger.py      # Model registry and loading
│   │   └── prompt_enhancement.py # Prompt enhancement
│   │
│   ├── core_utils.py       # Core utilities
│   └── types.py            # Type definitions (LatentState, VideoLatentShape, etc.)
│
├── tests/                  # Test suite
│   ├── conftest.py             # Pytest configuration and fixtures
│   ├── test_scheduler.py       # Scheduler & diffusion step tests (22 tests)
│   ├── test_conditioning.py    # Conditioning logic tests (15 tests)
│   ├── test_upscalers.py       # Upscaler component tests (23 tests)
│   ├── test_video_generation.py # Integration tests (requires weights)
│   └── README.md               # Test suite documentation
│
├── scripts/                # Executable scripts
│   ├── generate.py             # Main video generation CLI
│   └── archive/                # Archived debug scripts (49 files)
│       └── README.md               # Archive documentation
│
├── docs/                   # Documentation
│   ├── HIGH_QUALITY_GENERATION.md    # High-quality generation guide
│   ├── LTX-2-ARCHITECTURE.md         # Model architecture details
│   ├── PORTING.md                    # PyTorch → MLX porting notes
│   ├── PYTORCH_MLX_DIFFERENCES.md    # Framework differences
│   └── TROUBLESHOOTING.md            # Troubleshooting guide
│
├── examples/               # Example images for testing
│   ├── beach_palm_trees.png
│   ├── green_grass.png
│   └── sunset_car.png
│
├── weights/                # Model weights (gitignored)
│   ├── ltx-2/                  # LTX-2 transformer weights
│   └── gemma/                  # Gemma text encoder weights
│
├── gens/                   # Generated outputs (gitignored)
├── LTX-2/                  # PyTorch reference (git submodule, gitignored)
│
├── .gitignore              # Git ignore rules
├── pyproject.toml          # Python package configuration
├── README.md               # Main project documentation
└── PROJECT_STRUCTURE.md    # This file

```

## Component Responsibilities

### LTX_2_MLX/components/
**Purpose**: Reusable building blocks for diffusion pipelines

- **schedulers.py**: Sigma schedule generation (LTX2Scheduler, distilled values)
- **diffusion_steps.py**: Denoising step logic (EulerDiffusionStep)
- **guiders.py**: Classifier-free guidance (CFGGuider)
- **noisers.py**: Noise injection (GaussianNoiser)
- **patchifiers.py**: Video ↔ token conversion (VideoLatentPatchifier)

### LTX_2_MLX/conditioning/
**Purpose**: Condition video generation on images/latents

- **VideoConditionByLatentIndex**: Replace tokens at specific frame indices
- **VideoConditionByKeyframeIndex**: Append keyframe tokens with temporal offsets
- **VideoLatentTools**: Utility for managing patchified latent state

### LTX_2_MLX/loader/
**Purpose**: Load and convert model weights

- **registry.py**: Central model weight registry
- **lora_loader.py**: Fuse LoRA weights into base model
- **fp8_loader.py**: Load FP8 quantized weights (27GB vs 80GB FP32)
- **weight_converter.py**: Convert PyTorch tensors to MLX arrays

### LTX_2_MLX/model/
**Purpose**: Neural network implementations

#### model/transformer/
Core transformer blocks for diffusion denoising:
- **LTXModel**: Main transformer (19B parameters)
- **X0Model**: Wrapper converting velocity predictions to denoised samples
- **Attention**: Self-attention and cross-attention with RoPE
- **RoPEEmbedding**: Rotary position embeddings (SPLIT mode)

#### model/text_encoder/
Gemma-3-12B text encoder:
- **GemmaTextEncoder**: Full 49-layer Gemma model
- **FeatureExtractor**: Extract embeddings from Gemma layers
- **FeatureConnector**: Project text embeddings to latent space

#### model/video_vae/
Encode/decode between pixel and latent space:
- **SimpleVideoEncoder**: Encode video frames → latents (8x compression)
- **SimpleVideoDecoder**: Decode latents → video frames
- **TilingConfig**: Enable tiled decoding for low-memory systems

#### model/upscaler/
2x spatial and temporal upscaling:
- **SpatialUpscaler**: (B,C,F,H,W) → (B,C,F,H*2,W*2)
- **TemporalUpscaler**: (B,C,F,H,W) → (B,C,F*2,H,W)
- **ResBlock3d**: Stabilized residual blocks (prevents value explosion)

### LTX_2_MLX/pipelines/
**Purpose**: High-level generation pipelines

- **distilled.py**: Fast 6-step distilled pipeline (no CFG)
- **one_stage.py**: Single-stage CFG pipeline (30 steps, full resolution)
- **two_stage.py**: Two-stage pipeline (CFG @ half res + spatial upsampler + 3-step refinement)
- **ic_lora.py**: Image-conditioned LoRA pipeline
- **keyframe_interpolation.py**: Keyframe-based interpolation
- **common.py**: Shared utilities (image loading, conditioning application)

## Key Files

### Types and State Management
- **types.py**: Core type definitions
  - `LatentState`: Patchified latent with masks and positions
  - `VideoLatentShape`: Latent tensor shape specification
  - `VideoPixelShape`: Pixel-space shape specification

### Utilities
- **core_utils.py**: General utilities
- **utils/model_ledger.py**: Model weight management and loading
- **utils/prompt_enhancement.py**: Prompt preprocessing

## Code Organization Principles

1. **Modularity**: Each component is self-contained and reusable
2. **Type Safety**: Dataclasses and type hints throughout
3. **Clear Separation**: Pipelines orchestrate, components provide building blocks
4. **Minimal Duplication**: Common code in shared modules (pipelines/common.py)
5. **Testing**: Unit tests for components, integration tests for pipelines

## Import Patterns

### Internal Imports
```python
# Absolute imports from package root
from LTX_2_MLX.components import LTX2Scheduler, EulerDiffusionStep
from LTX_2_MLX.model.transformer import LTXModel
from LTX_2_MLX.pipelines.two_stage import TwoStagePipeline

# Relative imports within modules
from ..components import VideoLatentPatchifier
from .common import load_image_tensor
```

### External Dependencies
- **mlx**: Apple's ML framework (core arrays, nn modules)
- **numpy**: Array operations and conversions
- **PIL**: Image loading
- **safetensors**: Model weight loading
- **tqdm**: Progress bars

## Notable Fixes and Improvements

1. **Spatial Upscaler Stability** (LTX_2_MLX/model/upscaler/spatial.py:158)
   - Added output normalization to ResBlock3d
   - Prevents 15x-800x value amplification

2. **Code Deduplication** (LTX_2_MLX/pipelines/common.py)
   - Eliminated 400+ lines of duplicate code across 5 pipelines
   - Centralized image loading, conditioning, and state management

3. **Error Handling** (LTX_2_MLX/conditioning/latent.py:84)
   - Added bounds checking for token overflow
   - Replaced assert with proper ValueError

4. **Vectorized Upsampling** (LTX_2_MLX/pipelines/two_stage.py:345-363)
   - 2-3x speedup by processing all frames at once

## Development Workflow

### Adding New Components
1. Create module in appropriate subdirectory
2. Add unit tests in tests/test_<component>.py
3. Update __init__.py for public exports
4. Document in docstrings and this file

### Modifying Pipelines
1. Prefer editing common.py for shared functionality
2. Add error handling for edge cases
3. Update tests to cover new behavior
4. Document changes in docs/

### Running Tests
```bash
# Fast unit tests only
pytest tests/ -m unit -v

# All tests including integration
pytest tests/ -v
```

## Future Enhancements

1. **Audio Integration**: Complete audio generation in two-stage pipeline
2. **FP8 Support**: Finish FP8 quantized weight loading
3. **LoRA Testing**: Add tests for LoRA weight fusion
4. **Pipeline Exposure**: Add CLI flags for IC-LoRA and keyframe pipelines
5. **Performance**: Profile and optimize hotspots
