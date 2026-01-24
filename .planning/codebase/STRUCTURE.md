# Codebase Structure

**Analysis Date:** 2026-01-23

## Directory Layout

```
LTX-2-MLX/
├── LTX_2_MLX/                 # Main package (18.8K lines)
│   ├── __init__.py            # Package root, exports types and core utils
│   ├── types.py               # Type definitions (193 lines)
│   ├── core_utils.py          # Shared utilities: RMS norm, velocity conversion
│   │
│   ├── model/                 # Neural network components
│   │   ├── transformer/       # LTX transformer (video/audio)
│   │   │   ├── model.py       # LTXModel, LTXAVModel, Modality, preprocessors
│   │   │   ├── transformer.py # BasicTransformerBlock, BasicAVTransformerBlock
│   │   │   ├── attention.py   # Multi-head attention implementation
│   │   │   ├── feed_forward.py # Feed-forward networks
│   │   │   ├── rope.py        # RoPE positional encodings
│   │   │   ├── timestep_embedding.py # AdaLayerNormSingle
│   │   │   └── __init__.py    # Exports model classes
│   │   │
│   │   ├── video_vae/         # Video VAE encoder/decoder
│   │   │   ├── simple_decoder.py  # SimpleVideoDecoder (809 lines)
│   │   │   ├── simple_encoder.py  # SimpleVideoEncoder (564 lines)
│   │   │   ├── decoder.py     # Full VideoDecoder (not used, kept for reference)
│   │   │   ├── encoder.py     # Full VideoEncoder (300 lines)
│   │   │   ├── convolution.py # 3D conv operations, normalization
│   │   │   ├── resnet.py      # ResnetBlock3D, UNetMidBlock3D
│   │   │   ├── ops.py         # patchify, unpatchify, pixel_shuffle_3d
│   │   │   ├── sampling.py    # Space-to-depth upsampling
│   │   │   ├── tiling.py      # Tiled decoding for memory efficiency (475 lines)
│   │   │   └── __init__.py    # Exports all VAE components
│   │   │
│   │   ├── text_encoder/      # Text encoding (Gemma 3)
│   │   │   ├── gemma3.py      # Gemma3Model, config loading
│   │   │   ├── encoder.py     # TextEncoder wrapper, Gemma 3 + projection
│   │   │   ├── connector.py   # AVTextEncoder (audio-video unified encoder)
│   │   │   ├── feature_extractor.py # Image feature extraction for IC-LoRA
│   │   │   └── __init__.py    # Exports text encoder classes
│   │   │
│   │   ├── audio_vae/         # Audio VAE encoder/decoder
│   │   │   ├── decoder.py     # AudioDecoder (mel-spectrogram → latent)
│   │   │   ├── encoder.py     # AudioEncoder
│   │   │   ├── vocoder.py     # Vocoder (latent → waveform)
│   │   │   └── __init__.py    # Exports audio components
│   │   │
│   │   ├── upscaler/          # 2x spatial and temporal upscalers
│   │   │   ├── spatial.py     # SpatialUpscaler (U-Net based)
│   │   │   ├── temporal.py    # TemporalUpscaler
│   │   │   └── __init__.py    # Exports upscaler classes
│   │   │
│   │   └── __init__.py        # Empty, model components exported individually
│   │
│   ├── components/            # Diffusion sampling utilities
│   │   ├── schedulers.py      # LTX2Scheduler, LinearQuadraticScheduler, get_sigma_schedule
│   │   ├── guiders.py         # CFGGuider, STGGuider, LtxAPGGuider, etc.
│   │   ├── noisers.py         # GaussianNoiser, DeterministicNoiser
│   │   ├── diffusion_steps.py # EulerDiffusionStep, HeunDiffusionStep
│   │   ├── patchifiers.py     # VideoLatentPatchifier, AudioPatchifier
│   │   ├── perturbations.py   # STG perturbations for structural guidance
│   │   └── __init__.py        # Exports all component classes
│   │
│   ├── pipelines/             # Generation pipelines
│   │   ├── text_to_video.py   # TextToVideoPipeline, legacy basic pipeline (377 lines)
│   │   ├── one_stage.py       # OneStagePipeline with CFG (537 lines) ⭐ recommended
│   │   ├── two_stage.py       # TwoStagePipeline with upscaling + LoRA (812 lines) ⭐ high-quality
│   │   ├── distilled.py       # DistilledPipeline fast generation (382 lines)
│   │   ├── keyframe_interpolation.py # KeyframeInterpolationPipeline (526 lines)
│   │   ├── ic_lora.py         # ICLoraPipeline for image conditioning with LoRA (785 lines)
│   │   ├── common.py          # Shared utilities: ImageCondition, load_image_tensor, etc.
│   │   └── __init__.py        # Exports all pipeline classes and config dataclasses
│   │
│   ├── conditioning/          # Frame/image conditioning logic
│   │   ├── item.py            # ConditioningItem (frame, latent, denoise_mask, blend_mode)
│   │   ├── tools.py           # VideoLatentTools, AudioLatentTools for coordinate conversion
│   │   ├── keyframe.py        # VideoConditionByKeyframeIndex
│   │   ├── latent.py          # VideoConditionByLatentIndex, ConditioningError
│   │   └── __init__.py        # Exports conditioning classes
│   │
│   ├── loader/                # Weight loading and conversion
│   │   ├── weight_converter.py # load_safetensors, convert_*_key, extract_*_weights (500+ lines)
│   │   ├── lora_loader.py      # LoRA loading, fusion, target module registry
│   │   ├── registry.py         # Registry pattern for state dict management
│   │   └── __init__.py         # Exports loader functions and classes
│   │
│   ├── utils/                 # Utility functions
│   │   ├── model_ledger.py    # Model catalog with weight URLs and configs
│   │   ├── prompt_enhancement.py # Optional prompt enhancement with LLMs
│   │   └── __init__.py        # Exports utilities
│   │
│   └── kernels/               # Fused operations (optional)
│       ├── fused_ops.py       # Future: fused kernels
│       └── __init__.py
│
├── scripts/                   # Entry points and utilities
│   ├── generate.py            # Main video generation script (98KB, 2900+ lines)
│   ├── download_weights.py    # Download models from HuggingFace
│   └── generate_pytorch_checkpoints.py # Convert PyTorch → safetensors reference
│
├── tests/                     # Test suite
│   ├── conftest.py            # pytest configuration and fixtures
│   ├── test_parity.py         # Compare MLX vs PyTorch outputs
│   ├── test_spatial_upscaler_parity.py
│   ├── test_upscaler_full_parity.py
│   ├── test_pipelines.py      # Pipeline integration tests
│   ├── test_conditioning.py   # Conditioning logic tests
│   ├── test_scheduler.py      # Scheduler tests
│   ├── test_loaders.py        # Weight loading tests
│   ├── test_upscalers.py      # Upscaler component tests
│   ├── compare_upscaler_outputs.py
│   ├── fixtures/              # Test data (embeddings, schedules, weights)
│   │   ├── embeddings/
│   │   ├── guidance/
│   │   ├── pipeline/
│   │   ├── sigma_schedules/
│   │   ├── transformer/
│   │   └── vae/
│   └── outputs/               # Test outputs directory
│
├── docs/                      # Documentation
│   ├── PIPELINES.md           # Detailed pipeline guide
│   ├── USAGE.md               # Usage instructions and examples
│   ├── ARCHITECTURE.md        # Architecture overview
│   ├── PARITY.md              # PyTorch/MLX verification details
│   └── [other docs]
│
├── examples/                  # Pre-generated video examples
│   ├── golden_retriever.mp4   # Example outputs
│   ├── ocean_sunset.mp4
│   └── [other examples]
│
├── weights/                   # Downloaded model weights
│   ├── ltx-2/                 # Transformer weights
│   ├── gemma-3-12b/           # Text encoder weights
│   ├── loras/                 # LoRA weights
│   │   ├── camera-control/
│   │   └── ic/
│   └── [other weights]
│
├── outputs/                   # Generated video outputs
│   └── [user-generated content]
│
├── pyproject.toml             # Package metadata, dependencies, tool config
├── README.md                  # Project overview
├── .gitignore                 # Git ignore rules
├── .planning/                 # GSD planning documents
│   └── codebase/              # Architecture analysis (this file)
└── [configuration files]
```

## Directory Purposes

**`LTX_2_MLX/`:**
- Purpose: Main package containing all implementation code
- Contains: Models, pipelines, components, loaders, utilities, types
- Key files: `__init__.py`, `types.py`, `core_utils.py`

**`LTX_2_MLX/model/`:**
- Purpose: Encapsulates all neural network model implementations
- Contains: Transformer, VAE, text encoder, audio components, upscalers
- Key files: Each subdir has `__init__.py` exporting public API
- Entry point: `model.transformer.LTXModel`, `model.transformer.LTXAVModel`

**`LTX_2_MLX/model/transformer/`:**
- Purpose: Multi-modal transformer for video/audio denoising
- Contains: Main model, attention, feed-forward, positional encodings, timestep embeddings
- Key files: `model.py` (LTXModel, 400+ lines), `transformer.py` (blocks)
- Structure: Input → preprocessor (patchify, embeddings) → stack of BasicTransformerBlocks → output

**`LTX_2_MLX/model/video_vae/`:**
- Purpose: Video codec for compressing/decompressing video latents
- Contains: Encoder, decoder, 3D convolutions, resampling, tiling
- Key files: `simple_decoder.py` (primary, 809 lines), `simple_encoder.py` (564 lines)
- Notable: Tiling support for decoding large videos without loading entire latent

**`LTX_2_MLX/components/`:**
- Purpose: Reusable diffusion sampling components (not model-specific)
- Contains: Schedulers, guiders, noisers, patchifiers, diffusion steps, perturbations
- Key files: One file per concept class
- Pattern: Each component follows Protocol (duck-typing interface)

**`LTX_2_MLX/pipelines/`:**
- Purpose: Complete generation workflows combining models + components
- Contains: 6 pipeline implementations + common utilities
- Key files:
  - `text_to_video.py` - Legacy baseline (simple CFG loop)
  - `one_stage.py` - Standard CFG pipeline (recommended)
  - `two_stage.py` - CFG + upscaling + LoRA refinement (highest quality)
  - `distilled.py` - Fast generation with distilled model
  - `keyframe_interpolation.py` - Generate between keyframes
  - `ic_lora.py` - Image/video conditioning with LoRA
- Entry: Factory functions `create_*_pipeline()`

**`LTX_2_MLX/conditioning/`:**
- Purpose: Manage latent-space modifications for conditioning
- Contains: Frame conditioning items, latent coordinate tools, indexing strategies
- Key files: `tools.py` (VideoLatentTools handles frame↔latent conversion), `latent.py` (indexing)
- Pattern: `ConditioningItem` + `VideoLatentTools` → apply in diffusion loop

**`LTX_2_MLX/loader/`:**
- Purpose: Weight management: convert PyTorch → MLX, handle LoRA
- Contains: SafeTensors loader, key converters, weight extractors, LoRA fusion
- Key files: `weight_converter.py` (500+ lines), `lora_loader.py`
- Pattern: `load_safetensors()` → `extract_*_weights()` → model.update(converted_weights)

**`scripts/`:**
- Purpose: Executable entry points
- Contains: `generate.py` (main), download script, reference converters
- Key files: `generate.py` (2900+ lines) - all CLI logic, pipeline creation, model loading

**`tests/`:**
- Purpose: Unit, integration, and parity tests
- Contains: Test files for each component, fixtures with test data
- Key files: `test_parity.py` (MLX vs PyTorch comparison), `test_pipelines.py` (end-to-end)
- Fixtures: `fixtures/` contains pre-computed embeddings, schedules, weights for reproducibility
- Pattern: Tests use markers: `@pytest.mark.requires_weights`, `@pytest.mark.slow`

## Key File Locations

**Entry Points:**
- `scripts/generate.py`: Command-line video generation (execute here to generate)
- `LTX_2_MLX/pipelines/__init__.py`: Pipeline class exports (import pipelines from here)

**Configuration:**
- `pyproject.toml`: Project metadata, dependencies, tool configs (pytest, black, ruff, pyright)
- `LTX_2_MLX/pipelines/one_stage.py:OneStageCFGConfig`: Config dataclass for one-stage pipeline
- `LTX_2_MLX/pipelines/two_stage.py:TwoStageCFGConfig`: Config dataclass for two-stage pipeline

**Core Logic:**
- `LTX_2_MLX/model/transformer/model.py`: LTXModel, LTXAVModel (main diffusion model)
- `LTX_2_MLX/pipelines/one_stage.py`: OneStagePipeline (recommended pipeline)
- `LTX_2_MLX/model/video_vae/simple_decoder.py`: Video latent decoder
- `LTX_2_MLX/loader/weight_converter.py`: Weight conversion logic

**Testing:**
- `tests/conftest.py`: pytest setup, fixtures, weight loading for tests
- `tests/test_parity.py`: MLX vs PyTorch output comparison
- `tests/test_pipelines.py`: Pipeline integration tests
- `tests/fixtures/`: Pre-computed test data

**Types & Utilities:**
- `LTX_2_MLX/types.py`: VideoLatentShape, AudioLatentShape, LatentState, VideoPixelShape
- `LTX_2_MLX/core_utils.py`: rms_norm, to_velocity, to_denoised

## Naming Conventions

**Files:**
- `snake_case.py`: All Python files use snake_case (e.g., `weight_converter.py`, `video_vae.py`)
- `__init__.py`: Package markers, export public API via `__all__`
- Test files: `test_*.py` or `*_test.py` (all use `test_` prefix)

**Directories:**
- `snake_case/`: All package directories use snake_case (e.g., `model/`, `components/`, `pipelines/`)
- Logical grouping by feature: `video_vae/`, `text_encoder/`, `audio_vae/`, `transformer/`

**Classes:**
- `PascalCase`: All classes use PascalCase (e.g., `LTXModel`, `OneStagePipeline`, `CFGGuider`)
- Dataclasses: Use `Config` suffix for configuration (e.g., `OneStageCFGConfig`, `TwoStageCFGConfig`)
- Protocols: Use `Protocol` suffix (e.g., `GuiderProtocol`, `DiffusionStepProtocol`)

**Functions:**
- `snake_case`: All functions use snake_case (e.g., `create_pipeline`, `load_transformer_weights`)
- Factory functions: Prefix with `create_` or `load_` (e.g., `create_one_stage_pipeline`, `load_safetensors`)
- Private functions: Prefix with `_` (e.g., `_prepare_timestep`)

**Variables:**
- `snake_case`: Local variables, parameters, constants (e.g., `latent`, `text_encoding`, `sigma_schedule`)
- Constants: `UPPER_CASE` for module-level constants (e.g., `DISTILLED_SIGMA_VALUES`, `VIDEO_SCALE_FACTORS`)

**Type Hints:**
- Imports: `from typing import Optional, List, Tuple, Dict, Union, Callable`
- MLX arrays: Annotate as `mx.array` (no type parameters, MLX doesn't have generic typing)
- Optional fields: Use `Optional[Type]` = None pattern in dataclasses

## Where to Add New Code

**New Pipeline:**
- Primary code: `LTX_2_MLX/pipelines/{new_pipeline_name}.py`
- Config dataclass: Define `{Name}Config` in same file
- Factory function: Export `create_{new_pipeline_name}_pipeline()` from `__init__.py`
- Tests: `tests/test_pipelines.py` or new `tests/test_{name}_pipeline.py`

**New Component (Guider, Scheduler, etc.):**
- Implementation: `LTX_2_MLX/components/{component_type}.py` (add to existing file if same type)
- Protocol: Update protocol class in same file if creating new interface
- Exports: Add to `LTX_2_MLX/components/__init__.py` in `__all__`
- Tests: `tests/test_{component_type}.py`

**New Model Component:**
- Implementation: `LTX_2_MLX/model/{feature_name}/`
- Sub-files: Organize by responsibility (e.g., `encoder.py`, `decoder.py`, `ops.py`)
- Exports: `LTX_2_MLX/model/{feature_name}/__init__.py` with `__all__`
- Tests: `tests/test_{feature_name}.py`

**Utilities:**
- Shared helpers: `LTX_2_MLX/utils/{utility_name}.py`
- Cross-component utilities: Add to `core_utils.py` if used by many modules
- Exports: List in `LTX_2_MLX/utils/__init__.py`

**Tests:**
- Unit tests: `tests/test_{module_name}.py`
- Use fixtures from `conftest.py` (e.g., `weights_dir`, `device`)
- Mark with decorators: `@pytest.mark.unit`, `@pytest.mark.requires_weights`, `@pytest.mark.slow`

## Special Directories

**`LTX_2_MLX/`:**
- Purpose: Main package namespace
- Generated: No (hand-written)
- Committed: Yes (core code)

**`weights/`:**
- Purpose: Downloaded model weight files
- Generated: Yes (by `scripts/download_weights.py`)
- Committed: No (git-ignored, too large for git)
- Location: Downloaded via HuggingFace `hub_download()` or manual download

**`outputs/`:**
- Purpose: User-generated video/audio outputs
- Generated: Yes (by `scripts/generate.py`)
- Committed: No (git-ignored, user content)

**`tests/fixtures/`:**
- Purpose: Pre-computed test data (embeddings, schedules, reference outputs)
- Generated: No (committed reference data for reproducible tests)
- Committed: Yes (small test data, not full weights)

**`.planning/codebase/`:**
- Purpose: GSD codebase analysis documents (ARCHITECTURE.md, STRUCTURE.md, etc.)
- Generated: Yes (by GSD mapper)
- Committed: Yes (guides future implementations)

---

*Structure analysis: 2026-01-23*
