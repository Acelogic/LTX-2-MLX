# Architecture

**Analysis Date:** 2026-01-23

## Pattern Overview

**Overall:** Modular Video Generation Framework

LTX-2-MLX implements a **layered pipeline architecture** for text-to-video generation. The codebase is organized into independent, composable layers with clear separation of concerns:

1. **Model Layer** - Neural network components (transformer, VAE encoders/decoders)
2. **Components Layer** - Diffusion utilities (schedulers, guiders, noisers, patchifiers)
3. **Pipeline Layer** - Generation strategies (text-to-video, two-stage, keyframe interpolation)
4. **Loader Layer** - Weight conversion and LoRA management
5. **Conditioning Layer** - Input masking and frame replacement logic
6. **Type/Utils Layer** - Shared types and utilities

**Key Characteristics:**
- MLX-native implementation for Apple Silicon acceleration
- Protocol-based abstraction for pluggable components (diffusion steps, guiders)
- Multi-format support (video, audio, images as conditioning)
- Two-stage generation with optional upscaling and LoRA refinement
- Modular pipeline implementations (text-to-video, one-stage, two-stage, keyframe interpolation, IC-LoRA, distilled)

## Layers

**Model Layer:**
- Purpose: Core neural network implementations ported from PyTorch to MLX
- Location: `LTX_2_MLX/model/`
- Contains: Transformer, text encoder (Gemma 3), video/audio VAE, upscalers
- Depends on: MLX, safetensors for weight loading
- Used by: Pipelines pass model instances to generate video

**Components Layer:**
- Purpose: Reusable diffusion sampling utilities independent of any specific model
- Location: `LTX_2_MLX/components/`
- Contains: Schedulers (sigma schedules), guiders (CFG, STG, APG), noisers, patchifiers, diffusion steps (Euler, Heun), perturbations
- Depends on: Core utils, MLX arrays
- Used by: Pipelines instantiate components and pass them to generator loops

**Pipeline Layer:**
- Purpose: Orchestrates generation workflow by composing models and components
- Location: `LTX_2_MLX/pipelines/`
- Contains: TextToVideoPipeline, OneStagePipeline, TwoStagePipeline, DistilledPipeline, KeyframeInterpolationPipeline, ICLoraPipeline
- Depends on: Model layer, components layer, conditioning layer
- Used by: Entry scripts (generate.py) instantiate and call pipelines

**Loader Layer:**
- Purpose: Handles weight conversion from PyTorch safetensors format to MLX, LoRA fusion
- Location: `LTX_2_MLX/loader/`
- Contains: Weight converter, safetensors loader, LoRA loader with target module registry
- Depends on: safetensors library, MLX
- Used by: Pipelines call loader functions to initialize model weights

**Conditioning Layer:**
- Purpose: Manages latent space modifications for image/video conditioning and keyframe controls
- Location: `LTX_2_MLX/conditioning/`
- Contains: ConditioningItem, VideoLatentTools, latent indexing strategies
- Depends on: Types layer
- Used by: Pipelines apply conditioning items to latent state during diffusion

**Type/Utils Layer:**
- Purpose: Shared type definitions and utility functions
- Location: `LTX_2_MLX/types.py`, `LTX_2_MLX/core_utils.py`
- Contains: VideoLatentShape, AudioLatentShape, LatentState, RMS norm, velocity/denoised conversion
- Depends on: MLX
- Used by: All other layers reference these types

## Data Flow

**Text-to-Video Generation (Primary Flow):**

1. **Initialization** (`scripts/generate.py`):
   - Load transformer weights via `loader.load_transformer_weights()` → `LTXModel` or `LTXAVModel`
   - Load text encoder (Gemma 3) weights → text encoder model
   - Load VAE decoder weights → `SimpleVideoDecoder`
   - Create components (patchifier, scheduler, guider, noiser, diffusion_step)
   - Instantiate pipeline (e.g., `OneStagePipeline`, `TwoStagePipeline`)

2. **Encode Text**:
   - Prompt → text encoder → `text_context` (B, S, C_ctx) and `text_mask` (B, S)
   - Unconditional prompt (empty string) → `null_context` and `null_mask` for CFG

3. **Noise Latent**:
   - Initialize noise in latent space: `(batch, channels, frames, height, width)` = 128D x 8 channels x 121 frames
   - Get sigma schedule from scheduler (e.g., `LTX2Scheduler`)
   - Create `LatentState` with: latent (noise), denoise_mask (1.0 for full denoising), positions (grid), clean_latent (same as latent)

4. **Diffusion Loop** (Euler stepping):
   - For each step in sigma schedule:
     - **Patchify**: Convert 5D latent `(B, C, T, H, W)` → 3D `(B, T_patches, C_proj)` via `VideoLatentPatchifier`
     - **Prepare modality**: Create `Modality` dataclass with patchified latent, text context, timestep embedding, positions
     - **Model forward pass** (with CFG batching optimization):
       - Stack cond and uncond modalities along batch dimension (2x speedup)
       - `transformer(modality)` → denoised prediction
       - Split and separate cond/uncond outputs
     - **Guidance**: Apply guider (CFG, STG, or APG) to blend cond/uncond outputs
     - **Diffusion step**: Use `EulerDiffusionStep` to compute velocity and step toward less noise
     - **Update latent**: New sample = sample + velocity * dt

5. **Post-Processing**:
   - Decode latent → video pixels via `SimpleVideoDecoder`
   - Optional: Upsample with spatial/temporal upscalers (`SpatialUpscaler`, `TemporalUpscaler`)
   - Optional: Apply LoRA refinement (two-stage) with distilled LoRA weights

6. **Audio Generation** (if enabled):
   - Run `LTXAVModel` (audio-video transformer) in parallel
   - Generate audio latent in similar diffusion loop
   - Decode audio latent → mel-spectrogram
   - Vocoder → waveform

**State Management:**

- **PipelineState** (text-to-video): `(latent, sigmas, step, text_context, text_mask)` — mutable during iteration
- **LatentState** (general): `(latent, denoise_mask, positions, clean_latent)` — immutable frozen dataclass, used for conditioning
- **Modality**: Encapsulates a single modality (video or audio) at one diffusion step with all its inputs

## Key Abstractions

**Pipeline Base** (`TextToVideoPipeline` / `OneStagePipeline` / `TwoStagePipeline`):
- Purpose: Encapsulate complete generation workflow (text → noise → denoise → video)
- Examples: `LTX_2_MLX/pipelines/text_to_video.py`, `LTX_2_MLX/pipelines/one_stage.py`, `LTX_2_MLX/pipelines/two_stage.py`
- Pattern: Factory functions like `create_pipeline()`, `create_one_stage_pipeline()`, `create_two_stage_pipeline()` return initialized pipeline with all components

**GuiderProtocol** (CFG, STG, LtxAPG):
- Purpose: Pluggable guidance strategies for controlling denoising direction
- Examples: `CFGGuider`, `STGGuider`, `LtxAPGGuider` in `LTX_2_MLX/components/guiders.py`
- Pattern: All implement `delta(cond, uncond) → guidance_delta` and `guide(cond, uncond) → final_output`

**DiffusionStepProtocol** (Euler, Heun):
- Purpose: Interchangeable ODE solvers for moving between noise levels
- Examples: `EulerDiffusionStep`, `HeunDiffusionStep` in `LTX_2_MLX/components/diffusion_steps.py`
- Pattern: Both implement `step(sample, denoised_sample, sigmas, step_index) → next_sample`

**Patchifier** (VideoLatentPatchifier, AudioPatchifier):
- Purpose: Convert dense latent tensors to token sequences for transformer input
- Examples: `VideoLatentPatchifier` in `LTX_2_MLX/components/patchifiers.py`
- Pattern: Project 5D latent → 3D tokens, handle spatial/temporal structure with position embeddings

**VAE Encoder/Decoder**:
- Purpose: Compress/decompress video between pixel space and latent space
- Examples: `SimpleVideoEncoder`, `SimpleVideoDecoder` in `LTX_2_MLX/model/video_vae/`
- Pattern: Encoder (video → latent), Decoder (latent → video); also `AudioEncoder`, `AudioDecoder` for audio

**LoRAConfig & Weight Fusion**:
- Purpose: Manage low-rank adaptation weights for efficient fine-tuning (two-stage distilled LoRA)
- Examples: `LTX_2_MLX/loader/lora_loader.py`
- Pattern: Define target modules, load LoRA weights, compute deltas, fuse into base weights

**ConditioningItem & VideoLatentTools**:
- Purpose: Encode frame-level control (image conditioning, keyframes, latent masks)
- Examples: `LTX_2_MLX/conditioning/latent.py`, `LTX_2_MLX/conditioning/tools.py`
- Pattern: Tools convert frames to latent indices, apply masks, blend conditioned latents into generation

## Entry Points

**`scripts/generate.py`:**
- Location: `scripts/generate.py` (98KB main entry point)
- Triggers: Command-line invocation: `python scripts/generate.py "prompt" [--pipeline {distilled|text-to-video|one-stage|two-stage}] [options]`
- Responsibilities:
  - Parse CLI arguments (height, width, frames, steps, cfg_scale, seed, etc.)
  - Load all weights (transformer, text encoder, VAE, upscalers if needed)
  - Create selected pipeline
  - Call pipeline.generate() with config
  - Save output video/audio to disk

**Pipeline Factory Functions:**
- Location: Each pipeline module exports `create_*_pipeline()` function
- Triggers: Called by generate.py or tests
- Responsibilities: Instantiate all components and return initialized pipeline

**Weight Loader Functions:**
- Location: `LTX_2_MLX/loader/weight_converter.py`
- Triggers: Called during pipeline initialization
- Responsibilities: Convert PyTorch keys to MLX naming, load safetensors, transpose if needed

## Error Handling

**Strategy:** Exceptions and validation at component boundaries

**Patterns:**

1. **Configuration Validation**:
   - Config dataclasses have `__post_init__` methods (e.g., `GenerationConfig` validates `num_frames % 8 == 1`)
   - `check_config_value()` in `core_utils.py` validates loaded weights match expected config

2. **Weight Loading**:
   - `load_safetensors()` catches file errors and tensor conversion failures
   - `load_transformer_weights()` checks weight shapes and counts against model config

3. **Image/Video Loading**:
   - `load_image_tensor()` in `common.py` validates file exists, format is RGB/RGBA/L, can be resized
   - Raises specific exceptions: `FileNotFoundError`, `ValueError` for format/shape mismatches

4. **Numerical Stability**:
   - Diffusion steps compute in float32 then cast back to avoid precision loss
   - `to_velocity()` checks sigma != 0, raises ValueError
   - RMS norm uses `eps=1e-6` for stability

5. **Conditioning Errors**:
   - `ConditioningError` in `conditioning/latent.py` for invalid frame indices or mask shapes

## Cross-Cutting Concerns

**Logging:**
- Approach: No centralized logger; uses `tqdm` for progress bars (optional import, fallback to simple iteration)
- Pattern: `progress_bar()` wrapper in generate.py handles missing tqdm

**Validation:**
- Approach: Type hints + dataclass validation + runtime checks
- Pattern: Dataclass `__post_init__` for config validation, shape assertions in model forward passes

**Authentication:**
- Approach: Not applicable (no external APIs)

**Performance Optimization:**
- Batched CFG forward pass (combine cond+uncond into single batch) for 2x speedup in `generate.py`
- Patchified representation reduces transformer sequence length
- Tiling for VAE decoding handles memory constraints (optional `TilingConfig`)
- FP16 precision option (default) reduces memory ~50%
- LoRA fusion avoids loading full-size weights for refinement stage

**Data Type Management:**
- Default: float32 for stability
- Optional: float16 (FP16) for memory efficiency, bfloat16 for experiments
- Specified in pipeline config: `OneStageCFGConfig.dtype`, `TwoStageCFGConfig.dtype`

**Batch Processing:**
- Single batch (B=1) typical for local generation
- Components support variable batch size via shape inference
- CFG uses batched optimization: [cond, uncond] concatenated, single forward pass, split results

---

*Architecture analysis: 2026-01-23*
