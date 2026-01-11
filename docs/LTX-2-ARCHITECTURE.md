# LTX-2 Architecture Overview

LTX-2 is a **19-billion parameter DiT (Diffusion Transformer)** that generates synchronized video and audio from text/image prompts. It's the first DiT-based audio-video foundation model combining all core capabilities in one model.

## High-Level Architecture

```
Text Prompt → Gemma 3 (12B) → [Video Context, Audio Context]
                                        ↓
                    48-Layer Asymmetric Dual-Stream Transformer
                    ┌─────────────────────────────────────────┐
                    │  Video Stream (14B) ←→ Audio Stream (5B) │
                    │  Cross-attention for synchronization     │
                    └─────────────────────────────────────────┘
                                        ↓
                    Video VAE Decoder → Pixels (512×768, 25fps)
                    Audio VAE Decoder → Mel → Vocoder → 24kHz audio
```

---

## Package Structure

The repository is organized as a monorepo with three packages:

```
LTX-2/
├── packages/
│   ├── ltx-core/           # Core model implementations & inference primitives
│   ├── ltx-pipelines/      # Ready-to-use inference pipelines
│   └── ltx-trainer/        # Training & fine-tuning tools
└── pyproject.toml          # Workspace configuration
```

### Package Dependencies

- `ltx-core` provides building blocks (models, components, utilities)
- `ltx-pipelines` and `ltx-trainer` consume `ltx-core`
- `ltx-pipelines` and `ltx-trainer` are independent of each other

---

## LTX-Core: Foundation Layer

Location: `packages/ltx-core/src/ltx_core/`

### Models

| Component | Location | Purpose |
|-----------|----------|---------|
| Transformer | `model/transformer/` | 48-layer asymmetric dual-stream architecture |
| Video VAE | `model/video_vae/` | Encode/decode video pixels to/from latents |
| Audio VAE | `model/audio_vae/` | Encode/decode audio spectrograms to/from latents |
| Vocoder | `model/audio_vae/vocoder.py` | Convert mel spectrograms → audio waveforms (HiFi-GAN) |
| Text Encoder | `text_encoders/gemma/` | Gemma 3-based multilingual encoder |
| Spatial Upsampler | `model/upsampler/` | Upsample latent representations |

### Diffusion Components (`components/`)

| Component | File | Purpose |
|-----------|------|---------|
| Schedulers | `schedulers.py` | Generate noise schedules (LTX2Scheduler, LinearQuadratic, Beta) |
| Guiders | `guiders.py` | Guidance strategies (CFG, STG, APG) |
| Noisers | `noisers.py` | Add noise to latents according to diffusion schedule |
| Diffusion Steps | `diffusion_steps.py` | Update latents (Euler stepper) |
| Patchifiers | `patchifiers.py` | Convert between spatial `[B, C, F, H, W]` and sequence `[B, seq_len, dim]` |

### Conditioning & Control

- **ConditioningItem**: Abstract base for image/video/keyframe conditioning
- **VideoConditionByLatentIndex**: Replace latents at specific frames (strong control)
- **VideoConditionByKeyframeIndex**: Add latents as guiding signal (smooth interpolation)
- **Perturbations**: Selectively skip attention operations for fine-grained control

---

## LTX-Pipelines: Inference Pipelines

Location: `packages/ltx-pipelines/src/ltx_pipelines/`

### Available Pipelines

| Pipeline | Stages | CFG | Upsampling | Best For |
|----------|--------|-----|------------|----------|
| **TI2VidTwoStagesPipeline** | 2 | Yes | 2x | Production video (recommended) |
| **TI2VidOneStagePipeline** | 1 | Yes | No | Quick prototyping |
| **DistilledPipeline** | 2 | No | 2x | Batch processing (fastest) |
| **ICLoraPipeline** | 2 | Yes | 2x | Video-to-video control |
| **KeyframeInterpolationPipeline** | 2 | Yes | 2x | Animation/interpolation |

### Pipeline Details

#### TI2VidTwoStagesPipeline (Recommended)
- Two-stage: Low-res generation + 2x upsampling
- CFG guidance in stage 1
- Distilled LoRA refinement in stage 2
- Supports image conditioning
- File: `ti2vid_two_stages.py`

#### DistilledPipeline
- 8 predefined sigmas (no CFG needed)
- Fastest inference (8 steps stage 1, 4 steps stage 2)
- File: `distilled.py`

#### ICLoraPipeline
- Video-to-video transformations using control signals (depth, pose, canny edges)
- Requires IC-LoRA trained models
- File: `ic_lora.py`

#### KeyframeInterpolationPipeline
- Smooth transitions between keyframe images
- Uses additive conditioning for smooth motion
- File: `keyframe_interpolation.py`

---

## LTX-Trainer: Training Tools

Location: `packages/ltx-trainer/src/ltx_trainer/`

### Training Strategies

1. **TextToVideoStrategy**: Standard text-to-video training
2. **VideoToVideoStrategy**: IC-LoRA control training (depth/pose maps + target video)

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| Config | `config.py` | Pydantic-based configuration |
| Trainer | `trainer.py` | Accelerate-based distributed training |
| Dataset | `datasets.py` | PrecomputedDataset for VAE latents |
| Validation | `validation_sampler.py` | Inference-based validation |

### Training Scripts

- `train.py` - Main training script
- `process_dataset.py` - Preprocess datasets
- `process_videos.py` - Encode videos to VAE latents
- `process_captions.py` - Compute text embeddings
- `caption_videos.py` - Automatic captioning
- `inference.py` - Inference with trained models

---

## Model Architecture: 48-Layer Asymmetric Dual-Stream Transformer

### Data Flow

```
Text Input → Gemma 3 Encoder → [Video Context, Audio Context]
                                         ↓
Video Pixels → Video VAE Encoder → Video Latents [B, 128, F', H/32, W/32]
Audio Waveform → Audio VAE Encoder → Audio Latents [B, 8, T/4, 16]
                                         ↓
         ┌─────────────────────────────────────────────────┐
         │     Noise + Add to Latents (Diffusion)           │
         ├─────────────────────────────────────────────────┤
         │  LTX-2 Transformer (48 layers, Asymmetric)       │
         │                                                   │
         │  Video Stream (14B params):  [B, seq_len, 4096]  │
         │  - Self-Attention (3D RoPE: x,y,t)               │
         │  - Text Cross-Attention                          │
         │  - Audio↔Video Cross-Attention (1D temporal RoPE)│
         │  - Feed-Forward                                  │
         │                                                   │
         │  Audio Stream (5B params):   [B, seq_len, 2048]  │
         │  - Self-Attention (1D RoPE: temporal)            │
         │  - Text Cross-Attention                          │
         │  - Video↔Audio Cross-Attention (1D temporal RoPE)│
         │  - Feed-Forward                                  │
         │                                                   │
         │  Cross-Modality AdaLN for synchronization        │
         └─────────────────────────────────────────────────┘
                          ↓ (Iterative Denoising)
         ┌─────────────────────────────────────────────────┐
         │         VAE Decoding                             │
         │                                                   │
         │  Video Latents → Video VAE Decoder → Pixels      │
         │  Audio Latents → Audio VAE Decoder → Mel Spec    │
         │  Mel Spectrogram → Vocoder → Waveform (24 kHz)   │
         └─────────────────────────────────────────────────┘
```

### Transformer Block (48 identical blocks)

Each block performs 4 operations on both streams:

```python
# Video Stream (14B):
video = RMSNorm(video) + AdaLN(timestep) → Self-Attention(video)
      → RMSNorm(video) → Text Cross-Attention(video, text_context)
      → RMSNorm(video) + AdaLN(timestep) → Audio↔Video Cross-Attn
      → RMSNorm(video) + AdaLN(timestep) → Feed-Forward(video)

# Audio Stream (5B):
audio = RMSNorm(audio) + AdaLN(timestep) → Self-Attention(audio)
      → RMSNorm(audio) → Text Cross-Attention(audio, text_context)
      → RMSNorm(audio) + AdaLN(timestep) → Video↔Audio Cross-Attn
      → RMSNorm(audio) + AdaLN(timestep) → Feed-Forward(audio)
```

### Key Technical Details

- **Asymmetric Streams**: Video=14B params (complex spatiotemporal), Audio=5B params (1D temporal)
- **3D RoPE (Video)**: Rotary Position Embeddings for x, y, t dimensions
- **1D RoPE (Audio)**: Temporal position embeddings only
- **Cross-Modal RoPE**: Audio↔Video cross-attention uses 1D temporal RoPE for frame-level sync
- **AdaLN**: Adaptive Layer Normalization gates conditioned on timesteps

---

## Inference Flow (TI2VidTwoStagesPipeline)

### Stage 1: Low-Resolution Generation

1. **Text Encoding**
   - Prompt + System Prompt → Gemma 3 (12B)
   - Multi-layer feature extraction from all decoder layers
   - Separate connectors → Video Context `[B, T, 4096]`, Audio Context `[B, T, 2048]`

2. **Initialization**
   - Initialize noise latents: `[B, 128, F', H/32, W/32]`
   - F' = 1 + (F-1)/8 (e.g., 121 frames → 16 latent frames)
   - Resolution: 1/2 of target (e.g., 256×384 for 512×768)

3. **Patchification**
   - Convert spatial `[B, 128, F', H', W']` → sequence `[B, T, 4096]`

4. **Denoising Loop (40 steps)**
   - LTX2Scheduler generates adaptive sigmas
   - CFG guidance: `pred = pred_cond + scale * (pred_cond - pred_uncond)`
   - Euler stepper updates latents
   - Optional image conditioning at specific frames

5. **Unpatchification**
   - Convert sequence back to spatial format

### Stage 2: Upsampling & Refinement

1. **Spatial Upsampling**
   - LatentUpsampler: 2x spatial via ResBlocks + pixel shuffle

2. **Distilled Refinement**
   - Load distilled LoRA weights
   - 8 predefined sigmas, 4 denoising steps
   - No CFG (distilled model already optimized)

### Decoding

- **Video VAE Decoder**: `[B, 128, F', H, W]` → `[B, 3, F, H, W]` pixels
- **Audio VAE Decoder**: `[B, 8, T/4, 16]` → mel spectrogram
- **Vocoder**: Mel → 24 kHz stereo waveform

---

## Text Encoder: Gemma 3 Integration

```
User Prompt + System Prompt → Gemma 3 (12B decoder-only LLM)
                              ↓
                   All layer outputs [B, T, hidden_dim, L]
                              ↓
                   Multi-layer Feature Extractor
                              ↓
            ┌─────────────────┬─────────────────┐
            ↓                 ↓
    Video Connector     Audio Connector
   (Bidirectional       (Bidirectional
    Transformer)        Transformer)
            ↓                 ↓
    Video Context      Audio Context
    [B, T, 4096]       [B, T, 2048]
```

### Key Features

- **Multilingual**: Supports multiple languages
- **Multi-Layer Extraction**: Uses features from ALL decoder layers
- **Separate Contexts**: Video and audio get different embeddings optimized for their modality
- **Learnable Registers**: "Thinking tokens" for contextual mixing

### Text Encoder Pipeline (MLX Implementation)

```
Gemma 3 Forward Pass → 49 Hidden States [B, T, 3840] each
                              ↓
Feature Extractor (CRITICAL - must use all 49 layers):
  Stack all layers → [B, T, 3840, 49]
  Per-layer normalization (norm_and_concat_padded_batch):
    - Compute masked mean per layer
    - Compute masked min/max per layer
    - Normalize: 8 * (x - mean) / range
  Concatenate → [B, T, 188160]
  Linear projection (aggregate_embed) → [B, T, 3840]
                              ↓
Embeddings Connector:
  Replace padding with learnable registers (128 tokens, tiled to fill padding)
  2x Transformer blocks with RoPE (30 heads, 128 dim/head)
  Final RMSNorm
                              ↓
Caption Projection (in transformer):
  Linear 3840 → 4096 (with SiLU)
  Linear 4096 → 4096
                              ↓
Cross-Attention Context [B, T, 4096]
```

**Important**: The feature extractor MUST use all 49 Gemma layers with proper normalization. Using only the final layer produces embeddings that don't match what the model was trained with, resulting in noise output.

---

## LoRA Support

### LoRA Types

| Type | Purpose | Used In |
|------|---------|---------|
| **Distilled LoRA** | Stage 2 refinement | All two-stage pipelines |
| **IC-LoRA** | Control signals (depth, pose, edges) | ICLoraPipeline |
| **Camera Control** | Motion control (dolly, jib, static) | Any pipeline |
| **Custom LoRA** | User-trained adapters | Via ltx-trainer |

### LoRA Loading Process

1. Load base model weights from checkpoint
2. Load LoRA state dicts from `.safetensors`
3. Compute delta: `ΔW = LoRA_B @ LoRA_A`
4. Apply with strength: `W_final = W_base + strength * ΔW`

---

## Upscaler Architecture

### Spatial Upscaler

```python
LatentUpsampler:
  - Input: [B, 128, F, H, W]
  - Initial Conv: 128 → 512 channels
  - Initial ResBlocks (4 blocks)
  - Spatial Upsampling:
    * Conv3d: 512 → 2048 channels
    * PixelShuffleND(2): 2x spatial upsampling
  - Post-upsampling ResBlocks (4 blocks)
  - Final Conv: 512 → 128 channels
  - Output: [B, 128, F, H*2, W*2]
```

### Temporal Upscaler

- Similar architecture but upsamples temporal dimension
- Supported but not yet used in current pipelines

---

## Technical Constraints

### Frame Requirements

**Formula**: `frames % 8 == 1`

- Valid: 1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 121
- Invalid: 24, 32, 48, 64, 100

**Reason**: Video VAE uses 8x temporal compression. Formula ensures integer latent frames:
- `F' = 1 + (F-1)/8`

### Resolution Requirements

- Width and height must be divisible by 32 (VAE spatial compression)
- Recommended: Powers of 2 or multiples of 32

### Shape Conventions

| Data | Shape | Notes |
|------|-------|-------|
| Video Pixels | `[B, 3, F, H, W]` | BGR channels |
| Video Latents | `[B, 128, F', H/32, W/32]` | F' = 1 + (F-1)/8 |
| Audio Latents | `[B, 8, T/4, 16]` | 4x temporal compression |
| Video Context | `[B, seq_len, 4096]` | From Gemma |
| Audio Context | `[B, seq_len, 2048]` | From Gemma |

---

## Memory & Optimization

### Memory Optimization

1. **FP8 Transformer**: Quantize to float8 with bfloat16 upcast
2. **Gradient Checkpointing**: Recompute activations (training)
3. **Memory Cleanup**: Delete models between stages + `torch.cuda.empty_cache()`
4. **Tiling**: Process large videos in chunks

### Performance Optimization

1. **Gradient Estimation**: Reduce steps from 40 to 20-30
2. **Attention Optimizations**: xFormers or Flash Attention 3
3. **Stage Separation**: Load/unload models between stages

---

## Key Files Reference

| Component | File | Purpose |
|-----------|------|---------|
| Transformer | `ltx-core/model/transformer/model.py` | 48-layer dual-stream |
| Video VAE | `ltx-core/model/video_vae/video_vae.py` | Encode/decode video |
| Audio VAE | `ltx-core/model/audio_vae/audio_vae.py` | Encode/decode audio |
| Text Encoder | `ltx-core/text_encoders/gemma/` | Gemma conditioning |
| Upscaler | `ltx-core/model/upsampler/model.py` | Spatial upsampling |
| Scheduler | `ltx-core/components/schedulers.py` | LTX2Scheduler |
| Guider | `ltx-core/components/guiders.py` | CFG guidance |
| Patchifier | `ltx-core/components/patchifiers.py` | Spatial ↔ Sequence |
| TI2VidTwoStages | `ltx-pipelines/ti2vid_two_stages.py` | Recommended pipeline |
| ICLora | `ltx-pipelines/ic_lora.py` | Video-to-video control |
| Trainer | `ltx-trainer/trainer.py` | Training orchestration |
| ModelLedger | `ltx-pipelines/utils/model_ledger.py` | Model coordinator |

---

## Complete Inference Flow Diagram

```
User Input (Prompt + Optional Image/Video)
         ↓
   Text Encoder (Gemma 3)
   Video/Audio Separate Contexts
         ↓
┌────────────────────────────────────────┐
│    Pipeline (TI2VidTwoStages)          │
├────────────────────────────────────────┤
│  Stage 1: Low-Resolution Generation    │
│  ├─ Scheduler → Sigmas [40 steps]      │
│  ├─ Noiser → Add noise to latents      │
│  ├─ Patchifier → Spatial → Sequence    │
│  ├─ Denoising Loop:                    │
│  │  ├─ Transformer (video + audio)     │
│  │  ├─ CFGGuider (blend predictions)   │
│  │  ├─ Euler Stepper (update latents)  │
│  │  └─ Image Conditioning (if any)     │
│  └─ Unpatchifier → Sequence → Spatial  │
│                                         │
│  Stage 2: Upsampling & Refinement      │
│  ├─ Upsampler → 2x spatial resolution  │
│  ├─ Scheduler → Distilled Sigmas [4]   │
│  ├─ Patchifier → Spatial → Sequence    │
│  ├─ Denoising Loop (distilled LoRA)    │
│  └─ Unpatchifier → Sequence → Spatial  │
│                                         │
│  Decoding:                              │
│  ├─ Video VAE Decoder → RGB pixels     │
│  ├─ Audio VAE Decoder → Mel spectrogram│
│  └─ Vocoder → Audio waveform (24 kHz)  │
└────────────────────────────────────────┘
         ↓
    Output (MP4 video + audio)
```
