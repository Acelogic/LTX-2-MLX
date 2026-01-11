# Porting LTX-2 to MLX

Status document for the LTX-2 19B video generation model port to Apple Silicon MLX.

## Current State

### What Works

#### Core Models
- **Transformer**: Full 48-layer DiT (19B parameters) loads and runs inference
- **Video VAE Decoder**: Decodes latents to video with timestep conditioning
- **Video VAE Encoder**: Encodes video/images to latents
- **Audio VAE Decoder**: Decodes audio latents to mel spectrograms
- **Vocoder (HiFi-GAN)**: Converts mel spectrograms to 24kHz audio waveforms
- **Text Encoder**: Native Gemma 3 encoder (12B parameters)
- **Spatial Upscaler**: 2x resolution upscaling (e.g., 256×384 → 512×768)
- **Temporal Upscaler**: 2x framerate interpolation

#### Weight Loading
- **Standard Loading**: All 4094 tensors load from safetensors (BF16)
- **FP8 Loading**: Automatic dequantization from FP8 quantized checkpoints (27GB → ~12GB memory)
- **LoRA Support**: Load and fuse LoRA adapters (distilled, IC-LoRA, custom)

#### Pipelines
- **OneStagePipeline**: Single-stage CFG generation at target resolution
- **TwoStagePipeline**: Two-stage CFG + distilled refinement with upscaling
- **DistilledPipeline**: Fast 7+3 step generation without CFG
- **ICLoraPipeline**: Video-to-video control (depth, pose, canny edges)
- **KeyframeInterpolationPipeline**: Smooth transitions between keyframe images

#### Diffusion Components
- **LTX2Scheduler**: Adaptive sigma schedule generation
- **CFG (Classifier-Free Guidance)**: Configurable scale (default 3.0)
- **EulerDiffusionStep**: Euler stepper for denoising
- **GaussianNoiser**: Add noise to latents
- **VideoLatentPatchifier**: Spatial ↔ sequence conversion

#### Conditioning System
- **VideoConditionByLatentIndex**: Replace latents at specific frames (image-to-video)
- **VideoConditionByKeyframeIndex**: Additive conditioning for smooth interpolation
- **VideoLatentTools**: Utilities for conditioning management

#### Memory Optimization
- **FP16 Computation**: `--fp16` flag reduces memory ~50% during inference
- **Tiled VAE Decoding**: Process large videos in spatial/temporal tiles
- **Intermediate Cleanup**: Tensor cleanup every 8 transformer layers

---

## Quick Start

```bash
# Basic video generation (single stage)
python scripts/generate.py "A cat walking in a garden" \
    --height 480 --width 704 --frames 97 \
    --output output.mp4

# Two-stage with upscaling (higher quality)
python scripts/generate.py "A cat walking in a garden" \
    --height 480 --width 704 --frames 97 \
    --two-stage --output output.mp4

# With FP16 for lower memory usage
python scripts/generate.py "A cat walking in a garden" \
    --height 256 --width 384 --frames 33 \
    --fp16 --output output.mp4

# Image-to-video generation
python scripts/generate.py "A cat walking in a garden" \
    --image input.jpg --image-frame 0 \
    --height 480 --width 704 --frames 97 \
    --output output.mp4
```

---

## LTX-2 Architecture Overview

LTX-2 is a **joint audio-video generation model** - fundamentally different from video-only models.

```
┌─────────────────────────────────────────────────────────┐
│                    LTX-2 (19B params)                   │
├─────────────────────────────────────────────────────────┤
│  Video Stream (14B)          Audio Stream (5B)          │
│  ┌─────────────────┐        ┌─────────────────┐        │
│  │ Video DiT       │◄──────►│ Audio DiT       │        │
│  │ 3D RoPE (x,y,t) │  xattn │ 1D RoPE (t)     │        │
│  └────────┬────────┘        └────────┬────────┘        │
│           │                          │                  │
│  ┌────────▼────────┐        ┌────────▼────────┐        │
│  │ Video VAE       │        │ Audio VAE       │        │
│  │ 128ch, 1:192    │        │ 8ch → Vocoder   │        │
│  └─────────────────┘        └─────────────────┘        │
├─────────────────────────────────────────────────────────┤
│  Text Encoder: Gemma 3 (3840-dim embeddings)            │
└─────────────────────────────────────────────────────────┘
```

### Model Variants

| Model | Size | Steps | Quality |
|-------|------|-------|---------|
| `distilled` | 43GB (BF16) | 3-7 | Fast, good quality |
| `distilled-fp8` | 27GB (FP8) | 3-7 | Same as distilled, smaller file |
| `dev` | 43GB (BF16) | 25-50 | Highest quality |
| `dev-fp8` | 27GB (FP8) | 25-50 | Same as dev, smaller file |

### Upscalers

| Model | Size | Effect |
|-------|------|--------|
| `spatial-upscaler-x2` | 995MB | 2x resolution (256→512) |
| `temporal-upscaler-x2` | 262MB | 2x framerate (17→33 frames) |

---

## Available Pipelines

### OneStagePipeline
Single-stage generation at target resolution with CFG.

```python
from LTX_2_MLX.pipelines import OneStagePipeline, OneStageCFGConfig

pipeline = OneStagePipeline(transformer, video_encoder, video_decoder)
config = OneStageCFGConfig(height=480, width=704, num_frames=97)
video = pipeline(pos_encoding, pos_mask, neg_encoding, neg_mask, config)
```

### TwoStagePipeline
Two-stage generation: half-resolution with CFG → upsample → distilled refinement.

```python
from LTX_2_MLX.pipelines import TwoStagePipeline, TwoStageCFGConfig

pipeline = TwoStagePipeline(transformer, video_encoder, video_decoder, spatial_upscaler)
config = TwoStageCFGConfig(height=480, width=704, num_frames=97)
video = pipeline(pos_encoding, pos_mask, neg_encoding, neg_mask, config)
```

### DistilledPipeline
Fast generation without CFG using distilled sigma values.

```python
from LTX_2_MLX.pipelines import DistilledPipeline, DistilledConfig

pipeline = DistilledPipeline(transformer, video_encoder, video_decoder, spatial_upscaler)
config = DistilledConfig(height=480, width=704, num_frames=97)
video = pipeline(text_encoding, text_mask, config)
```

### ICLoraPipeline
Video-to-video control with depth maps, pose, or edge detection.

```python
from LTX_2_MLX.pipelines import ICLoraPipeline, ICLoraConfig, VideoCondition

pipeline = ICLoraPipeline(transformer, video_encoder, video_decoder, spatial_upscaler, base_weights, lora_configs)
config = ICLoraConfig(height=480, width=704, num_frames=97)
video_cond = [VideoCondition(video_path="depth_map.mp4", strength=0.95)]
video = pipeline(text_encoding, text_mask, config, video_conditioning=video_cond)
```

### KeyframeInterpolationPipeline
Generate smooth transitions between keyframe images.

```python
from LTX_2_MLX.pipelines import KeyframeInterpolationPipeline, KeyframeInterpolationConfig, Keyframe

pipeline = KeyframeInterpolationPipeline(transformer, video_encoder, video_decoder, spatial_upscaler)
keyframes = [
    Keyframe(image_path="start.jpg", frame_index=0),
    Keyframe(image_path="end.jpg", frame_index=96),
]
config = KeyframeInterpolationConfig(height=480, width=704, num_frames=97)
video = pipeline(text_encoding, text_mask, keyframes, config)
```

---

## Implementation Details

### 1. VAE Timestep Conditioning

LTX-2's VAE decoder performs a **final denoising step** during decode:

```python
# Scale timestep
scaled_t = timestep * self.timestep_scale_multiplier  # 0.05 * 916 = 45.8

# Create sinusoidal embedding
t_emb = get_timestep_embedding(scaled_t, 256)

# Project through MLP
time_emb = self.time_embedder(t_emb)

# Add to scale/shift table
ss_table = self.scale_shift_table + time_emb.reshape(B, 4, C)
```

### 2. Conv3d Implementation

MLX doesn't have native Conv3d. Implemented as iterated Conv2d:

```python
for kt in range(kernel_t):
    w_2d = weight[:, :, kt, :, :]  # Extract 2D kernel slice
    # Apply conv2d to corresponding temporal slice
    # Accumulate results
```

### 3. FP8 Weight Loading

FP8 quantized weights are automatically dequantized:

```python
from LTX_2_MLX.loader import load_fp8_weights

weights, num_fp8, num_regular = load_fp8_weights(
    "weights/ltx-2/ltx-2-19b-distilled-fp8.safetensors",
    target_dtype=mx.float16,
)
```

### 4. LoRA Fusion

LoRA weights are fused into base model weights:

```python
from LTX_2_MLX.loader import LoRAConfig, fuse_lora_into_weights

lora_configs = [
    LoRAConfig(path="distilled_lora.safetensors", strength=1.0),
]
fused_weights = fuse_lora_into_weights(base_weights, lora_configs)
```

### 5. Tiled Decoding

For memory-efficient decoding of high-resolution videos:

```python
from LTX_2_MLX.model.video_vae.tiling import TilingConfig, decode_tiled

config = TilingConfig(
    spatial=SpatialTilingConfig(tile_size_in_pixels=256, tile_overlap_in_pixels=32),
    temporal=TemporalTilingConfig(tile_size_in_frames=17, tile_overlap_in_frames=1),
)
video = decode_tiled(latent, video_decoder, config)
```

---

## MLX vs PyTorch Comparison

| Component | MLX Implementation | PyTorch LTX-2 |
|-----------|-------------------|---------------|
| **Text Encoder** | Native MLX Gemma 3 12B | PyTorch Gemma 3 12B |
| **Compute Dtype** | FP32 or FP16 (`--fp16`) | BF16 |
| **Tokenizer Padding** | RIGHT padding (required) | LEFT padding |
| **Denoising Schedule** | Distilled 3-7 step or LTX2 | Dynamic 25-50 step |
| **CFG Scale** | 3.0 (default) | 3.0-7.0 |
| **VAE Decode Timestep** | 0.05 | 0.05 |
| **Memory (Generation)** | ~25GB (FP16) | ~45GB+ |

---

## File Structure

```
LTX_2_MLX/
├── model/
│   ├── transformer/
│   │   ├── model.py          # LTXModel with FP16 support
│   │   ├── transformer.py    # BasicTransformerBlock
│   │   ├── attention.py      # Self/Cross attention
│   │   └── rope.py           # 3D RoPE positional embeddings
│   ├── text_encoder/
│   │   ├── gemma3.py         # Native MLX Gemma 3 model
│   │   ├── encoder.py        # Text encoder pipeline
│   │   └── feature_extractor.py  # Multi-layer projection
│   ├── video_vae/
│   │   ├── simple_decoder.py # VAE decoder with timestep conditioning
│   │   ├── simple_encoder.py # VAE encoder
│   │   ├── tiling.py         # Tiled encoding/decoding
│   │   └── ops.py            # patchify/unpatchify operations
│   ├── audio_vae/
│   │   ├── decoder.py        # Audio VAE decoder
│   │   ├── encoder.py        # Audio VAE encoder
│   │   └── vocoder.py        # HiFi-GAN vocoder
│   └── upscaler/
│       ├── spatial.py        # 2x spatial upscaler
│       └── temporal.py       # 2x temporal upscaler
├── conditioning/
│   ├── item.py               # ConditioningItem protocol
│   ├── latent.py             # VideoConditionByLatentIndex
│   ├── keyframe.py           # VideoConditionByKeyframeIndex
│   └── tools.py              # VideoLatentTools
├── pipelines/
│   ├── one_stage.py          # OneStagePipeline (CFG)
│   ├── two_stage.py          # TwoStagePipeline (CFG + upscale)
│   ├── distilled.py          # DistilledPipeline (fast, no CFG)
│   ├── ic_lora.py            # ICLoraPipeline (video control)
│   └── keyframe_interpolation.py  # KeyframeInterpolationPipeline
├── components/
│   ├── diffusion_steps.py    # EulerDiffusionStep
│   ├── guiders.py            # CFGGuider
│   └── schedulers.py         # LTX2Scheduler, sigma values
├── loader/
│   ├── weight_converter.py   # Weight loading utilities
│   ├── fp8_loader.py         # FP8 dequantization
│   └── lora_loader.py        # LoRA loading and fusion
└── types.py                  # LatentState, VideoLatentShape, etc.

scripts/
├── generate.py               # Main generation script
├── interpolate.py            # Keyframe interpolation script
├── download_gemma.py         # Gemma weights download
├── compare_text_embeddings.py # Compare MLX vs PyTorch text embeddings
└── debug_*.py                # Various debugging scripts (see TROUBLESHOOTING.md)
```

---

## Testing

### Run Test Suite
```bash
python -m pytest tests/ -v
```

### Test Audio Components
```python
from LTX_2_MLX.model.audio_vae import AudioDecoder, Vocoder
from LTX_2_MLX.model.audio_vae import load_audio_decoder_weights, load_vocoder_weights

# Load decoder
decoder = AudioDecoder()
load_audio_decoder_weights(decoder, 'weights/ltx-2/ltx-2-19b-distilled.safetensors')

# Test inference
latent = mx.random.normal((1, 8, 4, 64))
mel = decoder(latent)  # (1, 2, 13, 64)

# Load vocoder
vocoder = Vocoder()
load_vocoder_weights(vocoder, 'weights/ltx-2/ltx-2-19b-distilled.safetensors')

# Convert to audio
audio = vocoder(mel)  # (1, 2, 3120) at 24kHz
```

---

## What's In Progress

- **Full AudioVideo Mode**: Joint audio-video generation with synchronized cross-modal attention
- **Extended Testing**: Comprehensive parity tests against PyTorch reference

---

## Recent Fixes

### Text Encoding Pipeline (2026-01-11) - MAJOR FIX
Fixed the feature extractor to use the full text encoding pipeline matching PyTorch:

**Problem**: `extract_from_hidden_states` was only using Layer 48 directly, bypassing:
- Multi-layer aggregation (all 49 Gemma hidden states)
- Per-layer normalization (`norm_and_concat_padded_batch`)
- Learned linear projection (`aggregate_embed`)

**Fix**: Updated to stack all 49 layers, normalize, and project:
```python
# Stack all hidden states: [B, T, 3840, 49]
stacked = mx.stack(hidden_states, axis=-1)
# Normalize and concatenate: [B, T, 3840*49]
normed_concat = norm_and_concat_padded_batch(stacked, sequence_lengths, padding_side)
# Project: [B, T, 3840]
return self.aggregate_embed(normed_concat)
```

**Result**: Video generation now produces semantic content matching text prompts (palm trees, sunsets, grass, etc.).

### RoPE Type Fix (2026-01-10)
Changed RoPE (Rotary Position Embedding) type from INTERLEAVED to SPLIT to match PyTorch:
```python
# Before (incorrect)
rope_type = RoPEType.INTERLEAVED

# After (correct)
rope_type = RoPEType.SPLIT
```
This fixed transformer block outputs to match PyTorch exactly (correlation = 1.0).

### Sigma Schedule (2025-01-10)
Fixed distilled model sigma schedule to match official ComfyUI values:
```python
# Before (incorrect)
[1.0, 0.702, 0.432, 0.226, 0.095, 0.028, 0.003, 0.0]

# After (official)
[1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
```
The official schedule stays near 1.0 longer before dropping, which is important for the distilled model's training.

---

## References

- [LTX-2 GitHub](https://github.com/Lightricks/LTX-2)
- [LTX-2 Paper](https://arxiv.org/abs/2501.00103)
- [HuggingFace Model](https://huggingface.co/Lightricks/LTX-2)
- [MLX Documentation](https://ml-explore.github.io/mlx/)
