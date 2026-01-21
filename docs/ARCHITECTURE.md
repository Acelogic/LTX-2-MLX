# LTX-2 MLX Architecture

This document describes the architecture of the LTX-2 model and its MLX implementation.

## Overview

LTX-2 is a **19-billion parameter Diffusion Transformer (DiT)** that generates synchronized video and audio from text prompts. This MLX port enables native Apple Silicon inference.

```
Text Prompt
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Gemma 3 (12B)                                              │
│  Text encoder with 48 layers, 3840-dim hidden states        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Text Encoder Projection                                    │
│  Projects 3840-dim → 4096-dim for video context             │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  48-Layer Transformer (19B parameters)                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Video Stream (14B)                                 │    │
│  │  • 32 attention heads × 128 dim = 4096 hidden       │    │
│  │  • 3D RoPE positional encoding (x, y, t)            │    │
│  │  • Cross-attention to text embeddings               │    │
│  └─────────────────────────────────────────────────────┘    │
│                          ↕                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Audio Stream (5B) - optional                       │    │
│  │  • 16 attention heads × 128 dim = 2048 hidden       │    │
│  │  • 1D RoPE positional encoding (temporal)           │    │
│  │  • Bidirectional cross-attention with video         │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Video VAE Decoder                                          │
│  • 128 latent channels → 3 RGB channels                     │
│  • 1:192 compression (32× spatial, 8× temporal)             │
│  • Timestep conditioning for final denoising                │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
  Video Output (up to 768×1024, 24fps)
```

## MLX Package Structure

```
LTX_2_MLX/
├── model/
│   ├── transformer/
│   │   ├── model.py          # LTXModel (velocity) and X0Model (denoised)
│   │   ├── transformer.py    # LTXTransformer with 48 blocks
│   │   ├── attention.py      # Multi-head attention with RoPE
│   │   └── rope.py           # 3D rotary position embeddings
│   │
│   ├── video_vae/
│   │   ├── simple_decoder.py # VAE decoder with timestep conditioning
│   │   └── ops.py            # Depth-to-space, pixel shuffle operations
│   │
│   ├── audio_vae/
│   │   ├── decoder.py        # Audio latent → mel spectrogram
│   │   └── encoder.py        # Audio encoding (experimental)
│   │
│   ├── text_encoder/
│   │   ├── gemma3.py         # Native Gemma 3 12B implementation
│   │   └── encoder.py        # Text encoder projection layers
│   │
│   └── upscaler/
│       ├── spatial.py        # 2× spatial resolution upscaler
│       └── temporal.py       # 2× framerate interpolation
│
├── components/
│   ├── schedulers.py         # Sigma schedules (distilled, LTX2Scheduler)
│   ├── patchifiers.py        # Latent ↔ sequence conversion
│   ├── noisers.py            # Gaussian noise addition
│   └── guiders.py            # CFG implementation
│
├── conditioning/
│   └── tools.py              # Latent conditioning utilities
│
├── pipelines/
│   ├── text_to_video.py      # Basic text-to-video
│   ├── distilled.py          # Fast 8-step generation
│   ├── one_stage.py          # Single-stage CFG
│   ├── two_stage.py          # Two-stage with upscaling
│   ├── ic_lora.py            # Image conditioning LoRA
│   └── keyframe_interpolation.py
│
├── loader/
│   └── weight_converter.py   # PyTorch → MLX weight conversion
│
└── types.py                  # VideoLatentShape, SpatioTemporalScaleFactors
```

## Key Components

### Transformer (19B parameters)

The transformer uses a **velocity prediction** formulation:
- Raw model output is velocity `v`
- Denoised prediction: `x0 = latent - sigma * velocity`
- The `X0Model` wrapper handles this conversion

Each of the 48 transformer blocks contains:
- Self-attention with 3D RoPE (video stream)
- Cross-attention to text embeddings
- Feed-forward network with GELU activation
- AdaLN (Adaptive Layer Norm) conditioning on timestep

### Video VAE Decoder

The VAE decoder converts 128-channel latents to RGB video:
- **Compression ratio**: 32× spatial, 8× temporal
- **Input**: `(B, 128, T, H, W)` latents
- **Output**: `(B, 3, T×8, H×32, W×32)` pixels

Key feature: **Timestep conditioning** - the decoder performs a final denoising step during decode, using learned scale/shift tables indexed by the current sigma value.

### Text Encoder

Two-stage text encoding:
1. **Gemma 3 (12B)**: Extracts 3840-dim hidden states from all 48 layers
2. **Projection**: Linear projection from 3840 → 4096 dimensions

The projection includes:
- Multi-layer hidden state aggregation
- Layer normalization
- Caption projection to transformer hidden dimension

### Patchifier

Converts between spatial latent format and sequence format:
- **Patchify**: `(B, C, T, H, W)` → `(B, T×H×W, C)` for transformer
- **Unpatchify**: `(B, T×H×W, C)` → `(B, C, T, H, W)` for VAE

### 3D RoPE (Rotary Position Embeddings)

Position encoding for video tokens:
- Separate frequency components for (x, y, t) dimensions
- Uses "SPLIT" rope type: cos/sin applied to separate halves of features
- Precomputed based on latent grid coordinates

## Inference Flow

### Distilled Pipeline (8 steps)

```python
# 1. Text encoding
text_encoding = text_encoder.encode(prompt)  # (1, 1024, 4096)

# 2. Initialize noise
latent = mx.random.normal(shape=(1, 128, T, H, W))

# 3. Denoising loop (8 steps)
sigmas = [1.0, 0.994, 0.988, 0.981, 0.975, 0.909, 0.725, 0.422, 0.0]

for i in range(8):
    # Patchify latent for transformer
    latent_seq = patchifier.patchify(latent)

    # Create modality with positions
    modality = Modality(
        latent=latent_seq,
        context=text_encoding,
        timesteps=mx.array([sigmas[i]]),
        positions=pixel_coords,
    )

    # Predict denoised (x0)
    x0_seq = x0_model(modality)

    # Unpatchify
    denoised = patchifier.unpatchify(x0_seq)

    # Euler step
    velocity = (latent - denoised) / sigmas[i]
    latent = latent + velocity * (sigmas[i+1] - sigmas[i])

# 4. VAE decode
video = vae_decoder.decode(latent)  # (T, H, W, 3) uint8
```

## Memory Requirements

| Configuration | RAM Usage |
|---------------|-----------|
| Gemma 3 text encoder | ~12 GB |
| Transformer (FP16) | ~20 GB |
| VAE decoder | ~2 GB |
| **Total (sequential)** | **~25 GB** |

The implementation loads models sequentially to fit within unified memory:
1. Load Gemma 3, encode text, unload
2. Load transformer, denoise, unload
3. Load VAE, decode video

## Precision

- **Weights**: BFloat16 (loaded from safetensors)
- **Computation**: BFloat16 for transformer, Float32 for VAE
- **Activations**: BFloat16 to reduce memory

## References

- [LTX-2 Technical Report](LTX_2_Technical_Report_compressed.pdf) - Official Lightricks paper
- [LTX-2 PyTorch](https://github.com/Lightricks/LTX-2) - Reference implementation
- [MLX Documentation](https://ml-explore.github.io/mlx/) - Apple MLX framework
