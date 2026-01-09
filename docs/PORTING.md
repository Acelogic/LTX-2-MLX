# Porting LTX-2 to MLX

Status document for the LTX-2 19B video generation model port to Apple Silicon MLX.

## Current State

### What Works

- **Transformer**: Full 48-layer DiT loads and runs inference
- **Video VAE Decoder**: Decodes latents to video with timestep conditioning
- **Video VAE Encoder**: Encodes video to latents
- **Audio VAE Decoder**: Decodes audio latents to mel spectrograms
- **Vocoder (HiFi-GAN)**: Converts mel spectrograms to 24kHz audio waveforms
- **Weight Loading**: All 4094 tensors load from safetensors
- **Text Encoding**: Native Gemma 3 encoder (12B parameters)
- **CFG (Classifier-Free Guidance)**: Implemented with configurable scale (default 3.0)
- **FP16 Computation**: `--fp16` flag reduces memory ~50% during inference
- **Memory Optimization**: Intermediate tensor cleanup every 8 transformer layers

### What's In Progress

- **FP8 Weight Loading**: Support for 27GB quantized weights
- **Spatial Upscaler**: 2x resolution upscaling (256→512)
- **Temporal Upscaler**: 2x framerate interpolation
- **AudioVideo Mode**: Full audio-video generation with cross-modal attention

---

## Quick Start

```bash
# Basic video generation
python scripts/generate.py "A cat walking in a garden" \
    --height 256 --width 256 --frames 17 \
    --output output.mp4

# With FP16 for lower memory usage
python scripts/generate.py "A cat walking in a garden" \
    --height 256 --width 256 --frames 17 \
    --fp16 --output output.mp4
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

### 3. FP16 Computation

Added compute dtype support to reduce memory during inference:

```python
# In LTXModel.__init__
self.compute_dtype = compute_dtype  # mx.float16 or mx.float32

# In forward pass
if self.compute_dtype != mx.float32:
    x = x.astype(self.compute_dtype)
    # ... computation ...
    output = output.astype(mx.float32)  # Cast back for stability
```

### 4. Audio Components

**Audio VAE Decoder** (`LTX_2_MLX/model/audio_vae/decoder.py`):
- Input: Latent `(B, 8, T, 64)` - 8 channels, mel_bins=64
- Output: Mel spectrogram `(B, 2, T*4, 64)` - stereo
- Architecture: CausalConv2d, SimpleResBlock2d, Upsample2d

**Vocoder** (`LTX_2_MLX/model/audio_vae/vocoder.py`):
- Input: Mel spectrogram `(B, 2, T, 64)`
- Output: Audio waveform `(B, 2, T*240)` at 24kHz
- Architecture: HiFi-GAN with ConvTranspose1d upsampling

---

## MLX vs PyTorch Comparison

| Component | MLX Implementation | PyTorch LTX-2 |
|-----------|-------------------|---------------|
| **Text Encoder** | Native MLX Gemma 3 12B | PyTorch Gemma 3 12B |
| **Compute Dtype** | FP32 or FP16 (`--fp16`) | BF16 |
| **Tokenizer Padding** | RIGHT padding (required) | LEFT padding |
| **Denoising Schedule** | Distilled 3-7 step | Dynamic 25-50 step |
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
│   │   ├── encoder.py        # VAE encoder
│   │   └── ops.py            # patchify/unpatchify operations
│   └── audio_vae/
│       ├── decoder.py        # Audio VAE decoder
│       └── vocoder.py        # HiFi-GAN vocoder
├── inference/
│   └── scheduler.py          # Sigma schedules
├── loader/
│   └── weight_converter.py   # Weight loading utilities
└── components.py             # Patchifier, CFG guider

scripts/
├── generate.py               # Main generation script
└── download_gemma.py         # Gemma weights download
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

## References

- [LTX-2 GitHub](https://github.com/Lightricks/LTX-2)
- [LTX-2 Paper](https://arxiv.org/abs/2501.00103)
- [HuggingFace Model](https://huggingface.co/Lightricks/LTX-2)
- [MLX Documentation](https://ml-explore.github.io/mlx/)
