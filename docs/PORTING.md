# Porting LTX-2 to MLX

Status document for the LTX-2 19B video generation model port to Apple Silicon MLX.

## Current State

### What Works

- **Transformer**: Full 48-layer DiT loads and runs inference
- **VAE Decoder**: Decodes latents to video with timestep conditioning
- **Weight Loading**: All 4094 tensors load from safetensors
- **Basic Generation**: End-to-end pipeline produces video output
- **Text Encoding**: Native Gemma 3 encoder integrated into generate.py
- **CFG (Classifier-Free Guidance)**: Implemented with configurable scale (default 3.0)
- **Brightness Fix**: Output now matches expected ~50% brightness

### What Doesn't Work Yet

- **Audio Generation**: Audio VAE and vocoder not implemented
- **Performance**: Not fully optimized for Apple Silicon (FP16 Gemma supported)

### Recently Implemented

- **Full Text Encoding Integration**: `generate.py` now uses real Gemma 3 encoding by default
- **Brightness Correction**: Fixed VAE decoder output bias (+0.31 correction)
- **FP16 Gemma Loading**: Reduces memory from ~24GB to ~12GB for text encoder

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

### Key Differences from LTX-Video

| Aspect | LTX-Video | LTX-2 |
|--------|-----------|-------|
| Parameters | ~2B | 19B |
| Modalities | Video only | Video + Audio |
| Text Encoder | T5/CLIP | Gemma 3 |
| timestep_scale | 1000 | 916 |
| Cross-attention | Text only | Text + Audio-Video |

---

## Porting Challenges

### 1. VAE Timestep Conditioning

**Problem**: Initial VAE decoder produced extremely dark output (~12% brightness).

**Discovery**: LTX-2's VAE decoder performs a **final denoising step** during decode. It requires:
- `timestep_scale_multiplier` (916.0)
- `last_time_embedder` for final normalization
- Per-block `time_embedder` in residual groups

**Solution**: Implemented full timestep conditioning pipeline:
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

**Result**: Brightness improved from 12% to 35%.

### 2. Scale/Shift Ordering

**Problem**: With timestep conditioning, output was still wrong - negative scale values.

**Discovery**: LTX-2 uses `(shift, scale)` ordering, not `(scale, shift)`:
```python
# LTX-2 convention
shift, scale = table.unbind(dim=1)  # row 0 = shift, row 1 = scale
```

**Solution**: Fixed all scale_shift_table accesses to use correct row ordering.

### 3. Conv3d Implementation

**Problem**: MLX doesn't have native Conv3d.

**Solution**: Implemented Conv3d as iterated Conv2d over temporal kernel positions:
```python
for kt in range(kernel_t):
    w_2d = weight[:, :, kt, :, :]  # Extract 2D kernel slice
    # Apply conv2d to corresponding temporal slice
    # Accumulate results
```

Weights stored in PyTorch format `(out, in, D, H, W)`, transposed on-the-fly.

### 4. Depth-to-Space Upsampling

**Problem**: LTX-2 uses 3D pixel shuffle for upsampling.

**Solution**: Implemented as reshape + transpose:
```python
# (B, C*8, T, H, W) → (B, C, T*2, H*2, W*2)
x = x.reshape(B, C, 2, 2, 2, T, H, W)
x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)
x = x.reshape(B, C, T*2, H*2, W*2)
```

With causal trimming for temporal consistency.

---

## Remaining Challenges

### 1. Audio Generation

Not implemented:
- Audio VAE encoder/decoder (8 latent channels)
- Vocoder (mel spectrogram → audio)
- Audio-video cross-attention in DiT

### 2. Performance Optimization

Current implementation is functional but not fully optimized:
- No Metal shader optimization
- No quantization (model is bfloat16)
- Sequential frame processing in Conv3d
- FP16 Gemma loading implemented (reduces ~12GB memory)

---

## MLX vs PyTorch Pipeline Comparison

Key differences between our MLX implementation and stock PyTorch LTX-2:

| Component | MLX Implementation | PyTorch LTX-2 |
|-----------|-------------------|---------------|
| **Text Encoder** | Native MLX Gemma 3 12B | PyTorch Gemma 3 12B |
| **Tokenizer Padding** | RIGHT padding (required) | LEFT padding |
| **Embedding Dim** | 3840 (pre-projection) | 3840 (pre-projection) |
| **Caption Projection** | 3840 → 4096 | 3840 → 4096 |
| **Denoising Schedule** | Distilled 7-step | Dynamic 30-step |
| **CFG Scale** | 3.0 (default) | 3.0-7.0 |
| **VAE Decode Timestep** | 0.05 | 0.05 |
| **VAE Brightness Fix** | +0.31 bias correction | Native output |
| **Memory (Gemma)** | ~12GB (FP16) | ~24GB (FP32) |

### Key Implementation Notes

1. **RIGHT Padding Required**: MLX implementation requires RIGHT padding for the tokenizer to avoid NaN values during attention. This differs from the PyTorch default of LEFT padding.

2. **Distilled Schedule**: We use the 7-step distilled sigma schedule (`[1.0, 0.99, 0.98, 0.93, 0.85, 0.50, 0.05]`) which produces good results with fewer steps than the 30-step dynamic schedule.

3. **Brightness Correction**: The MLX VAE decoder has a consistent -0.31 bias in output. We correct this by adding +0.31 before final conversion:
   ```python
   video = video + 0.31  # Bias correction
   video = mx.clip((video + 1) / 2, 0, 1) * 255
   ```

4. **CFG Higher Values**: CFG 7.0 produces better prompt differentiation than the default 3.0. Test different values for your use case.

---

## File Structure

```
LTX_2_MLX/
├── model/
│   ├── transformer.py        # DiT implementation
│   ├── text_encoder/
│   │   ├── gemma3.py         # Native MLX Gemma 3 model
│   │   ├── encoder.py        # Text encoder pipeline
│   │   ├── feature_extractor.py  # Multi-layer hidden state projection
│   │   └── connector.py      # 1D transformer connector
│   └── video_vae/
│       ├── simple_decoder.py # VAE decoder with timestep conditioning
│       └── ops.py            # patchify/unpatchify operations
├── components.py             # Sigma schedules, patchifier
├── loader.py                 # Weight loading utilities
└── types.py                  # Type definitions

scripts/
├── generate.py               # Main generation script with CFG
├── encode_prompt.py          # Gemma 3 text encoding pipeline
├── diagnose_decoder.py       # VAE debugging tool
└── download_gemma.py         # Gemma weights download helper
```

---

## Testing

### Basic Decode Test
```bash
python -c "
from LTX_2_MLX.model.video_vae.simple_decoder import *
decoder = SimpleVideoDecoder()
load_vae_decoder_weights(decoder, 'weights/ltx-2/ltx-2-19b-distilled.safetensors')

latent = mx.random.normal((1, 128, 5, 16, 16))
video = decode_latent(latent, decoder, timestep=0.05)
print(f'Output: {video.shape}, brightness: {video.mean()/255*100:.1f}%')
"
```

### Full Generation
```bash
python scripts/generate.py "A dog running in a park" \
    --height 256 --width 256 --frames 17 \
    --output output.mp4
```

---

## References

- [LTX-2 GitHub](https://github.com/Lightricks/LTX-2)
- [LTX-2 Paper](https://arxiv.org/abs/2601.03233)
- [HuggingFace Model](https://huggingface.co/Lightricks/LTX-2)
- [Diffusers Implementation](https://github.com/huggingface/diffusers/tree/main/src/diffusers/models)
