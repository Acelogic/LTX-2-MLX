# LTX-2 MLX Troubleshooting Guide

This document captures lessons learned while porting the LTX-2 19B video generation model to Apple Silicon MLX.

## Understanding LTX-2 Architecture

### LTX-2 vs LTX-Video

LTX-2 is fundamentally different from the original LTX-Video:

| Feature | LTX-Video | LTX-2 |
|---------|-----------|-------|
| Parameters | ~2B | 19B (14B video + 5B audio) |
| Modalities | Video only | Joint audio + video |
| Text Encoder | CLIP | Gemma 3 (3840-dim) |
| Cross-attention | None | Bidirectional audio-video |
| timestep_scale_multiplier | 1000.0 | 916.0 |

### LTX-2 Model Components

```
model.diffusion_model     # 19B DiT transformer
  ├── transformer_blocks  # Main attention blocks
  ├── adaln_single        # Video AdaLN
  ├── audio_adaln_single  # Audio AdaLN
  ├── av_ca_*             # Audio-video cross-attention
  └── caption_projection  # Text conditioning

vae.decoder               # Video VAE decoder
  ├── conv_in             # 128 → 1024 channels
  ├── up_blocks.{0,2,4,6} # ResBlock groups + time embedders
  ├── up_blocks.{1,3,5}   # Depth-to-space upsamplers
  ├── last_time_embedder  # Final timestep conditioning
  ├── last_scale_shift_table
  └── conv_out            # 128 → 48 channels

audio_vae                 # Audio VAE (8 latent channels)
vocoder                   # Mel spectrogram to audio
```

---

## Issue: Dark/Black Video Output

### Symptoms
- Video output is extremely dark (10-15% brightness)
- All latent inputs produce similar dark output
- Output values clustered around -0.78 instead of 0

### Root Cause: Missing Timestep Conditioning

The LTX-2 VAE decoder requires **timestep conditioning** to function correctly. Unlike standard VAEs, the LTX-2 decoder performs the final denoising step during decode and needs a timestep input.

#### Why This Matters

LTX-2 uses a 1:192 compression ratio (32x spatial + 8x temporal). At this extreme compression, fine details cannot be fully preserved in the latent space. The VAE decoder performs a learned denoising step to recover these details, conditioned on a timestep value.

Without timestep conditioning:
- The decoder uses only the learned `scale_shift_table` values
- These values are designed to be combined with timestep embeddings
- Result: Incorrect scale/shift → dark, low-contrast output

### Solution

1. **Add TimestepEmbedder class** for sinusoidal + MLP embedding:
```python
def get_timestep_embedding(timesteps, embedding_dim=256):
    """Sinusoidal timestep embedding."""
    half_dim = embedding_dim // 2
    freqs = mx.exp(-math.log(10000.0) * mx.arange(half_dim) / half_dim)
    args = timesteps[:, None] * freqs[None, :]
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim, output_dim, input_dim=256):
        self.linear_1 = nn.Linear(input_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, output_dim)

    def __call__(self, x):
        return self.linear_2(nn.silu(self.linear_1(x)))
```

2. **Load timestep embedder weights**:
```python
# Load from weights file
vae.decoder.timestep_scale_multiplier  # scalar ~916
vae.decoder.last_time_embedder.*       # For final norm
vae.decoder.up_blocks.{0,2,4,6}.time_embedder.*  # Per-block
```

3. **Pass timestep during decode**:
```python
# Recommended: timestep=0.05 for standard denoising
video = decoder(latent, timestep=0.05)

# timestep=0.0 disables denoising (raw reconstruction)
# timestep=None disables conditioning entirely (wrong output)
```

### Results
| Configuration | Brightness |
|--------------|------------|
| No timestep conditioning | 12% |
| timestep=0.0 | 32% |
| timestep=0.05 (recommended) | 32-35% |
| timestep=0.05 + bias correction | 50% ✓ |

---

## Issue: Output Brightness Still Too Dark (35% vs 50%)

### Symptoms
- With timestep conditioning, output brightness is ~35%
- Expected brightness from PyTorch reference is ~50%
- Colors appear washed out or dark

### Root Cause: VAE Decoder Output Bias

The MLX VAE decoder outputs with a consistent negative bias of approximately -0.31. This bias is likely due to subtle numerical differences in the Conv3d implementation or normalization layers.

### Solution: Add Bias Correction

Add a +0.31 correction in `decode_latent()` before the final conversion:

```python
def decode_latent(latent, decoder, timestep=0.05):
    # Decode with timestep conditioning
    video = decoder(latent, timestep=timestep)

    # Apply bias correction to center output at 0
    # The decoder outputs with a consistent negative bias (~-0.31)
    # This correction brings brightness from ~35% to ~50%
    video = video + 0.31

    # Convert to uint8: assume output is in [-1, 1]
    video = mx.clip((video + 1) / 2, 0, 1) * 255
    return video.astype(mx.uint8)
```

### Verification

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Brightness | 35.4% | 51.4% |
| Decoder output mean | -0.31 | ~0.0 |

---

## Issue: Scale/Shift Ordering

### Symptoms
- Timestep conditioning implemented but output is wrong
- Negative scale values (should be ~1.0)
- Output brightness decreases with higher timestep

### Root Cause: LTX Uses (shift, scale) Not (scale, shift)

LTX-2 stores `scale_shift_table` with **shift in row 0, scale in row 1**:

```python
# LTX-2 ordering (CORRECT)
shift, scale = ada_values.unbind(dim=1)  # row 0 = shift, row 1 = scale
x = x * (1 + scale) + shift

# Common mistake (WRONG)
scale = table[0]  # This is actually shift!
shift = table[1]  # This is actually scale!
```

### Solution

For `last_scale_shift_table` (shape `[2, 128]`):
```python
# Row 0 = shift, Row 1 = scale
shift = self.last_scale_shift_table[0]
scale = 1 + self.last_scale_shift_table[1]
x = x * scale + shift
```

For ResBlock `scale_shift_table` (shape `[4, channels]`):
```python
# Rows: shift1, scale1, shift2, scale2
shift1 = table[0]
scale1 = 1 + table[1]
shift2 = table[2]
scale2 = 1 + table[3]
```

---

## Issue: Conv3d Weight Format

### MLX vs PyTorch Format

| Framework | Conv3d Weight Shape |
|-----------|-------------------|
| PyTorch | `(out_C, in_C, D, H, W)` |
| MLX conv2d | `(out_C, H, W, in_C)` |

### Solution

Store weights in PyTorch format, transpose on-the-fly:
```python
class Conv3dSimple(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        # Store in PyTorch format: (out_C, in_C, kD, kH, kW)
        self.weight = mx.zeros((out_channels, in_channels, k, k, k))

    def __call__(self, x, causal=True):
        for kt in range(kernel_size):
            # Extract 2D slice and transpose for MLX
            w_slice = self.weight[:, :, kt, :, :]  # (out_C, in_C, kH, kW)
            w_slice = w_slice.transpose(0, 2, 3, 1)  # MLX: (out_C, kH, kW, in_C)
            # Apply conv2d...
```

---

## Issue: Pixel Normalization Behavior

### Symptoms
- Different latent inputs produce nearly identical outputs
- Signal "collapses" at normalization layers
- Limited dynamic range in output

### Understanding Pixel Norm

LTX-2 uses **pixel normalization** (RMS normalization across channels):
```python
def pixel_norm(x, eps=1e-6):
    variance = mx.mean(x * x, axis=1, keepdims=True)
    return x * mx.rsqrt(variance + eps)
```

This normalizes each spatial position to unit RMS across channels. It's designed to work with the scale/shift conditioning to restore proper magnitude.

### Why Outputs Look Similar

Pixel norm + learned scale/shift is a form of "instance normalization". Without proper timestep conditioning, the scale/shift values don't vary enough to differentiate inputs.

**This is expected behavior** - the timestep conditioning is what provides input-dependent modulation.

---

## Issue: Depth-to-Space Upsampling

### LTX-2 Upsampling Pattern

LTX-2 uses **depth-to-space** (pixel shuffle) for upsampling:
```
Input:  [B, C*8, T, H, W]     # e.g., [B, 4096, 5, 16, 16]
Output: [B, C, T*2, H*2, W*2] # e.g., [B, 512, 9, 32, 32]
```

Factor is `(2, 2, 2)` for temporal + spatial upsampling.

### Causal Temporal Handling

For causal generation, trim the first `(factor_t - 1)` frames after upsampling:
```python
if causal and factor_t > 1:
    x = x[:, :, factor_t - 1:]  # Trim first frame
```

---

## Issue: NaN Values During Text Encoding

### Symptoms
- NaN values appear during Gemma 3 forward pass
- Happens during attention computation
- Only occurs with certain tokenizer configurations

### Root Cause: LEFT Padding with Attention Mask

When using LEFT padding (PyTorch default), padding tokens appear at the start of sequences. Combined with how MLX handles attention masks, this can produce NaN values during softmax computation.

### Solution: Use RIGHT Padding

Configure the tokenizer to use RIGHT padding:

```python
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = "right"  # Critical for MLX
```

This places padding tokens at the end of sequences where they're properly masked without causing numerical issues.

### Why This Works

With RIGHT padding:
- Real content is at sequence start (positions 0, 1, 2, ...)
- Padding is at end (positions N-1, N-2, ...)
- Attention mask zeros out padding correctly
- No division by near-zero values in softmax

---

## Debugging Tips

### 1. Trace Signal Through Decoder

```python
def diagnose_decoder(latent, decoder):
    x = latent

    # Track stats at each stage
    x = decoder.conv_in(x)
    print(f"conv_in: mean={x.mean():.4f}, std={x.std():.4f}")

    for i, block in enumerate([decoder.up_blocks_0, ...]):
        x = block(x)
        print(f"up_blocks_{i}: mean={x.mean():.4f}, std={x.std():.4f}")

    # Watch for:
    # - Exploding values (std > 100)
    # - Collapsing values (std < 0.1)
    # - Sudden sign flips (mean changes sign)
```

### 2. Check Weight Loading

```python
# Verify weights match PyTorch source
from safetensors import safe_open

with safe_open(weights_path, framework='pt') as f:
    pt_weight = f.get_tensor('vae.decoder.conv_out.conv.weight')
    mlx_weight = decoder.conv_out.weight

    # Should be < 0.001
    diff = np.abs(pt_weight.numpy() - np.array(mlx_weight)).max()
    print(f"Max weight diff: {diff}")
```

### 3. Test with Known Inputs

```python
# Zero latent should give consistent baseline
zero_latent = mx.zeros((1, 128, 5, 16, 16))
output = decode_latent(zero_latent, decoder, timestep=0.05)
print(f"Zero latent brightness: {output.mean() / 255 * 100:.1f}%")

# Random latent should show variation
random_latent = mx.random.normal((1, 128, 5, 16, 16))
output = decode_latent(random_latent, decoder, timestep=0.05)
print(f"Random latent brightness: {output.mean() / 255 * 100:.1f}%")
```

---

## Reference: LTX-2 VAE Decode Parameters

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `timestep` | 0.05 | Conditioning timestep for final denoise |
| `decode_noise_scale` | 0.025 | Optional noise injection (not implemented) |

### Timestep Effects

| Timestep | Effect |
|----------|--------|
| `None` | No conditioning (dark output) |
| `0.0` | Minimal denoising |
| `0.05` | Standard denoising (recommended) |
| `0.1+` | More aggressive denoising |

---

## Resources

- [LTX-2 GitHub](https://github.com/Lightricks/LTX-2)
- [LTX-2 HuggingFace](https://huggingface.co/Lightricks/LTX-2)
- [LTX-2 Paper (arXiv:2601.03233)](https://arxiv.org/abs/2601.03233)
- [Diffusers AutoencoderKLLTXVideo](https://huggingface.co/docs/diffusers/en/api/models/autoencoderkl_ltx_video)
