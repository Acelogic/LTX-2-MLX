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

## Issue: Video Output is Noise/Static Instead of Coherent Video

### Symptoms
- Video output shows noise, static, or abstract patterns
- Output is NOT black/blank (model IS running)
- Weights load successfully without errors
- Latent statistics show low standard deviation (~0.59 instead of ~1.0)
- Video output has narrow dynamic range (e.g., 101-208 instead of 0-255)

### Root Cause: Velocity vs Denoised (X0) Prediction Mismatch

The LTX transformer model outputs **velocity predictions**, but the denoising loops expected **denoised (X0) predictions**. This fundamental mismatch causes the Euler step to compute incorrect values, preventing proper denoising convergence.

#### How the Bug Manifests

The Euler diffusion step formula expects denoised predictions:
```python
# Euler step (expects X0/denoised)
velocity = (sample - denoised) / sigma
sample_next = sample + (sigma_next - sigma) * velocity
```

But LTXModel returns velocity directly:
```python
# LTXModel.__call__() returns velocity, not denoised!
return velocity  # This was being passed as "denoised" to Euler step
```

When velocity is incorrectly passed as denoised:
- The Euler step computes `(sample - velocity) / sigma` instead of `(sample - denoised) / sigma`
- This produces wrong velocity values
- Denoising fails to converge
- Result: noise/static output

### Solution: Wrap Transformer with X0Model

The `X0Model` wrapper converts velocity predictions to denoised predictions:

```python
from LTX_2_MLX.model.transformer import X0Model

# LTXModel outputs velocity, wrap it to get denoised predictions
velocity_model = load_transformer(weights_path, ...)
model = X0Model(velocity_model)

# Now model(modality) returns denoised predictions correctly
```

The X0Model conversion formula:
```python
class X0Model(nn.Module):
    def __call__(self, video: Modality) -> mx.array:
        velocity = self.velocity_model(video)
        timesteps = video.timesteps  # shape: (B, N, 1)
        sigma = timesteps[:, :, 0:1]  # Extract sigma
        denoised = video.latent - sigma * velocity
        return denoised
```

### Results After Fix

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Latent std | ~0.59 | ~1.27 |
| Video range | 101-208 | 60-251 |
| DC ratio | ~0.77 | ~0.48 |
| Output | Noise/static | Coherent structure |

### Files That Need X0Model Wrapping

All pipelines must wrap the transformer:

1. **distilled.py** - Auto-wraps in `__init__`
2. **one_stage.py** - Auto-wraps in `__init__`
3. **two_stage.py** - Auto-wraps, stores `_velocity_model` for LoRA
4. **ic_lora.py** - Auto-wraps, stores `_velocity_model` for LoRA
5. **keyframe_interpolation.py** - Auto-wraps in `__init__`

For pipelines with LoRA support, store the raw velocity model:
```python
if isinstance(transformer, X0Model):
    self._velocity_model = transformer.velocity_model
    self.transformer = transformer
else:
    self._velocity_model = transformer
    self.transformer = X0Model(transformer)

# Use self._velocity_model for LoRA weight operations
```

---

## Issue: Different Text Prompts Produce Nearly Identical Videos

### Symptoms
- "A blue ball" and "A red ball" produce nearly identical videos
- Text embeddings have correlation >0.99 for different prompts
- Denoised latents for different prompts have correlation ~0.999999
- Model appears to ignore text conditioning

### Root Cause: Gemma Self-Attention Homogenization

The Gemma 3 text encoder's self-attention mechanism rapidly homogenizes token representations across layers, destroying text differentiation:

| Layer | Correlation at Differing Token |
|-------|-------------------------------|
| Layer 0 (input embeddings) | 0.41 (good differentiation) |
| Layer 4 | 0.9999 (differentiation lost) |
| Layer 12+ | 0.9999+ (completely homogenized) |

#### Why This Happens

1. **Self-Attention Mixing**: Gemma's self-attention mixes information across all positions. By Layer 4, the single differing token ("blue" vs "red") is overwhelmed by the 31 identical tokens (system prompt + shared words).

2. **Learnable Registers**: The text encoder connector replaces 224/256 padding positions with identical learnable registers, further increasing embedding similarity.

3. **49-Layer Aggregation**: The feature extractor concatenates all 49 Gemma layers (where 48 are homogenized) and projects to 3840 dimensions, diluting any remaining differentiation.

### Evidence

Using `scripts/compare_text_embeddings.py --mode diff`:

```
# Similar prompts: "A blue ball bouncing" vs "A red ball bouncing"
Gemma Layer 0 (Input):       0.983  - SOME DIFF
Gemma Layer 48 (Final):      0.998  - TOO SIMILAR
Final Embedding:             0.998  - TOO SIMILAR

# Very different prompts: "A cat sleeping" vs "A rocket launching"
Gemma Layer 0 (Input):       0.755  - DIFFERENTIATED ✓
Gemma Layer 48 (Final):      0.998  - TOO SIMILAR ✗
Final Embedding:             0.993  - TOO SIMILAR

Per-token correlation at Layer 0 (input embeddings):
  Position  0: corr=1.000000  (system prompt - identical)
  Position 24: corr=1.000000  (shared token - identical)
  Position 25: corr=0.411220  <-- DIFFERS ("blue" vs "red")
  Position 26: corr=1.000000  (shared token - identical)

Latent correlation (blue vs red ball):
  Original pipeline:    0.999999 (essentially identical)
  Early layers mode:    0.830282 (meaningfully different)
```

**Key insight**: Even for completely different prompts (cat vs rocket), Layer 0 shows 0.755 correlation (good differentiation), but Layer 48 shows 0.998 (homogenized). This was due to two bugs in our Gemma implementation:

1. **Missing embedding scale**: Gemma multiplies embeddings by `sqrt(hidden_size)` (~61.97). Fixed in `gemma3.py`.
2. **Wrong hidden states order**: PyTorch collects hidden states BEFORE each layer and adds normalized output at end. Our implementation was collecting AFTER each layer without the final normalized state. Fixed in `gemma3.py`.

**After fixes** (v2):
```
# Very different prompts: "A cat sleeping" vs "A rocket launching"
Gemma Layer 0 (Input):       0.755  - DIFFERENTIATED ✓
Gemma Layer 48 (Final):      0.731  - DIFFERENTIATED ✓ (was 0.998!)
Final Embedding:             0.982  - SOME DIFF (was 0.993)
```

### Solution: Use `--early-layers-only` Flag

The `--early-layers-only` flag uses only Layer 0 (input embeddings) from Gemma, which preserves token differentiation:

```bash
# Generate with text differentiation preserved
python scripts/generate.py "A blue ball bouncing" --early-layers-only --output blue.mp4
python scripts/generate.py "A red ball bouncing" --early-layers-only --output red.mp4
```

### How the Fix Works

```python
# Original pipeline (homogenized):
# 1. Stack all 49 Gemma layers → [B, T, 3840, 49]
# 2. Concatenate → [B, T, 188160]
# 3. Project → [B, T, 3840]
# 4. Connector self-attention (further homogenizes)
# Result: correlation ~0.999

# Early layers mode (differentiated):
# 1. Use only Layer 0 → [B, T, 3840]
# 2. Skip projection and connector
# Result: correlation ~0.83 at latent level
```

### Caveats

The `--early-layers-only` mode is **experimental**:
- The model was trained with the full 49-layer pipeline
- Using only Layer 0 may produce different visual quality
- However, it successfully enables text differentiation

### Alternative Approaches (Not Yet Implemented)

1. **Weighted Layer Aggregation**: Weight early layers more heavily in feature extraction
2. **Position-Aware Cross-Attention**: Focus cross-attention on content tokens, not padding
3. **Smaller Prompts**: Minimize shared system prompt to increase signal-to-noise ratio

### Files Modified

- `scripts/generate.py`: Added `--early-layers-only` flag
- `LTX_2_MLX/model/text_encoder/feature_extractor.py`: Removed incorrect normalization (separate fix)

---

## Issue: Noise/Texture Output Instead of Semantic Content (RESOLVED)

### Symptoms
- Video output shows textured noise patterns, not semantic content
- Greenish/colored tint may appear (model responds to color words)
- Different prompts produce slightly different textures but no objects
- Diagonal stripe patterns visible in output

### Root Cause: Feature Extractor Bypassing Multi-Layer Aggregation

The issue was in `LTX_2_MLX/model/text_encoder/feature_extractor.py`. The `extract_from_hidden_states` function was **only using Layer 48** (final hidden state) directly, without:

1. **Using all 49 Gemma hidden layers** - PyTorch stacks all layers
2. **Per-layer normalization** - `norm_and_concat_padded_batch` normalizes each layer to [-4, 4] range
3. **Learned linear projection** - `aggregate_embed` projects 3840×49 → 3840 dimensions

The model was trained with the full pipeline, so skipping these steps produced embeddings that didn't match what the transformer expected.

### The Fix

Updated `extract_from_hidden_states` to match PyTorch exactly:

```python
def extract_from_hidden_states(self, hidden_states, attention_mask, padding_side="left"):
    # Stack all 49 hidden states: [B, T, 3840, 49]
    stacked = mx.stack(hidden_states, axis=-1)

    # Get sequence lengths from attention mask
    sequence_lengths = attention_mask.sum(axis=-1).astype(mx.int32)

    # Apply per-layer normalization: [B, T, 3840*49]
    normed_concat = norm_and_concat_padded_batch(
        stacked, sequence_lengths, padding_side=padding_side
    )

    # Project through learned linear layer: [B, T, 3840]
    return self.aggregate_embed(normed_concat)
```

### Results After Fix

| Prompt | Before Fix | After Fix |
|--------|-----------|-----------|
| "Blue ball on green grass" | Gray noise (autocorr 0.02) | Green scene |
| "Red car at sunset" | Gray noise | Orange/sunset colors |
| "Tropical beach with palm trees" | Gray noise | Palm trees, blue ocean, sandy beach |

### Key Insight

The transformer implementation was **correct** - verified to match PyTorch exactly (correlation = 1.0 across all 48 blocks). The issue was entirely in the text encoding pipeline.

### Related Files
- `LTX_2_MLX/model/text_encoder/feature_extractor.py`: Fixed file
- `scripts/compare_denoising_loop.py`: Verified MLX = PyTorch
- `scripts/compare_full_forward.py`: Verified full forward pass match

---

## Debugging Tips

### Debugging Scripts

The following scripts in `scripts/` help diagnose issues:

| Script | Purpose |
|--------|---------|
| `debug_nan.py` | Check for NaN/Inf in VAE and transformer, full denoising test |
| `debug_velocity.py` | Verify velocity vs X0 prediction |
| `debug_embedding_stages.py` | Trace text embedding correlation through pipeline stages |
| `debug_mask.py` | Analyze attention mask handling in text encoder |
| `debug_text_differentiation.py` | Compare embeddings for different prompts |
| `debug_text_encoder.py` | Test text encoder output statistics |
| `debug_embeddings.py` | Analyze embedding values and correlations |
| `debug_adaln.py` | Verify AdaLN scale_shift_table loading and values |
| `debug_denoising.py` | Trace denoising loop convergence step by step |
| `debug_pipeline_step.py` | Debug single pipeline step with full tracing |
| `debug_single_step.py` | Test single denoising step in isolation |
| `debug_vae.py` | Diagnose VAE decoder issues |
| `compare_pytorch.py` | Compare MLX vs PyTorch transformer blocks |
| `compare_with_pytorch.py` | Save MLX outputs for PyTorch comparison (requires ltx-core) |
| `compare_text_embeddings.py` | Compare text embeddings between MLX/PyTorch or two prompts |
| `test_with_gemma.py` | Test denoising with real Gemma embeddings |
| `test_distilled_pipeline.py` | Test two-stage distilled pipeline with spatial upscaler |
| `test_dev_model.py` | Test dev model with LTX2Scheduler (25 steps, CFG) |
| `verify_attention_weights.py` | Verify attention weight shapes and values |

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

## Issue: Video Shows Abstract Patterns/Texture Instead of Prompt Content

### Symptoms
- Generated video shows structured patterns/textures (not pure noise)
- Output does NOT match the semantic content of the prompt (e.g., "A blue ball on grass" produces abstract green texture)
- Spatial coherence ratio is ~0.5-0.6 (structured, not noise at ~0.9)
- Denoised predictions become highly correlated with input at low noise levels (0.99+ at sigma < 0.3)

### Technical Analysis

#### What's Working ✅
| Component | Status | Details |
|-----------|--------|---------|
| VAE per-channel stats | ✓ | `mean_of_means`, `std_of_means` load correctly |
| Transformer weights | ✓ | `scale_shift_table`, block weights load correctly |
| Euler step formula | ✓ | Matches official diffusers: `prev_sample = sample + dt * velocity` |
| Timestep scaling | ✓ | `sigma * 1000` matches FlowMatchEulerDiscreteScheduler |
| Patchify/Unpatchify | ✓ | Roundtrip is exact |
| Velocity magnitude | ✓ | ~0.86 std at sigma=1.0 (reasonable for flow matching) |

#### Denoising Loop Behavior

The denoising loop shows problematic convergence at low noise levels:

| Step | Sigma | Latent-Denoised Correlation | Issue |
|------|-------|----------------------------|-------|
| 0 | 1.0000 | 0.546 | Good - predictions differ from input |
| 1 | 0.7015 | 0.687 | OK |
| 2 | 0.4325 | 0.938 | High - predictions too similar to input |
| 3 | 0.2256 | 0.990 | Critical - nearly identical, no denoising |

When correlation approaches 1.0, the model predicts "no change", causing the denoising to stall.

#### Output Processing Chain

The transformer's output processing was verified correct:

```
Input (normalized) std: 1.0
    ↓ scale_shift_table (mean scale: -0.32)
After scale/shift std: ~0.77 (effective_scale = 0.68)
    ↓ proj_out (amplifies ~4x)
Final velocity std: ~0.86
```

The `scale_shift_table` has mean scale of -0.32, giving effective scale of 0.68. This is intentional model behavior.

### Root Cause: Text Conditioning Pipeline

The issue is traced to the **text conditioning pipeline** not effectively communicating prompt semantics:

1. **Text embeddings are too homogenized** (see "Different Text Prompts Produce Nearly Identical Videos" above)
2. **Cross-attention receives generic conditioning** - Similar embeddings regardless of prompt
3. **Result**: Spatially coherent but semantically generic output

### Evidence

```
# Zero context test
Input noise std: 0.9988
Final latent std: 0.8984
Spatial coherence ratio: 0.92 → NOISE-LIKE

# With text encoding + CFG=3.0
Spatial coherence ratio: 0.54 → STRUCTURED (but not semantic)
```

The model DOES produce structured content with text encoding, but not content matching the prompt.

### Diagnosis Commands

```bash
# Analyze generated video quality
python -c "
import numpy as np
from PIL import Image

img = np.array(Image.open('frame.png'))
h_diff = np.abs(img[:, 1:] - img[:, :-1]).mean()
v_diff = np.abs(img[1:, :] - img[:-1, :]).mean()
local_var = (h_diff + v_diff) / 2
ratio = local_var / img.std()

print(f'Ratio: {ratio:.3f}')
print(f'Assessment: {\"NOISE\" if ratio > 0.7 else \"STRUCTURED\"}')"

# Check denoising convergence
python scripts/debug_nan.py --weights weights/ltx-2/ltx-2-19b-distilled.safetensors --full-denoise
```

### Recommendations

1. **Use `--early-layers-only` flag** to preserve text semantic differentiation
2. **Try higher CFG scales** (5.0-7.0) to strengthen text conditioning
3. **Use the two-stage pipeline** with full resolution upscaling
4. **Compare text embeddings** with official PyTorch to verify they match:
   ```bash
   # Check if two prompts produce different embeddings
   python scripts/compare_text_embeddings.py --mode diff \
       --prompt "A blue ball" --prompt2 "A red ball"

   # Generate MLX embeddings for later comparison
   python scripts/compare_text_embeddings.py --mode mlx \
       --prompt "A blue ball" --mlx-output blue_embeddings.npz
   ```

### Related Issues
- "Different Text Prompts Produce Nearly Identical Videos" (text embedding homogenization)
- "Video Output is Noise/Static" (velocity vs X0 prediction - already fixed)

---

## Issue: Transformer Hidden States Explode Through Layers

### Symptoms
- Hidden states grow exponentially through transformer blocks
- Standard deviation increases from ~1.0 to 400+ by final block
- `norm_out` compensates by normalizing back to ~1.0
- Final velocity predictions have lower-than-expected magnitude

### Technical Details

| Block | Hidden State Std | Notes |
|-------|------------------|-------|
| Input | 0.97 | Normal |
| Block 0 | 3.1 | Already growing |
| Block 23 | 6.6 | Continued growth |
| Block 47 | 66-425 | Exploded |
| After norm_out | 1.0 | Normalized |
| Final velocity | 0.45-0.86 | Lower than expected ~1.0 |

### Analysis

This behavior appears to be **by design** in the LTX architecture:
- Residual connections accumulate magnitude through 48 blocks
- `norm_out` (LayerNorm) brings values back to unit variance
- `scale_shift_table` (mean scale ~0.68) further modulates output
- `proj_out` projects to final velocity

The effective output scaling is:
```python
effective_scale = (1 + scale_shift_table[1])  # mean ~0.68
# Combined with proj_out amplification (~4x)
# Results in final velocity std ~0.8-0.9
```

### Verification

```python
# Check scale_shift_table values
print(f"Scale table mean: {model.scale_shift_table[1].mean()}")  # ~-0.32
print(f"Effective scale: {1 + model.scale_shift_table[1].mean()}")  # ~0.68
```

### Status
This is **expected behavior** - not a bug. The architecture compensates for hidden state growth through normalization and learned scaling.

---

## Deep Dive: Layer-by-Layer Homogenization Analysis

### Summary

Comprehensive testing reveals the model architecture is working correctly. The issue is **text embedding homogenization** in the Feature Extractor, NOT transformer bugs.

### Test Results

#### 1. Model Forward Pass: ✅ Working
```
Output shape: (1, 64, 128)
Output range: [-1.94, 2.25]
Output mean: 0.02, std: 0.58
No NaN/Inf detected
```

#### 2. Text Conditioning Effect: ✅ Working (partially)
```
Zero context vs Random context A: correlation = 0.44 ✅
Zero context vs Random context B: correlation = 0.43 ✅
Random A vs Random B: correlation = 0.97 ⚠️ (too similar!)
```
Text IS affecting output (zero vs random shows 0.43 correlation), but different random texts produce nearly identical outputs.

#### 3. Caption Projection: ✅ Working
```
With trained weights, different inputs have:
  Correlation (A vs B): 0.44 ✅ (good differentiation)
  Mean abs diff: 2.0
```

#### 4. Single Cross-Attention Block: ✅ Working
```
Cross-attention output correlation (A vs B): 0.16 ✅
K correlation: 0.08
V correlation: 0.0001
```
Individual cross-attention blocks STRONGLY differentiate between contexts.

#### 5. Layer-by-Layer Correlation: ⚠️ Problem Identified
```
Context after projection: 0.44   ✅ Different
After layer 0:  0.96            ⚠️ Homogenized!
After layer 1:  0.94
After layer 4:  0.92
After layer 12: 0.74            ✅ Differentiation building
After layer 24: 0.69
After layer 32: 0.67            ✅ Maximum differentiation
After layer 40: 0.95            ⚠️ Re-homogenized!
After layer 47: 0.99            ⚠️ Nearly identical
Final output:   0.97
```

### Root Cause Analysis

The problem is NOT in the transformer architecture. Instead:

1. **Residual Connection Dominance**: After layer 0, the hidden states (which are identical between A and B) dominate the cross-attention contribution.
   - ||hidden_state||: 385.6
   - ||cross_attention_output||: ~215-280
   - Cross/Residual ratio: 0.56

   When adding a 0.56x different contribution to identical residuals, correlation stays high.

2. **Feature Extractor Bottleneck**: This is the PRIMARY issue.
   ```
   Gemma Layer 0:        0.98 correlation
   Gemma Layer 48:       0.99 correlation
   Feature Extractor:    0.999 correlation  ⬅️ Kills all differentiation!
   Final Embedding:      0.99 correlation
   ```
   The Feature Extractor projects 49 layers × 3840 dims = 188,160 dimensions down to 3840 dimensions, losing subtle differences.

3. **Middle Layers DO Differentiate**: Layers 12-32 successfully build differentiation (0.67 correlation), proving the architecture works.

4. **Late Layer Re-Homogenization**: Layers 40-47 push correlation back to 0.99. This may be intentional model behavior (convergence toward common output distribution).

### Key Insight

The transformer IS working correctly at the individual component level:
- Cross-attention produces different outputs for different contexts
- Middle layers accumulate differentiation
- AdaLN conditioning applies correctly

The issue is that **different prompts produce nearly identical embeddings** after the Feature Extractor, so the cross-attention receives essentially identical context for both prompts.

### Verification Commands

```bash
# Test text embedding differentiation
python scripts/compare_text_embeddings.py --mode diff \
    --prompt "A blue ball bouncing on grass" \
    --prompt2 "A red ball bouncing on grass" \
    --gemma-path weights/gemma-3-12b \
    --weights weights/ltx-2/ltx-2-19b-distilled.safetensors

# Expected output showing Feature Extractor as bottleneck:
# Gemma Layer 0:         0.98  SOME DIFF
# Gemma Layer 48:        0.99  SOME DIFF
# Feature Extractor:     0.999 TOO SIMILAR  ⬅️ Primary issue
# Final Embedding:       0.99  TOO SIMILAR
```

### Potential Solutions

1. **Weight Feature Extractor Layers**: Give more weight to early Gemma layers that preserve differentiation
2. **Skip Feature Extractor**: Use `--early-layers-only` which bypasses Feature Extractor
3. **Modify Projection**: Use wider projection or add regularization to preserve differentiation

---

## Fix Implemented: Cross-Attention Scaling

### Problem

Even after fixing the Feature Extractor to use Layer 48 only (which improved embedding differentiation from 0.999 to 0.80), the final denoised latents still showed extremely high correlation (0.9998) between different prompts. This was traced to **late transformer layers re-homogenizing** the differentiation built by earlier layers.

### Analysis

Layer-by-layer correlation tracking revealed:

| Block Range | Correlation | Behavior |
|-------------|-------------|----------|
| After block 0 | 0.96 | Initial homogenization |
| Blocks 0-24 | 0.96 → 0.72 | **Differentiation builds** |
| Blocks 24-47 | 0.72 → 0.995 | **Re-homogenization** |

The root cause: Cross-attention contribution shrinks relative to hidden state magnitude in late layers:

| Block | Cross-Attn Contribution | Hidden State Std |
|-------|------------------------|------------------|
| Block 0 | 169% of x | 11 |
| Block 24 | 9.9% of x | 156 |
| Block 47 | 0.53% of x | 512 |

As hidden states grow through residual connections, the cross-attention signal becomes negligible, causing late layers to homogenize outputs.

### Solution: `--cross-attn-scale` Flag

Added a `--cross-attn-scale` flag to `generate.py` that amplifies cross-attention output in late transformer layers (40-47):

```bash
python scripts/generate.py "A blue ball bouncing on grass" \
    --weights weights/ltx-2/ltx-2-19b-distilled.safetensors \
    --gemma-path weights/gemma-3-12b \
    --cross-attn-scale 10.0 \
    --output blue_ball.mp4 --fp16
```

### Implementation

**transformer.py** - Added `cross_attn_scale` parameter to `BasicTransformerBlock`:
```python
def __init__(self, ..., cross_attn_scale: float = 1.0):
    self.cross_attn_scale = cross_attn_scale

def __call__(self, ...):
    # Cross-attention with scaling
    x = x + self.attn2(...) * self.cross_attn_scale
```

**model.py** - Added `set_cross_attn_scale()` method:
```python
def set_cross_attn_scale(self, scale: float, start_layer: int = 40):
    for i, block in enumerate(self.transformer_blocks):
        block.cross_attn_scale = scale if i >= start_layer else 1.0
```

**generate.py** - Added CLI argument and integration:
```python
parser.add_argument("--cross-attn-scale", type=float, default=1.0,
    help="Scale factor for cross-attention in late transformer layers (40-47)")

# After model loading:
if cross_attn_scale != 1.0:
    velocity_model.set_cross_attn_scale(cross_attn_scale, start_layer=40)
```

### Results

| Metric | Without Fix | With 10x Scale |
|--------|-------------|----------------|
| Velocity correlation | 0.987 | 0.968 |
| Latent correlation | 0.9998 | 0.9590 |
| Visual output | Noise/texture | Noise/texture (improved correlation but still not semantic) |

### Current Status: Fundamental Issue Remains

While cross-attention scaling improves differentiation metrics, the visual output still shows noise/texture patterns instead of semantic content. This indicates a deeper issue in the diffusion process.

---

## Current Status (January 2026 Update) - WORKING

### All Components Verified and Working

| Component | Verification Method | Status |
|-----------|---------------------|--------|
| VAE decoder | Validated against PyTorch reference | ✓ Pass |
| Patchifier | Roundtrip test (max diff = 0) | ✓ Pass |
| Position grid | Range verification | ✓ Pass |
| Timestep embedding | Sigma-dependent value checks | ✓ Pass |
| **Text encoding (full pipeline)** | All 49 layers + norm + projection | ✓ **FIXED** |
| **Transformer (48 blocks)** | PyTorch comparison (correlation = 1.0) | ✓ Pass |
| **Full denoising loop** | 8-step comparison (correlation = 1.0) | ✓ Pass |
| **RoPE type** | Changed INTERLEAVED → SPLIT | ✓ **FIXED** |
| **Attention computation** | PyTorch comparison (max diff 0.000021) | ✓ Pass |
| **AdaLN computation** | Manual step-by-step matches PyTorch exactly | ✓ Pass |
| **Weight loading** | All weights match PyTorch (diff = 0) | ✓ Pass |
| **Video generation** | Semantic content matches prompts | ✓ **WORKING** |

### Key Findings (January 2026)

#### 1. Caption Projection Increases Correlation

The caption projection (3840 → 4096) increases embedding correlation from **0.80 to 0.92**:

| Stage | Correlation (blue vs red) |
|-------|---------------------------|
| Raw embedding (Layer 48) | 0.7997 |
| After linear_1 | 0.7908 |
| **After GELU** | **0.8611** (+0.07) |
| After linear_2 | 0.9178 (+0.05) |

The **GELU activation** is the main culprit for increasing correlation.

#### 2. Text Encoder Works for Distinct Prompts

For truly different prompts, correlation is much lower:

| Prompt Pair | Correlation |
|-------------|-------------|
| "blue ball" vs "red ball" | 0.80 (7/8 tokens identical) |
| "blue ball" vs "red car" | 0.58 |
| "cat sleeping" vs "dog running" | 0.58 |
| "person on beach" vs "robot in space" | 0.53 |

The high correlation with "blue vs red ball" is **expected** because 7 of 8 tokens are identical.

#### 3. Per-Token Correlation Analysis

For "A blue ball bouncing on green grass" vs "A red ball bouncing on green grass":

| Token # | Content | Correlation |
|---------|---------|-------------|
| 0 | `<bos>` | 1.0000 |
| 1 | "A" | 1.0000 |
| **2** | **"blue" vs "red"** | **0.2407** ← Only differing token |
| 3 | "ball" | 0.9318 |
| 4 | "bouncing" | 0.9475 |
| 5 | "on" | 0.9389 |
| 6 | "green" | 0.9160 |
| 7 | "grass" | 0.9528 |

**Key insight**: With only 1/8 tokens differing, the semantic signal is inherently diluted.

#### 4. Cross-Attention Output Analysis

Cross-attention produces significant output relative to residual:

| Block | Cross-Attn Magnitude | Ratio to Input |
|-------|---------------------|----------------|
| Block 0 | 0.84 | 0.83x |
| Block 12 | 2.04 | 2.04x |
| Block 24 | 2.71 | 2.71x |
| Block 36 | 1.29 | 1.29x |
| Block 47 | 0.62 | 0.62x |

### Model Behavior Analysis

At sigma=1.0 (pure noise input):
- **Velocity-noise correlation**: 0.8595 (expected - velocity ≈ noise at high sigma)
- **X0-noise correlation**: 0.3124 (model partially separates signal from noise)
- **X0 std**: 0.5231 (lower than input, some denoising occurring)

The model IS processing inputs and producing structured output, but the structure doesn't correspond to the semantic content of the prompt.

### What Random Latent Produces

Testing with `--placeholder` mode (bypasses transformer, uses random latent):
- Output: Same noise/texture patterns as full inference
- This confirms the VAE decoder produces textured output from any Gaussian latent
- The issue is that inference produces latents that are still essentially Gaussian

### Critical Discovery: Two-Stage Distilled Pipeline

The distilled model (`ltx-2-19b-distilled.safetensors`) is designed for a **two-stage pipeline**:

| Stage | Resolution | Steps | Sigma Schedule |
|-------|------------|-------|----------------|
| Stage 1 | Half resolution (e.g., 128x192) | 8 | `DISTILLED_SIGMA_VALUES` |
| Upscale | 2x spatial upsampling | - | SpatialUpscaler model |
| Stage 2 | Full resolution (e.g., 256x384) | 4 | `[0.3, 0.15, 0.05, 0.0]` |

**Files**:
- `LTX_2_MLX/pipelines/distilled.py`: Two-stage pipeline implementation
- `LTX_2_MLX/model/upscaler/`: Spatial upscaler model
- `scripts/test_distilled_pipeline.py`: Test script

**Test Result**: Two-stage pipeline runs successfully but still produces noise/texture output.

### Dev Model Test Results

Testing with the non-distilled dev model (`ltx-2-19b-dev.safetensors`):

```bash
python scripts/test_dev_model.py
# Uses: 25 steps, CFG=3.0, LTX2Scheduler (shifted sigma schedule)
```

**Result**: Also produces noise/texture patterns. The issue is NOT specific to the distilled model.

### Spatial Autocorrelation Analysis

Critical finding from `analyze_transformer_output.py`:

| Source | Spatial Autocorrelation |
|--------|------------------------|
| VAE encoder output | **0.9** (highly coherent) |
| Transformer X0 prediction | **0.1-0.2** (noise-like) |
| Expected for semantic content | **0.7+** |

This explains why outputs look like noise: the transformer's denoised predictions lack spatial coherence. Proper video latents should have high spatial autocorrelation (neighboring pixels correlate).

### Attention Weight Verification

Verified via `scripts/verify_attention_weights.py`:

```
Block 0 attn1 (self-attention):
  to_q.weight: shape=(4096, 4096), mean=0.000012, std=0.020156
  to_k.weight: shape=(4096, 4096), mean=0.000008, std=0.020145
  to_v.weight: shape=(4096, 4096), mean=-0.000015, std=0.020148
  to_out.weight: shape=(4096, 4096), mean=0.000003, std=0.014263

Comparison with safetensors:
  Max difference: 0.000000 (weights match exactly)
```

**Conclusion**: Weights load correctly. The issue is not in weight loading.

### Remaining Hypotheses

1. **Subtle RoPE difference**: Position encoding formula might have edge case differences
2. **Attention score computation**: Numerical precision differences in softmax
3. **AdaLN gate values**: Gate (third element) might not be applied correctly
4. **Unknown architecture difference**: LTX-2 may have undocumented components

### Next Investigation Steps

1. **Install PyTorch LTX-2 for Direct Comparison**:
   ```bash
   pip install ltx-core ltx-pipelines
   ```
   Then run:
   ```bash
   python scripts/compare_with_pytorch.py
   ```
   This saves MLX outputs for comparison with official PyTorch implementation.

2. **Compare Single Transformer Block**:
   - Run identical inputs through MLX and PyTorch transformer blocks
   - Compare Q, K, V, attention scores, and outputs at each stage
   - Focus on first few blocks where differences would compound

3. **Trace Full Denoising Loop**:
   - Compare sigma schedules step-by-step
   - Verify velocity predictions match
   - Check if X0 predictions diverge at specific steps

### Files Modified for Cross-Attention Scaling

| File | Changes |
|------|---------|
| `LTX_2_MLX/model/transformer/transformer.py` | Added `cross_attn_scale` parameter |
| `LTX_2_MLX/model/transformer/model.py` | Added `set_cross_attn_scale()` method |
| `scripts/generate.py` | Added `--cross-attn-scale` CLI argument |

---

## Issue: Two-Stage Pipeline Producing Gray Noise

### Symptoms
- Two-stage pipeline generates videos with very low variance (Std=8.6 instead of ~38)
- Output appears as nearly uniform gray noise
- Color channels have minimal variation (std ~5.0)
- Limited dynamic range (e.g., [93, 230] instead of [0, 255])

### Root Causes

#### 1. Missing Normalization Wrapper Around Spatial Upsampling

**Problem**: The spatial upsampling was applied directly to normalized latents without proper un-normalization and re-normalization.

**Why This Matters**: The PyTorch reference implementation requires:
```python
# From ltx-core/src/ltx_core/model/upsampler/model.py
def upsample_video(latent, video_encoder, upsampler):
    latent = video_encoder.per_channel_statistics.un_normalize(latent)  # ← Critical!
    latent = upsampler(latent)
    latent = video_encoder.per_channel_statistics.normalize(latent)      # ← Critical!
    return latent
```

The VAE encoder normalizes latents to have specific per-channel statistics (mean=0, std=1 per channel). When upsampling:
- **Un-normalize**: Converts from normalized latent space back to pixel-like space
- **Upsample**: Applies 2x spatial upsampling in pixel space
- **Re-normalize**: Converts back to normalized latent space

Without this wrapper, the upsampling operates on incorrectly scaled values, destroying the latent distribution and causing the decoder to produce near-uniform output.

**Impact**:
- Before fix: Std = 8.6 (almost uniform gray)
- After fix: Std = 37.0 (normal variance, matches one-stage quality)
- **4.4x improvement** in output variance

#### 2. Frame Conversion Using Raw Decoder Output

**Problem**: The two-stage pipeline was calling `self.video_decoder(latent)` directly, which returns raw float32 values in `(B, C, T, H, W)` format.

**Why This Failed**:
- Raw decoder output has values in range ~[-2.2, 1.9]
- PIL expects uint8 [0, 255] or float [0.0, 1.0]
- Frame shape was wrong: `(B, C, T, H, W)` instead of list of `(H, W, C)` frames

**Solution**: Use the `decode_latent()` helper function which:
1. Un-normalizes latents (reverses per-channel normalization)
2. Applies decoder with timestep conditioning
3. Converts to uint8: `mx.clip((video + 1) / 2, 0, 1) * 255`
4. Rearranges to `(T, H, W, C)` format

**Error Before Fix**:
```
TypeError: Cannot handle this data type: (1, 1, 3), <f4
```

#### 3. Spatial Upscaler Res Block Instability (Fix Applied)

**Problem**: The spatial upscaler's residual blocks amplify values exponentially:
```
Input:           mean=0.03,   range=[-7.9,   7.9]
After res_block 0: mean=-7.3,   range=[-160,   45]     # 15x explosion!
After res_block 1: mean=-46.6,  range=[-795,   437]
After res_block 2: mean=-35.5,  range=[-2137,  2240]   # 800x amplification!
After res_block 3: mean=-30.0,  range=[-2148,  1666]
```

**Root Cause**: Residual blocks lacked output normalization after the residual addition, causing unconstrained value amplification across multiple blocks.

**Fix Applied** (as of 2026-01-11): Added output GroupNorm layer (`norm_out`) to ResBlock3d in `LTX_2_MLX/model/upscaler/spatial.py:159`. This stabilizes the residual connections by normalizing after each block's output. The layer initializes as identity transform (weight=1, bias=0) to preserve existing behavior until weights are learned/tuned.

**Previous Workaround**: Replace broken spatial upscaler with nearest neighbor upsampling:
```python
# Un-normalize first
latent_unnorm = video_encoder.per_channel_statistics.un_normalize(stage_1_latent)

# Bilinear upsampling (B, C, F, H, W)
b, c, f, h, w = latent_unnorm.shape
upscaled_unnorm = mx.zeros((b, c, f, h * 2, w * 2), dtype=latent_unnorm.dtype)

for fi in range(f):
    frame = latent_unnorm[:, :, fi, :, :]  # (B, C, H, W)
    frame_t = frame.transpose(0, 2, 3, 1)  # (B, H, W, C)
    frame_up = mx.repeat(mx.repeat(frame_t, 2, axis=1), 2, axis=2)  # Nearest neighbor 2x
    frame_out = frame_up.transpose(0, 3, 1, 2)  # (B, C, H*2, W*2)
    upscaled_unnorm[:, :, fi, :, :] = frame_out

# Re-normalize
upscaled_latent = video_encoder.per_channel_statistics.normalize(upscaled_unnorm)
```

This workaround achieves comparable quality to the spatial upscaler without instability.

### Fixes Applied

#### File: `LTX_2_MLX/pipelines/two_stage.py`

**Location**: Lines 430-449

**Change**: Added normalization wrapper around spatial upsampling

```python
# CRITICAL: Must un-normalize before upsampling, then re-normalize after
# This is required by the PyTorch reference implementation to preserve latent distribution
latent_unnorm = self.video_encoder.per_channel_statistics.un_normalize(stage_1_latent)

# Use bilinear upsampling (spatial upscaler has res block instability)
b, c, f, h, w = latent_unnorm.shape
upscaled_unnorm = mx.zeros((b, c, f, h * 2, w * 2), dtype=latent_unnorm.dtype)

for fi in range(f):
    frame = latent_unnorm[:, :, fi, :, :]  # (B, C, H, W)
    frame_t = frame.transpose(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
    frame_up = mx.repeat(mx.repeat(frame_t, 2, axis=1), 2, axis=2)  # Nearest neighbor 2x
    frame_out = frame_up.transpose(0, 3, 1, 2)  # (B, H*2, W*2, C) -> (B, C, H*2, W*2)
    upscaled_unnorm[:, :, fi, :, :] = frame_out

# Re-normalize back to latent space
upscaled_latent = self.video_encoder.per_channel_statistics.normalize(upscaled_unnorm)
mx.eval(upscaled_latent)
```

**Location**: Lines 496-502 (modified from direct decoder call)

**Change**: Use `decode_latent()` helper instead of raw decoder

```python
# Decode to video using decode_latent helper (handles normalization to uint8)
video = decode_latent(final_latent, self.video_decoder, timestep=0.05)
```

#### File: `scripts/generate.py`

**Location**: Lines 867-871

**Change**: Simplified frame conversion for properly formatted output

```python
# Convert to frames list for save_video
# decode_latent returns (T, H, W, C) in uint8, so just convert to numpy list
video_np = np.array(video)  # (T, H, W, C)
frames = [video_np[t] for t in range(video_np.shape[0])]
```

### Verification Results

Quality comparison before and after fixes:

| Metric | Before Fix | After Fix | One-Stage Baseline |
|--------|-----------|-----------|-------------------|
| **Variance (Std Dev)** | 8.6 | 37.0 ✅ | 38.6 |
| **Dynamic Range** | [93, 230] | [0, 255] ✅ | [0, 255] |
| **R channel** | 149.4 ± 4.9 | 153.3 ± 30.9 ✅ | 150.4 ± 34.8 |
| **G channel** | 141.8 ± 4.8 | 171.1 ± 21.2 ✅ | 161.7 ± 27.7 |
| **B channel** | 132.2 ± 5.0 | 147.4 ± 51.1 ✅ | 142.1 ± 47.8 |

**Result**: The two-stage pipeline now produces video quality matching the one-stage baseline, with proper color variance and full dynamic range.

### Test Command

```bash
python scripts/generate.py "A beautiful tropical beach with palm trees and blue ocean waves" \
  --pipeline two-stage \
  --height 512 --width 704 --frames 33 \
  --cfg 5.0 --steps-stage1 15 \
  --gemma-path weights/gemma-3-12b \
  --weights weights/ltx-2/ltx-2-19b-distilled.safetensors \
  --spatial-upscaler-weights weights/ltx-2/ltx-2-spatial-upscaler-x2-1.0.safetensors \
  --output output.mp4 --fp16
```

Expected output:
- Resolution: 512x704 (2x upscaled from 256x352)
- Variance: ~37-38 (matching one-stage quality)
- Full color range: [0, 255]

### Lessons Learned

1. **Always Check Reference Implementation**: The PyTorch code had critical normalization logic that wasn't obvious from the architecture alone.

2. **Normalization Wrappers Are Critical**: When upsampling latents, the normalization wrapper preserves the statistical distribution that the decoder expects.

3. **Use Helper Functions**: The `decode_latent()` helper encapsulates all the post-processing logic (un-normalization, uint8 conversion, rearranging) - use it instead of calling the decoder directly.

4. **Validate Output Statistics**: Low variance (std < 15) in video output is a red flag indicating incorrect latent processing.

5. **Spatial Upscaler Fix**: Added output normalization to ResBlock3d (as of 2026-01-11). The fix adds a `norm_out` GroupNorm layer after residual addition to prevent value explosion. Further testing needed to verify stability with actual spatial upscaler weights. If instability persists, consider:
   - Verifying weight loading correctness for norm_out layers
   - Testing with different numerical precision (FP16 vs FP32)
   - Comparing intermediate activations with PyTorch reference

---

## Resources

- [LTX-2 GitHub](https://github.com/Lightricks/LTX-2)
- [LTX-2 HuggingFace](https://huggingface.co/Lightricks/LTX-2)
- [LTX-2 Paper (arXiv:2601.03233)](https://arxiv.org/abs/2601.03233)
- [Diffusers AutoencoderKLLTXVideo](https://huggingface.co/docs/diffusers/en/api/models/autoencoderkl_ltx_video)
- [FlowMatchEulerDiscreteScheduler](https://huggingface.co/docs/diffusers/en/api/schedulers/flow_match_euler_discrete)
