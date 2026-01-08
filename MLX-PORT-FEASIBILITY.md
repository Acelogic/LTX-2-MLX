# LTX-2 MLX Port Feasibility Analysis

## Summary

**Can LTX-2 be ported to MLX?** Yes, but it's a significant undertaking.

LTX-2 is a 19-billion parameter DiT (Diffusion Transformer) for synchronized video and audio generation. Porting it to MLX (Apple's machine learning framework for Apple Silicon) is feasible but requires substantial work, primarily around 3D convolutions in the Video VAE.

---

## Major Porting Challenges

| Component | Difficulty | Notes |
|-----------|------------|-------|
| **3D Convolutions (Video VAE)** | Hard | MLX lacks native Conv3d. Would need custom implementation or decomposition to 2D+1D |
| **Triton Kernels** | Hard | FP8 weight upscaling kernel needs Metal shader replacement |
| **XFormers/FlashAttention** | Medium | Fall back to native attention (MLX has `scaled_dot_product_attention`) |
| **3D RoPE** | Medium | In-place `addcmul_()` needs rewrite; core math is portable |
| **Quantization (quanto)** | Medium | MLX has 4-bit/8-bit quantization, but different API |
| **Gemma Text Encoder** | Easy | LLMs work well in MLX (mlx-lm exists) |
| **Diffusion Components** | Easy | Schedulers, guiders, steppers are pure math |

---

## Critical Blockers

### 1. Video VAE 3D Convolutions

**Location**: `ltx-core/model/video_vae/convolution.py`

The Video VAE heavily relies on 3D convolutions for spatiotemporal compression:

```python
# These need MLX equivalents:
nn.Conv3d(...)                    # Standard 3D conv
F.conv3d(weight, bias, stride...) # Functional 3D conv
CausalConv3d(...)                 # Custom causal padding
DualConv3d(...)                   # Hybrid 2D+1D decomposition
```

**Porting Options**:
- Implement Conv3d as stacked 2D convolutions + 1D temporal convolutions
- Use the existing `DualConv3d.forward_with_2d()` path (already decomposes 3D→2D+1D)
- Write custom Metal kernels for native 3D conv support

**Key Files**:
- `convolution.py` lines 51, 182-206, 221-255 (Conv3d usage)
- `convolution.py` lines 266-313 (CausalConv3d class)
- `convolution.py` lines 90-260 (DualConv3d class)

### 2. Triton Kernel for FP8 Quantization

**Location**: `ltx-core/loader/kernels.py`

```python
@triton.jit
def fused_add_round_kernel(...):
    # FP8 weight upscaling with stochastic rounding
    # Uses bitcast operations and random number generation
```

**Porting Options**:
- Write equivalent Metal shader
- Implement in pure MLX (slower but functional)
- Skip FP8 initially, use FP16/BF16

### 3. Attention Backends

**Location**: `ltx-core/model/transformer/attention.py`

LTX-2 supports multiple attention backends:
- PyTorch native `scaled_dot_product_attention` (line 44)
- XFormers `memory_efficient_attention` (line 89)
- FlashAttention3 `flash_attn_func` (line 114)

**MLX Solution**: Use `mx.fast.scaled_dot_product_attention` - MLX's native implementation is efficient on Apple Silicon.

### 4. 3D RoPE (Rotary Position Embeddings)

**Location**: `ltx-core/model/transformer/rope.py`

The 3D RoPE implementation uses in-place operations:

```python
# Lines 59-60 - in-place operations not supported in MLX
first_half_output.addcmul_(-sin_freqs.unsqueeze(-2), second_half_input)
second_half_output.addcmul_(sin_freqs.unsqueeze(-2), first_half_input)
```

**MLX Solution**: Rewrite as functional operations (out-of-place). The core math is portable.

---

## What's Already Portable

These components have direct MLX equivalents or are pure math:

| Component | PyTorch | MLX Equivalent |
|-----------|---------|----------------|
| Transformer attention | `F.scaled_dot_product_attention` | `mx.fast.scaled_dot_product_attention` |
| RMSNorm | `torch.nn.RMSNorm` | `mx.fast.rms_norm` |
| GELU | `F.gelu(approximate="tanh")` | `nn.gelu` (verify tanh approx) |
| Linear layers | `nn.Linear` | `nn.Linear` |
| Layer normalization | `nn.LayerNorm` | `nn.LayerNorm` |
| Embeddings | `nn.Embedding` | `nn.Embedding` |
| SiLU/Swish | `F.silu` | `nn.silu` |
| Softmax | `F.softmax` | `mx.softmax` |
| Basic math ops | `torch.*` | `mx.*` |

### Diffusion Components (Pure Math)

These are straightforward to port:
- **Schedulers** (`components/schedulers.py`): Sigma/noise schedule generation
- **Guiders** (`components/guiders.py`): CFG blending logic
- **Noisers** (`components/noisers.py`): Noise addition
- **Steppers** (`components/diffusion_steps.py`): Euler step updates
- **Patchifiers** (`components/patchifiers.py`): Reshape operations

---

## Hardware Requirements

For running 19B parameters on Apple Silicon:

| Configuration | Memory | Precision | Feasibility |
|---------------|--------|-----------|-------------|
| M2/M3/M4 Max 64GB | 64GB | INT4 quantized | Minimum viable |
| M2/M3/M4 Max 96GB | 96GB | INT8 quantized | Good |
| M2/M3/M4 Ultra 128GB | 128GB | FP16/BF16 | Recommended |
| M4 Ultra 512GB | 512GB | FP16/BF16 | Ideal |
| M5 + Neural Accelerators | Varies | Mixed | Best performance |

**Note**: Unified memory architecture means the full model can fit in RAM without GPU memory constraints, but larger models still need sufficient total memory.

---

## Recommended Porting Strategy

### Phase 1: Core Infrastructure (1-2 weeks)
- Port diffusion components (schedulers, guiders, steppers)
- Port patchifiers (spatial ↔ sequence conversion)
- Implement basic tensor utilities
- Set up MLX project structure

### Phase 2: Text Encoder (1 week)
- Leverage existing mlx-lm Gemma implementation
- Port video/audio context connectors
- Port multi-layer feature extraction
- Test text encoding pipeline end-to-end

### Phase 3: Video VAE (3-4 weeks) - HARDEST
- Implement Conv3d via 2D+1D decomposition
- Port CausalConv3d with MLX padding operations
- Port DualConv3d hybrid convolution
- Port encoder architecture
- Port decoder architecture
- Validate latent space compatibility with PyTorch version

### Phase 4: Transformer (2-3 weeks)
- Port 48-layer dual-stream architecture
- Implement 3D RoPE without in-place operations
- Use MLX native scaled dot-product attention
- Port cross-modal (video↔audio) attention
- Port AdaLN (Adaptive Layer Normalization)
- Port feed-forward blocks

### Phase 5: Upscaler & Audio (2 weeks)
- Port spatial upsampler (LatentUpsampler)
- Port ResBlocks and pixel shuffle
- Port audio VAE encoder/decoder
- Port HiFi-GAN vocoder

### Phase 6: Optimization (2-3 weeks)
- Implement INT4/INT8 quantization using MLX quantization API
- Profile and identify bottlenecks
- Write custom Metal kernels for critical paths if needed
- Optimize memory usage for different hardware tiers

---

## Estimated Total Effort

| Phase | Duration | Complexity | Dependencies |
|-------|----------|------------|--------------|
| Phase 1: Core Infrastructure | 1-2 weeks | Low | None |
| Phase 2: Text Encoder | 1 week | Low-Medium | Phase 1 |
| Phase 3: Video VAE | 3-4 weeks | High | Phase 1 |
| Phase 4: Transformer | 2-3 weeks | Medium-High | Phases 1-3 |
| Phase 5: Upscaler & Audio | 2 weeks | Medium | Phase 3 |
| Phase 6: Optimization | 2-3 weeks | Medium-High | All phases |
| **Total** | **11-15 weeks** | - | - |

---

## Proof of Concept Recommendation

Start with a minimal proof-of-concept to validate the hardest parts:

1. **Week 1-2**: Port Video VAE encoder only
   - Implement Conv3d decomposition
   - Test encoding a single video frame
   - Compare latents with PyTorch version

2. **Week 3**: Port Video VAE decoder
   - Validate round-trip encoding/decoding
   - Measure quality degradation (if any)

3. **Week 4**: Port minimal transformer
   - Single denoising step
   - Validate output matches PyTorch

If the Video VAE port works well, the rest of the project is straightforward.

---

## Alternative Approaches

### Option A: Full Port (Recommended)
Port entire codebase to MLX for native Apple Silicon performance.
- **Pros**: Best performance, full control
- **Cons**: Most effort, maintenance burden

### Option B: Hybrid Approach
Keep Video VAE in PyTorch, port transformer to MLX.
- **Pros**: Less work on hardest component
- **Cons**: Data transfer overhead, complex setup

### Option C: ONNX/CoreML Export
Convert PyTorch model to ONNX, then CoreML.
- **Pros**: Apple's optimized runtime
- **Cons**: May not support all operations, less flexibility

### Option D: Wait for MLX Conv3d
Monitor MLX development for native 3D convolution support.
- **Pros**: Easiest long-term
- **Cons**: Unknown timeline

---

## Key Files Reference

| Component | File Path | Critical Lines |
|-----------|-----------|----------------|
| 3D Convolutions | `ltx-core/model/video_vae/convolution.py` | 51, 90-313 |
| Triton Kernel | `ltx-core/loader/kernels.py` | 1-73 |
| Attention | `ltx-core/model/transformer/attention.py` | 28-195 |
| 3D RoPE | `ltx-core/model/transformer/rope.py` | 42-66, 178-204 |
| Quantization | `ltx-trainer/quantization.py` | 19-90 |
| Video VAE | `ltx-core/model/video_vae/video_vae.py` | Entire file |
| Transformer | `ltx-core/model/transformer/model.py` | Entire file |
| Schedulers | `ltx-core/components/schedulers.py` | Entire file |

---

## Resources

- [MLX GitHub Repository](https://github.com/ml-explore/mlx)
- [MLX Documentation](https://ml-explore.github.io/mlx/build/html/index.html)
- [MLX Examples](https://github.com/ml-explore/mlx-examples)
- [mlx-lm (LLM support)](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm)
- [WWDC25: Get started with MLX](https://developer.apple.com/videos/play/wwdc2025/315/)
- [WWDC25: Explore LLMs on Apple Silicon](https://developer.apple.com/videos/play/wwdc2025/298/)
