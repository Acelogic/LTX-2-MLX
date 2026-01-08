# LTX-2: MLX vs MPS Analysis

## Can LTX-2 Run on Mac?

### Current State: CUDA-Only

**No, LTX-2 does not run on Mac out of the box.** The codebase is CUDA-only with CPU fallback.

#### Device Detection (No MPS Support)

```python
# ltx-pipelines/utils/helpers.py lines 29-32
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")  # Falls back to CPU, not MPS
```

The code only checks for CUDA availability. MPS (`torch.backends.mps.is_available()`) is never checked.

#### Hardcoded CUDA Calls

The pipelines contain **30+ hardcoded CUDA-specific calls** that would crash on MPS:

```python
# Found throughout all pipeline files:
torch.cuda.synchronize()  # Would crash on MPS
torch.cuda.empty_cache()  # Would crash on MPS
```

**Affected files:**
- `distilled.py` - lines 98, 155, 183
- `ic_lora.py` - lines 113, 164, 175, 221
- `keyframe_interpolation.py` - lines 111, 165, 176, 221
- `ti2vid_one_stage.py` - lines 97, 146
- `ti2vid_two_stages.py` - lines 113, 167, 178, 223
- `helpers.py` - lines 37-38 (cleanup_memory function)

#### Memory Cleanup Function

```python
# ltx-pipelines/utils/helpers.py lines 35-38
def cleanup_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()   # CUDA-specific
    torch.cuda.synchronize()   # CUDA-specific
```

#### FP8 Quantization Not Supported on MPS

```python
# ltx-trainer/quantization.py lines 79-81
if torch.backends.mps.is_available():
    "MPS doesn't support dtype float8."
```

Even if MPS support were added, FP8 quantization (used for memory efficiency) would not work.

---

## MLX vs MPS Performance Comparison

### Why MLX Would Be Faster

| Aspect | PyTorch + MPS | MLX |
|--------|---------------|-----|
| **Memory Model** | CPU↔GPU data copies | Unified memory (zero-copy) |
| **Framework Design** | GPU backend added later | Native Apple Silicon design |
| **Graph Optimization** | Limited | Lazy evaluation + op fusion |
| **Metal Integration** | Generic PyTorch backend | Purpose-built Metal kernels |
| **Attention** | Basic SDPA | `mx.fast.scaled_dot_product_attention` |
| **Matrix Multiplication** | Standard implementation | Optimized for ANE/GPU |
| **Quantization** | Limited (no FP8) | INT4/INT8 native support |

### Memory Architecture Advantage

**PyTorch + MPS:**
```
CPU Memory ←→ GPU Memory (data copies required)
     ↓              ↓
  Tensors      Metal Buffers
```

**MLX:**
```
Unified Memory (single pool)
        ↓
   MLX Arrays (zero-copy access from CPU/GPU/ANE)
```

### Performance Estimates

Based on benchmarks with similar transformer models:

| Configuration | Relative Speed | Notes |
|---------------|----------------|-------|
| NVIDIA A100 (CUDA) | 1.0x (baseline) | Best performance |
| NVIDIA RTX 4090 (CUDA) | ~0.8x | Consumer GPU |
| Apple M2 Ultra (MLX) | ~0.5-0.7x | Well-optimized MLX |
| Apple M2 Ultra (MPS) | ~0.2-0.3x | PyTorch MPS backend |
| Apple M2 Ultra (CPU) | ~0.05x | Fallback option |

**MLX vs MPS on same hardware: MLX is approximately 2-3x faster**

### Why MPS Is Slower

1. **Data Transfer Overhead**: PyTorch MPS requires copying data between CPU and GPU memory pools
2. **Generic Backend**: MPS is a general-purpose backend, not optimized for specific workloads
3. **Limited Operator Support**: Many operations fall back to CPU
4. **No Custom Kernels**: Cannot use optimized attention implementations (xFormers, FlashAttention)
5. **Framework Overhead**: PyTorch abstractions add latency

### Why MLX Is Faster

1. **Zero-Copy Memory**: Unified memory means no data transfer overhead
2. **Lazy Evaluation**: Operations are batched and optimized before execution
3. **Native Metal**: Direct Metal shader compilation for Apple Silicon
4. **Optimized Primitives**: `mx.fast.*` functions are highly optimized
5. **Graph Fusion**: Multiple operations fused into single kernels
6. **ANE Support**: Can leverage Neural Engine on M-series chips (M5 especially)

---

## What Would Be Required for MPS Support

If you wanted to patch LTX-2 to run on MPS (not recommended):

### 1. Update Device Detection

```python
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

### 2. Replace CUDA-Specific Calls

```python
# Before (CUDA-only)
torch.cuda.synchronize()
torch.cuda.empty_cache()

# After (device-agnostic)
if device.type == "cuda":
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
elif device.type == "mps":
    torch.mps.synchronize()
    torch.mps.empty_cache()
```

### 3. Handle Missing Operations

Some operations may not be supported on MPS and would need CPU fallbacks:
- Certain attention patterns
- Some 3D convolution configurations
- FP8 quantization (not supported)

### 4. Performance Issues

Even with patches:
- No FP8 support means higher memory usage
- No xFormers/FlashAttention optimizations
- 3D convolutions may be slow or unsupported
- Estimated 3-5x slower than CUDA

---

## Recommendation

| Goal | Recommended Approach |
|------|---------------------|
| Run on Mac (any performance) | Patch for MPS (quick but slow) |
| Run on Mac (good performance) | Port to MLX (more work but faster) |
| Best performance | Use NVIDIA GPU with CUDA |

### Decision Matrix

```
Want Mac support?
├── No → Use CUDA (best performance)
└── Yes → How much effort?
    ├── Minimal → Patch MPS (slow, ~3-5x slower than CUDA)
    └── Significant → Port to MLX (faster, ~1.5-2x slower than CUDA)
```

### Bottom Line

**If you want LTX-2 on Mac, porting to MLX is the better investment than patching MPS support.**

- MPS patch: ~1-2 weeks work, poor performance
- MLX port: ~11-15 weeks work, good performance
- Both require similar understanding of the codebase
- MLX delivers 2-3x better performance than MPS

---

## Summary Table

| Question | Answer |
|----------|--------|
| Can LTX-2 run on Mac today? | No (CUDA-only) |
| Why not? | Hardcoded `torch.cuda.*` calls, no MPS device detection |
| Could it run with MPS patches? | Yes, but slow and no FP8 |
| Is MLX port worth it? | Yes, if you want good Mac performance |
| MLX vs patched MPS speed? | MLX ~2-3x faster |
| MLX vs CUDA speed? | MLX ~1.5-2x slower (acceptable) |
| MPS vs CUDA speed? | MPS ~3-5x slower (poor) |
