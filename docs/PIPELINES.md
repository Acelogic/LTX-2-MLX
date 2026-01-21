# LTX-2 MLX Pipelines

LTX-2 MLX provides **6 specialized pipelines** for different use cases. Use the `--pipeline` flag to select.

## Available Pipelines

### 1. `text-to-video` (Default)

Standard text-to-video generation with simple CFG denoising.

```bash
python scripts/generate.py "A cat walking" \
    --pipeline text-to-video \
    --height 480 --width 704 --frames 25 --steps 7
```

**Best for**: Basic video generation, quick testing

### 2. `distilled` - Fast

Two-stage distilled model optimized for speed (no CFG, 10 total steps).

```bash
python scripts/generate.py "A cat walking" \
    --pipeline distilled \
    --height 480 --width 704 --frames 25
```

- **Stage 1**: 7 steps at half resolution
- **Stage 2**: 3 steps refinement
- **Speed**: ~2x faster than standard
- **Quality**: Good for most use cases
- **Best for**: Fast iteration, batch generation

### 3. `one-stage` - Quality

Single-stage CFG with full control and adaptive sigma scheduling.

```bash
python scripts/generate.py "A cat walking" \
    --pipeline one-stage \
    --cfg 5.0 --steps 20 \
    --height 480 --width 704 --frames 25
```

- Uses LTX2Scheduler for token-count-dependent sigma schedule
- Optional image conditioning via latent replacement
- Full CFG control with positive/negative prompts
- **Best for**: High-quality single-resolution generation

### 4. `two-stage` - HQ + Upscaling (Recommended for High-Res)

Two-stage pipeline with spatial upscaling for high-resolution output.

```bash
python scripts/generate.py "A cat walking" \
    --pipeline two-stage \
    --height 512 --width 704 \
    --cfg 5.0 --steps-stage1 15 \
    --spatial-upscaler-weights weights/ltx-2/ltx-2-spatial-upscaler-x2-1.0.safetensors \
    --distilled-lora weights/ltx-2/ltx-2-19b-distilled-lora-384.safetensors \
    --fp16
```

> **Note**: Resolution is automatically adjusted to be divisible by 64 (required for the two-stage pipeline).

- **Stage 1**: Generate at half resolution (256x352) with CFG
- **Stage 2**: 2x spatial upscale + 3 distilled refinement steps
- **Final output**: 512x704 (or higher)
- Combines CFG quality with distilled speed
- **Best for**: High-resolution video generation (512x704+)

**Quality**: Matches one-stage baseline with 2x resolution increase

## Additional Pipelines (Code-Only)

### 5. Image Conditioning LoRA (`ic_lora`)

Video-to-video generation with control signals (depth, pose, edges).

```python
from LTX_2_MLX.pipelines import ICLoraPipeline, create_ic_lora_pipeline

pipeline = create_ic_lora_pipeline(
    transformer=model,
    video_decoder=decoder,
    # ... requires IC-LoRA weights
)
```

- Two-stage with IC-LoRA in Stage 1 only
- Control signals: depth maps, human pose, edge detection
- **Best for**: Controlled video-to-video generation

### 6. Keyframe Interpolation (`keyframe_interpolation`)

Generate videos by interpolating between keyframe images.

```python
from LTX_2_MLX.pipelines import KeyframeInterpolationPipeline, Keyframe

keyframes = [
    Keyframe(image=img1, frame_index=0),
    Keyframe(image=img2, frame_index=24),
]

video = pipeline(keyframes=keyframes, ...)
```

- Two-stage with keyframe conditioning
- Smooth interpolation between images
- **Best for**: Image sequence animation

## Pipeline Comparison

| Pipeline | Speed | Quality | Resolution | CFG | Best Use Case |
|----------|-------|---------|------------|-----|---------------|
| `text-to-video` | Medium | Good | Any | Yes | Basic generation |
| `distilled` | **Fast** (10 steps) | Good | Up to 480p | No | Quick iteration |
| `one-stage` | Slow (20+ steps) | **High** | Any | Yes | Quality priority |
| `two-stage` | Medium (18 steps) | **High** | **512p+** | Yes | High-resolution |
| `ic_lora` | Medium | High | 512p+ | Yes | Controlled gen |
| `keyframe_interpolation` | Medium | High | 512p+ | Yes | Image animation |

## Recommended Settings by Use Case

### Fast Previews

```bash
python scripts/generate.py "Your prompt" \
    --pipeline distilled --height 512 --width 768 --frames 65
```

### High Quality

```bash
python scripts/generate.py "Your prompt" \
    --pipeline one-stage --height 480 --width 704 --frames 65 \
    --steps 15 --cfg 4.0 --fp16
```

### High Resolution

```bash
python scripts/generate.py "Your prompt" \
    --pipeline two-stage --height 768 --width 1024 --frames 65 \
    --steps-stage1 15 --cfg 5.0 --fp16
```

## Command Line Options

| Flag | Description | Default |
|------|-------------|---------|
| `--pipeline` | Pipeline: `text-to-video`, `distilled`, `one-stage`, `two-stage` | text-to-video |
| `--height` | Video height (divisible by 32) | 480 |
| `--width` | Video width (divisible by 32) | 704 |
| `--frames` | Number of frames (N*8+1) | 97 |
| `--steps` | Denoising steps | 8 |
| `--steps-stage1` | Stage 1 steps (two-stage pipeline) | 15 |
| `--steps-stage2` | Stage 2 steps (two-stage pipeline) | 3 |
| `--cfg` | Classifier-free guidance scale | 5.0 |
| `--seed` | Random seed | 42 |
| `--output` | Output video path | outputs/output.mp4 |
| `--weights` | Path to weights file | weights/ltx-2/ltx-2-19b-distilled.safetensors |
| `--fp16` | Use FP16 computation (~50% memory reduction) | True |
| `--fp8` | Load FP8-quantized weights | False |
| `--model-variant` | `distilled` (fast) or `dev` (quality) | distilled |
| `--spatial-upscaler-weights` | Path to spatial upscaler weights (for two-stage) | None |
| `--temporal-upscaler-weights` | Path to temporal upscaler weights | None |
| `--upscale-spatial` | Apply 2x spatial upscaling (legacy) | False |
| `--upscale-temporal` | Apply 2x temporal upscaling (legacy) | False |
| `--generate-audio` | Generate synchronized audio (experimental) | False |
| `--low-memory` | Aggressive memory optimization (~30% less) | False |
| `--skip-vae` | Skip VAE decoding (output latent visualization) | False |
| `--no-gemma` | Use dummy embeddings (testing only) | False |
| `--embedding` | Path to pre-computed text embedding (.npz) | None |
| `--gemma-path` | Path to Gemma 3 weights | weights/gemma-3-12b |

## Frame Count

LTX-2 requires frames to satisfy `frames % 8 == 1`:
- Valid: 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 121
- Formula: latent_frames = 1 + (frames - 1) / 8
