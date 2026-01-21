# LTX-2 MLX Usage Guide

Complete guide for generating videos with LTX-2 on Apple Silicon.

## Quick Start

```bash
# Generate a video with default settings
python scripts/generate.py "A golden retriever running through a meadow"

# Generate with specific settings
python scripts/generate.py "A rocket launching into space" \
    --height 512 --width 768 \
    --frames 65 --steps 8 \
    --seed 42 --output rocket.mp4
```

## Installation

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- ~25GB available RAM (128GB recommended for high resolution)
- ffmpeg for video encoding

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/LTX-2-MLX.git
cd LTX-2-MLX

# Install dependencies (using uv - recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Or using pip
pip install mlx safetensors numpy pillow tqdm einops transformers sentencepiece protobuf

# Install ffmpeg
brew install ffmpeg
```

### Download Weights

```bash
# Interactive download (recommended)
uv run scripts/download_weights.py

# Or download specific weights
uv run scripts/download_weights.py --weights distilled gemma

# Or download everything
uv run scripts/download_weights.py --weights all
```

Available weights from [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2):

| Weight | Size | Description |
|--------|------|-------------|
| `ltx-2-19b-distilled.safetensors` | 43GB | Fast generation (8 steps) |
| `ltx-2-19b-distilled-fp8.safetensors` | 27GB | FP8 quantized |
| `ltx-2-19b-dev.safetensors` | 43GB | Higher quality (25-50 steps) |
| `ltx-2-spatial-upscaler-x2-1.0.safetensors` | 995MB | 2x resolution |
| `ltx-2-temporal-upscaler-x2-1.0.safetensors` | 262MB | 2x framerate |
| `ltx-2-19b-distilled-lora-384.safetensors` | 1.5GB | LoRA for two-stage |

## Generation Options

### Resolution

Height and width must be divisible by 32:

| Resolution | Aspect Ratio | Use Case |
|------------|--------------|----------|
| 256×384 | 2:3 | Fast testing |
| 480×704 | ~2:3 | Balanced quality/speed |
| 512×768 | 2:3 | High quality |
| 768×1024 | 3:4 | Maximum quality |

```bash
python scripts/generate.py "Your prompt" --height 512 --width 768
```

### Frame Count

Frames must satisfy `frames % 8 == 1`:

| Frames | Duration (24fps) | Latent Frames |
|--------|------------------|---------------|
| 17 | 0.7s | 3 |
| 33 | 1.4s | 5 |
| 65 | 2.7s | 9 |
| 97 | 4.0s | 13 |
| 121 | 5.0s | 16 |

```bash
python scripts/generate.py "Your prompt" --frames 65
```

### Steps

More steps = higher quality but slower:

| Model | Recommended Steps |
|-------|-------------------|
| Distilled | 5-8 |
| Dev | 25-50 |

```bash
python scripts/generate.py "Your prompt" --steps 8
```

### Seed

Control randomness for reproducible results:

```bash
python scripts/generate.py "Your prompt" --seed 42
```

## Pipelines

### Distilled Pipeline (Default)

Optimized for speed with 8-step generation:

```bash
python scripts/generate.py "A cat walking through a garden" \
    --pipeline distilled \
    --height 512 --width 768 \
    --frames 65 --steps 8
```

**Characteristics:**
- No CFG (classifier-free guidance)
- Fast generation (~2 minutes for 65 frames at 512×768)
- Good quality for most use cases

### One-Stage Pipeline

Full CFG control for maximum quality:

```bash
python scripts/generate.py "A majestic eagle soaring over mountains" \
    --pipeline one-stage \
    --height 480 --width 704 \
    --frames 65 --steps 20 \
    --cfg 5.0
```

**Characteristics:**
- CFG scale controls prompt adherence (3.0-7.0 typical)
- Higher steps = better quality
- Slower than distilled

### Two-Stage Pipeline

High-resolution output with upscaling:

```bash
python scripts/generate.py "A waterfall in a tropical forest" \
    --pipeline two-stage \
    --height 768 --width 1024 \
    --frames 65 \
    --steps-stage1 15 \
    --cfg 5.0 \
    --fp16
```

**Characteristics:**
- Stage 1: Generate at half resolution with CFG
- Stage 2: 2× spatial upscale + refinement
- Best for high-resolution output

## Memory Optimization

### FP16 Mode

Reduce memory usage by ~50%:

```bash
python scripts/generate.py "Your prompt" --fp16
```

### Low Memory Mode

Aggressive optimization for systems with <32GB RAM:

```bash
python scripts/generate.py "Your prompt" --low-memory
```

### Sequential Loading

Models are loaded/unloaded sequentially by default:
1. Gemma 3 → encode text → unload
2. Transformer → denoise → unload
3. VAE → decode video

## Example Prompts

### Nature Scenes
```bash
python scripts/generate.py "Ocean waves crashing on a sandy beach at sunset, golden hour lighting"
python scripts/generate.py "A serene forest with sunlight filtering through the trees"
python scripts/generate.py "Snow falling gently in a mountain landscape"
```

### Action Sequences
```bash
python scripts/generate.py "A golden retriever running through a sunny meadow with wildflowers"
python scripts/generate.py "A rocket ship launching into space with flames and smoke"
python scripts/generate.py "A sports car racing on a winding mountain road"
```

### Urban/Architectural
```bash
python scripts/generate.py "A bustling city street at night with neon lights and rain reflections"
python scripts/generate.py "Time-lapse of clouds moving over a modern cityscape"
```

### Cinematic Style
```bash
python scripts/generate.py "Cinematic shot of a lone astronaut on Mars, dramatic lighting"
python scripts/generate.py "Epic wide shot of a medieval castle at dawn"
```

## Prompt Tips

1. **Be descriptive**: Include details about lighting, setting, and mood
2. **Specify motion**: Describe what's moving and how
3. **Use cinematic terms**: "cinematic", "dramatic lighting", "wide shot"
4. **Mention style**: "photorealistic", "animation style", "oil painting"

## Output

Videos are saved as MP4 files with H.264 encoding at 24fps:

```bash
# Default output location
python scripts/generate.py "Your prompt"
# → saves to outputs/output.mp4

# Custom output path
python scripts/generate.py "Your prompt" --output my_video.mp4
```

## Troubleshooting

### Out of Memory

- Use `--fp16` flag
- Reduce resolution: `--height 256 --width 384`
- Reduce frames: `--frames 17`
- Use `--low-memory` flag

### Slow Generation

- Use `--pipeline distilled` (fastest)
- Reduce steps: `--steps 5`
- Reduce resolution and frames

### Video Quality Issues

- Increase steps: `--steps 8` or higher
- Use `--pipeline one-stage` with `--cfg 5.0`
- Try different seeds: `--seed 123`
- Use more descriptive prompts

### Black/Dark Output

This is typically a timestep conditioning issue. Ensure you're using the latest code with proper VAE timestep handling.

## API Usage

```python
from LTX_2_MLX.pipelines import DistilledPipeline
from LTX_2_MLX.model.text_encoder import create_av_text_encoder
from LTX_2_MLX.model.transformer import LTXModel, X0Model
from LTX_2_MLX.model.video_vae import SimpleVideoDecoder

# Load models
text_encoder = create_av_text_encoder()
transformer = LTXModel(...)
x0_model = X0Model(transformer)
vae_decoder = SimpleVideoDecoder()

# Create pipeline
pipeline = DistilledPipeline(
    text_encoder=text_encoder,
    transformer=x0_model,
    video_decoder=vae_decoder,
)

# Generate
video = pipeline(
    prompt="A cat walking through a garden",
    height=512,
    width=768,
    num_frames=65,
    num_steps=8,
    seed=42,
)
```

## Performance Benchmarks

Measured on M3 Max with 128GB unified memory:

| Resolution | Frames | Steps | Time |
|------------|--------|-------|------|
| 512×768 | 65 | 8 | ~2 min |
| 768×1024 | 65 | 8 | ~4 min |
| 512×768 | 97 | 8 | ~3 min |

VAE decoding adds ~10-15 seconds regardless of resolution.

## Text Encoding

### Automatic (Recommended)

The generation script automatically loads Gemma 3 if available:

```bash
python scripts/generate.py "A cat walking through a garden" \
    --gemma-path weights/gemma-3-12b
```

### Dummy Embeddings (Testing)

For testing the pipeline without Gemma:

```bash
python scripts/generate.py "A cat walking" --no-gemma --height 128 --width 128
```

### Gemma 3 Requirements

LTX-2 requires **Gemma 3 12B** (~25GB download, ~12GB in memory as FP16):

```bash
uv run scripts/download_weights.py --weights gemma
```

**Requirements:**
1. Get a HuggingFace token at: https://huggingface.co/settings/tokens
2. Accept the Gemma license at: https://huggingface.co/google/gemma-3-12b-it

## Known Limitations

### Memory Requirements

| Configuration | RAM Required |
|--------------|--------------|
| Text-to-Video (480×704, 97 frames) | ~28GB (FP32), ~18GB (FP16) |
| Two-Stage (960×1408, 97 frames) | ~44GB (sequential), ~59GB (parallel) |
| With Audio Generation | Add ~6GB |

**Recommendation**: Use `--low-memory` flag on systems with <32GB RAM.

### Audio Generation

- Works with `text-to-video` pipeline only (`--generate-audio`)
- Two-stage and distilled pipelines do not support audio
- Audio quality is experimental

## Current Status

### Working

- Full PyTorch parity verified (97%+ correlation)
- Text-to-video generation with semantic content
- 6 specialized pipelines
- Two-stage pipeline with spatial upscaling
- FP8 weight loading (`--fp8`)
- Temporal upscaler (`--upscale-temporal`)
- IC-LoRA conditioning (`--control-video`)
- Generic LoRA support (`--lora`)
- Resolutions up to 768x1024

### In Progress

- Audio-video joint generation with two-stage pipeline

### Pending

- Keyframe interpolation CLI integration
