# LTX-2-MLX

Native Apple Silicon implementation of [Lightricks LTX-2](https://github.com/Lightricks/LTX-2), a 19B parameter video generation model, using MLX.

## Features

- **Native MLX Implementation**: Full transformer, VAE decoder, and text encoder components ported to MLX
- **Apple Silicon Optimized**: Designed for M-series Macs (tested on M3 Max 128GB)
- **Weight Loading**: Direct loading from PyTorch safetensors weights
- **End-to-End Pipeline**: Text-to-video generation with denoising, VAE decoding, and video export

## Project Structure

```
ltx_mlx/
├── model/
│   ├── transformer/     # 48-layer DiT with RoPE
│   ├── video_vae/       # VAE encoder/decoder with 3D convolutions
│   ├── text_encoder/    # Gemma feature extractor + connector
│   └── upsampler/       # 2x spatial upscaler
├── components/          # Schedulers, patchifiers, guiders
├── loader/              # Weight conversion utilities
└── types.py             # Type definitions
scripts/
└── generate.py          # Main generation script
```

## Requirements

- Python 3.10+
- macOS with Apple Silicon (M1/M2/M3)
- ~40GB RAM for FP32 inference (128GB recommended)

### Dependencies

```bash
pip install mlx mlx-lm safetensors numpy pillow tqdm
```

For video encoding:
```bash
brew install ffmpeg
```

## Weights

Download the LTX-2 distilled weights:

```bash
mkdir -p weights/ltx-2
# Download ltx-2-19b-distilled.safetensors to weights/ltx-2/
```

## Usage

### Basic Generation

```bash
python scripts/generate.py "A cat walking through a garden" \
    --height 128 --width 128 \
    --frames 17 --steps 3 \
    --output output.mp4
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--height` | Video height (divisible by 32) | 480 |
| `--width` | Video width (divisible by 32) | 704 |
| `--frames` | Number of frames (N*8+1) | 97 |
| `--steps` | Denoising steps | 7 |
| `--cfg` | Classifier-free guidance scale | 3.0 |
| `--seed` | Random seed | 42 |
| `--weights` | Path to weights file | weights/ltx-2/ltx-2-19b-distilled.safetensors |
| `--skip-vae` | Skip VAE decoding (output latent visualization) | False |
| `--placeholder` | Use random noise instead of model inference | False |

### Frame Count

LTX-2 requires frames to satisfy `frames % 8 == 1`:
- Valid: 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97
- After VAE temporal compression (8x), output frames = (input_frames - 1) + 1

## Architecture

### Transformer (19B parameters)

- 48 transformer layers
- 32 attention heads × 128 dim = 4096 hidden
- 3D RoPE for video position encoding
- Cross-attention to text embeddings

### VAE Decoder

- 128 latent channels → 3 RGB channels
- 8x temporal upsampling (3 stages of 2x)
- 32x spatial upsampling (8x from depth-to-space + 4x from unpatchify)
- Pixel norm with scale/shift conditioning

### Text Encoder

- **Gemma 3 12B** feature extraction (48 layers × 3840 dim)
- Aggregation projection (188160 → 3840)
- 2-layer 1D transformer connector
- Caption projection (3840 → 4096)

## Text Encoding Options

### Option 1: Native MLX Encoding (Recommended)

Use the native MLX pipeline with Gemma 3 12B:

```bash
# Step 1: Download Gemma 3 12B (requires HuggingFace token)
python scripts/download_gemma.py

# Step 2: Encode your prompt
python scripts/encode_text_mlx.py "A cat walking through a garden" \
    --gemma-path weights/gemma-3-12b \
    --ltx-weights weights/ltx-2/ltx-2-19b-distilled.safetensors \
    --output prompt_embedding.npz

# Step 3: Generate video
python scripts/generate.py --embedding prompt_embedding.npz \
    --height 480 --width 704 --frames 25 --steps 5
```

### Option 2: Dummy Embeddings (Testing)

For testing without Gemma, the pipeline uses deterministic random embeddings:

```bash
python scripts/generate.py "A cat walking" --height 128 --width 128
```

### Option 3: PyTorch Encoding (Alternative)

Use PyTorch with the official LTX-2 text encoder (requires triton):

```bash
python scripts/encode_with_pytorch.py "Your prompt here" \
    --gemma-path weights/gemma-3-12b \
    --output prompt_embedding.npz
```

### Downloading Gemma 3 12B

LTX-2 requires **Gemma 3 12B** in safetensors format (~25GB):

```bash
# Using the download script (requires HuggingFace token):
pip install huggingface_hub
python scripts/download_gemma.py --token YOUR_HF_TOKEN

# Or set environment variable:
export HF_TOKEN=your_token
python scripts/download_gemma.py
```

Get your token at: https://huggingface.co/settings/tokens

Accept the Gemma license at: https://huggingface.co/google/gemma-3-12b-it

## Current Status

### Working
- Full end-to-end text-to-video generation
- Native MLX Gemma 3 12B text encoder
- Transformer forward pass with loaded weights (19B parameters)
- VAE decoder with weight loading
- Video export via ffmpeg
- Tested at 480x704 resolution

### Performance (M3 Max 128GB)
- 128x128, 3 steps: ~16s
- 256x256, 5 steps: ~26s
- 480x704, 5 steps: ~56s

### Pending
- Memory optimization for longer videos
- Image-to-video conditioning
- LoRA support

## License

This project is for research and educational purposes. See the original [LTX-Video](https://github.com/Lightricks/LTX-Video) repository for model licensing.

## Acknowledgments

- [Lightricks](https://www.lightricks.com/) for LTX-2
- [Apple MLX Team](https://github.com/ml-explore/mlx) for the MLX framework
