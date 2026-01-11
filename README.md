# LTX-2-MLX

Native Apple Silicon implementation of [Lightricks LTX-2](https://github.com/Lightricks/LTX-2), a 19B parameter joint audio-video generation model, using MLX.

## Example Outputs

Generated videos from text prompts (frames shown):

| "A beautiful tropical beach with palm trees and blue ocean waves" | "A red sports car driving on a highway at sunset" | "A blue ball bouncing on green grass" |
|:---:|:---:|:---:|
| ![Beach](examples/beach_palm_trees.png) | ![Sunset Car](examples/sunset_car.png) | ![Green Grass](examples/green_grass.png) |

*Generated at 480x704 (beach) and 256x384 (others) with 8 denoising steps using the distilled model.*

## Features

- **Native MLX Implementation**: Full transformer (19B), VAE decoder, audio VAE, vocoder, and text encoder ported to MLX
- **Apple Silicon Optimized**: Designed for M-series Macs (tested on M3 Max 128GB)
- **FP16 Inference**: `--fp16` flag reduces memory usage by ~50%
- **Audio-Video Generation**: Support for synchronized audio with video (experimental)
- **Spatial/Temporal Upscaling**: 2x resolution and 2x framerate upscalers
- **CFG Support**: Classifier-Free Guidance with configurable scale
- **Memory Optimization**: Intermediate tensor cleanup during inference

## Project Structure

```
LTX_2_MLX/
├── model/
│   ├── transformer/     # 48-layer DiT with 3D RoPE (14B video + 5B audio)
│   ├── video_vae/       # VAE encoder/decoder with 3D convolutions
│   ├── audio_vae/       # Audio VAE decoder + HiFi-GAN vocoder
│   ├── text_encoder/    # Gemma 3 12B feature extractor + connector
│   └── upscaler/        # Spatial (2x res) and temporal (2x fps) upscalers
├── components/          # Schedulers, patchifiers, guiders
├── loader/              # Weight conversion utilities
└── types.py             # Type definitions
scripts/
├── generate.py          # Main generation script
├── download_gemma.py    # Gemma weights download
├── encode_text_mlx.py   # Standalone text encoding
└── validate_*.py        # Validation scripts
```

## Requirements

- Python 3.10+
- macOS with Apple Silicon (M1/M2/M3/M4)
- ~25GB RAM for FP16 inference, ~45GB for FP32 (128GB recommended)

### Dependencies

```bash
pip install mlx safetensors numpy pillow tqdm transformers
```

For video/audio encoding:
```bash
brew install ffmpeg
```

## Weights

Download the LTX-2 weights from HuggingFace:

```bash
mkdir -p weights/ltx-2
# Download one of:
# - ltx-2-19b-distilled.safetensors (43GB, BF16, 3-7 steps)
# - ltx-2-19b-distilled-fp8.safetensors (27GB, FP8, 3-7 steps)
# - ltx-2-19b-dev.safetensors (43GB, BF16, 25-50 steps)
```

Optional upscaler weights:
```bash
# - ltx-2-spatial-upscaler-x2-1.0.safetensors (995MB)
# - ltx-2-temporal-upscaler-x2-1.0.safetensors (262MB)
```

## Usage

### Basic Generation

```bash
python scripts/generate.py "A cat walking through a garden" \
    --height 256 --width 256 \
    --frames 17 --steps 5 \
    --output output.mp4
```

### With FP16 (Recommended for lower memory)

```bash
python scripts/generate.py "A cat walking through a garden" \
    --height 480 --width 704 \
    --frames 25 --steps 7 \
    --fp16 --output output.mp4
```

### With Upscaling

```bash
# Generate at 256x256, upscale to 512x512
python scripts/generate.py "A cat walking through a garden" \
    --height 256 --width 256 \
    --frames 17 --steps 5 \
    --upscale-spatial --output output.mp4
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
| `--output` | Output video path | gens/output.mp4 |
| `--weights` | Path to weights file | weights/ltx-2/ltx-2-19b-distilled.safetensors |
| `--fp16` | Use FP16 computation (~50% memory reduction) | False |
| `--fp8` | Load FP8-quantized weights | False |
| `--model-variant` | `distilled` (fast) or `dev` (quality) | distilled |
| `--upscale-spatial` | Apply 2x spatial upscaling | False |
| `--upscale-temporal` | Apply 2x temporal upscaling | False |
| `--generate-audio` | Generate synchronized audio (experimental) | False |
| `--low-memory` | Aggressive memory optimization (~30% less) | False |
| `--skip-vae` | Skip VAE decoding (output latent visualization) | False |
| `--no-gemma` | Use dummy embeddings (testing only) | False |
| `--embedding` | Path to pre-computed text embedding (.npz) | None |
| `--gemma-path` | Path to Gemma 3 weights | weights/gemma-3-12b |

### Frame Count

LTX-2 requires frames to satisfy `frames % 8 == 1`:
- Valid: 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 121
- Formula: latent_frames = 1 + (frames - 1) / 8

## Architecture

LTX-2 is a **joint audio-video diffusion transformer** with asymmetric dual streams:

```
Text Prompt → Gemma 3 (12B) → [Video Context, Audio Context]
                                      ↓
              48-Layer Asymmetric Dual-Stream Transformer
              ┌─────────────────────────────────────────┐
              │  Video Stream (14B) ←→ Audio Stream (5B) │
              │  Cross-attention for synchronization     │
              └─────────────────────────────────────────┘
                                      ↓
              Video VAE Decoder → Pixels (up to 768×512, 25fps)
              Audio VAE Decoder → Mel → Vocoder → 24kHz audio
```

### Transformer (19B parameters)

- 48 transformer layers
- **Video stream** (14B): 32 heads × 128 dim = 4096 hidden, 3D RoPE (x, y, t)
- **Audio stream** (5B): 16 heads × 128 dim = 2048 hidden, 1D RoPE (temporal)
- Bidirectional audio-video cross-attention for synchronization
- Cross-attention to text embeddings

### Video VAE Decoder

- 128 latent channels → 3 RGB channels
- 1:192 compression ratio (32x spatial × 8x temporal)
- **Timestep conditioning**: performs final denoising step during decode
- Pixel norm with scale/shift conditioning

### Audio VAE Decoder + Vocoder

- 8 latent channels → mel spectrogram
- HiFi-GAN vocoder: mel → 24kHz stereo audio waveform

### Text Encoder

- **Gemma 3 12B** feature extraction (48 layers × 3840 dim)
- Multi-layer aggregation from all decoder layers
- Separate video/audio context projections
- Caption projection (3840 → 4096 for video, 3840 → 2048 for audio)

## Text Encoding Options

### Option 1: Automatic (Recommended)

The generation script automatically loads Gemma 3 if available:

```bash
python scripts/generate.py "A cat walking through a garden" \
    --gemma-path weights/gemma-3-12b \
    --height 256 --width 256 --frames 17
```

### Option 2: Pre-computed Embeddings

Encode once, generate multiple times:

```bash
# Encode prompt
python scripts/encode_text_mlx.py "A cat walking through a garden" \
    --gemma-path weights/gemma-3-12b \
    --ltx-weights weights/ltx-2/ltx-2-19b-distilled.safetensors \
    --output prompt_embedding.npz

# Generate with pre-computed embedding
python scripts/generate.py --embedding prompt_embedding.npz \
    --height 480 --width 704 --frames 25 --steps 5
```

### Option 3: Dummy Embeddings (Testing)

For testing pipeline without Gemma:

```bash
python scripts/generate.py "A cat walking" --no-gemma --height 128 --width 128
```

### Downloading Gemma 3 12B

LTX-2 requires **Gemma 3 12B** (~25GB, loaded as FP16 = ~12GB in memory):

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
- **Text-to-video generation producing semantic content** (palm trees, sunsets, grass, etc.)
- Full text encoding pipeline (all 49 Gemma layers + normalization + projection)
- Native MLX Gemma 3 12B text encoder (FP16)
- 48-layer transformer (19B parameters) verified to match PyTorch exactly (correlation = 1.0)
- Video VAE decoder with timestep conditioning
- Audio VAE decoder (converts latents to mel spectrograms)
- HiFi-GAN vocoder (converts mel to 24kHz audio)
- CFG (Classifier-Free Guidance) with configurable scale
- Memory optimization with intermediate tensor cleanup
- Video export via ffmpeg
- Tested at resolutions up to 480x704

### In Progress
- FP8 weight loading (27GB quantized models)
- Spatial upscaler (2x resolution)
- Temporal upscaler (2x framerate)
- Full audio-video joint generation mode

### Pending
- Image-to-video conditioning
- LoRA support
- Video-to-video (IC-LoRA)

## Documentation

See the `docs/` folder for detailed documentation:

- [LTX-2 Architecture](docs/LTX-2-ARCHITECTURE.md) - Deep dive into model architecture
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions
- [Porting Guide](docs/PORTING.md) - Implementation status and details
- [PyTorch/MLX Differences](docs/PYTORCH_MLX_DIFFERENCES.md) - Framework comparison

## License

This project is for research and educational purposes. See the original [LTX-2](https://github.com/Lightricks/LTX-2) repository for model licensing.

## Acknowledgments

- [Lightricks](https://www.lightricks.com/) for LTX-2
- [Apple MLX Team](https://github.com/ml-explore/mlx) for the MLX framework
- [Google](https://ai.google.dev/gemma) for Gemma 3
