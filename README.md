# LTX-2-MLX

[![Model](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/Lightricks/LTX-2)
[![Original Repo](https://img.shields.io/badge/GitHub-LTX--2-181717?logo=github)](https://github.com/Lightricks/LTX-2)
[![Paper](https://img.shields.io/badge/Paper-PDF-EC1C24?logo=adobeacrobatreader&logoColor=white)](https://arxiv.org/abs/2601.03233)

Native Apple Silicon implementation of [Lightricks LTX-2](https://github.com/Lightricks/LTX-2), a 19B parameter video generation model, using [MLX](https://github.com/ml-explore/mlx).

<table>
<tr>
<td width="50%" align="center">

**"A golden retriever running through a sunny meadow"**

https://github.com/user-attachments/assets/59290fef-6900-460f-afe9-e24a40de01bb

</td>
<td width="50%" align="center">

**"A city street at night with neon lights and rain"**

https://github.com/user-attachments/assets/245e54ac-aff2-446f-a717-8797d0cc50c0

</td>
</tr>
<tr>
<td width="50%" align="center">

**"A rocket ship launching into space with flames"**

https://github.com/user-attachments/assets/f84284da-ea6d-4ce3-9b09-3ad773d7edc2

</td>
<td width="50%" align="center">

**"Ocean waves crashing on a beach at sunset"**

https://github.com/user-attachments/assets/a508dc05-6d02-453e-9ee9-07878613a137

</td>
</tr>
</table>

*768×512, 65 frames (~2.7s at 24fps), 8 steps on Apple Silicon*

## Quick Start

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Download weights
uv run scripts/download_weights.py

# 3. Generate video
uv run python scripts/generate.py "A golden retriever running through a meadow"
```

## Required Models

Download from [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2) on HuggingFace:

| Model | Size | Description |
|-------|------|-------------|
| [`ltx-2-19b-distilled.safetensors`](https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled.safetensors) | 43GB | Fast generation (8 steps) - **Recommended** |
| [`ltx-2-19b-distilled-fp8.safetensors`](https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled-fp8.safetensors) | 27GB | FP8 quantized (smaller download) |
| [`ltx-2-19b-dev.safetensors`](https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-dev.safetensors) | 43GB | Higher quality (25-50 steps) |
| [`ltx-2-spatial-upscaler-x2-1.0.safetensors`](https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors) | 995MB | 2x resolution upscaling |
| [`ltx-2-temporal-upscaler-x2-1.0.safetensors`](https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-temporal-upscaler-x2-1.0.safetensors) | 262MB | 2x framerate upscaling |
| [`ltx-2-19b-distilled-lora-384.safetensors`](https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled-lora-384.safetensors) | 1.5GB | LoRA for two-stage refinement |

**Text Encoder**: [Gemma 3 12B](https://huggingface.co/google/gemma-3-12b-it) (~25GB) - Requires accepting license

Or use the interactive downloader:
```bash
uv run scripts/download_weights.py --weights all
```

## Available Pipelines

| Pipeline | Speed | Quality | Best For |
|----------|-------|---------|----------|
| `text-to-video` | Medium | Good | Basic generation |
| `distilled` | **Fast** | Good | Quick iteration |
| `one-stage` | Slow | **High** | Quality priority |
| `two-stage` | Medium | **High** | High resolution (512p+) |

```bash
# Fast preview
python scripts/generate.py "Your prompt" --pipeline distilled

# High quality
python scripts/generate.py "Your prompt" --pipeline one-stage --steps 20 --cfg 5.0

# High resolution
python scripts/generate.py "Your prompt" --pipeline two-stage --height 768 --width 1024
```

See [Pipelines Guide](docs/PIPELINES.md) for all 6 pipelines and options.

## Optimization Tips

- **Use `--fp16`** - Reduces memory ~50% (enabled by default)
- **Use `--pipeline distilled`** - Fastest inference (8 steps)
- **Use `--low-memory`** - For systems with <32GB RAM
- **Reduce resolution** - Start with `--height 256 --width 384` for testing

See [Usage Guide](docs/USAGE.md) for memory requirements and benchmarks.

## Prompting Tips

Focus on detailed, chronological descriptions. Include movements, appearances, camera angles, and environment details in a flowing paragraph. Keep under 200 words.

**Structure your prompts:**
1. Main action in a single sentence
2. Specific movements and gestures
3. Character/object appearances
4. Background and environment
5. Camera angles and movements
6. Lighting and colors

See [Lightricks prompting guide](https://ltx.video/blog/how-to-prompt-for-ltx-2) for more tips.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- ~25GB RAM (128GB recommended for high resolution)
- ffmpeg: `brew install ffmpeg`

## Documentation

- [Usage Guide](docs/USAGE.md) - Options, examples, troubleshooting
- [Pipelines](docs/PIPELINES.md) - All 6 pipelines explained
- [Architecture](docs/ARCHITECTURE.md) - Model architecture details
- [Parity Testing](docs/PARITY.md) - PyTorch/MLX verification (97%+ correlation)
- [Technical Report](https://arxiv.org/abs/2601.03233) - Official Lightricks paper

## License

Research and educational use. See [LTX-2](https://github.com/Lightricks/LTX-2) for model licensing.

## Acknowledgments

- [Lightricks](https://www.lightricks.com/) for LTX-2
- [Apple MLX Team](https://github.com/ml-explore/mlx) for MLX
- [Google](https://ai.google.dev/gemma) for Gemma 3
