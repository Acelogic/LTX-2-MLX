#!/usr/bin/env python3
"""Generate video by interpolating between keyframe images using LTX-2 MLX.

This script provides keyframe interpolation - generating smooth video transitions
between provided images. It uses the KeyframeInterpolationPipeline with CFG guidance.

Example:
    python scripts/interpolate.py \
        --images frame1.png:0:0.95 frame2.png:96:0.95 \
        --prompt "A smooth transition between scenes" \
        --output interpolated.mp4
"""

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import List, Tuple

import mlx.core as mx
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from LTX_2_MLX.pipelines import (
    KeyframeInterpolationConfig,
    KeyframeInterpolationPipeline,
    Keyframe,
    create_keyframe_pipeline,
)
from LTX_2_MLX.model.transformer import LTXModel, LTXModelType
from LTX_2_MLX.model.video_vae.simple_decoder import SimpleVideoDecoder, load_vae_decoder_weights
from LTX_2_MLX.model.video_vae.simple_encoder import SimpleVideoEncoder, load_vae_encoder_weights
from LTX_2_MLX.model.video_vae.tiling import TilingConfig
from LTX_2_MLX.model.upscaler import SpatialUpscaler, load_spatial_upscaler_weights
from LTX_2_MLX.loader import load_transformer_weights
from LTX_2_MLX.model.text_encoder.gemma3 import Gemma3Config, Gemma3Model, load_gemma3_weights
from LTX_2_MLX.model.text_encoder.encoder import create_text_encoder, load_text_encoder_weights

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: Install tqdm for progress bars: pip install tqdm")


# LTX-2 system prompt for video generation
T2V_SYSTEM_PROMPT = """Describe the video in extreme detail, focusing on the visual content, without any introductory phrases."""


def load_tokenizer(model_path: str):
    """Load the Gemma tokenizer from HuggingFace transformers."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return tokenizer
    except ImportError:
        print("Error: transformers library required for tokenizer")
        print("Install with: pip install transformers")
        return None


def create_chat_prompt(user_prompt: str) -> str:
    """Create a chat-format prompt for Gemma 3."""
    chat = f"<bos><start_of_turn>user\n{T2V_SYSTEM_PROMPT}\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"
    return chat


def encode_with_gemma(
    prompt: str,
    gemma_path: str,
    ltx_weights_path: str,
    max_length: int = 256,
) -> Tuple[mx.array, mx.array]:
    """Encode a text prompt using Gemma 3 + LTX-2 text encoder pipeline."""
    print(f"  Loading tokenizer from {gemma_path}...")
    tokenizer = load_tokenizer(gemma_path)
    if tokenizer is None:
        return None, None

    tokenizer.padding_side = "right"

    print(f"  Loading Gemma 3 model...")
    config = Gemma3Config()
    gemma = Gemma3Model(config)
    load_gemma3_weights(gemma, gemma_path)

    print(f"  Loading text encoder projection...")
    text_encoder = create_text_encoder()
    load_text_encoder_weights(text_encoder, ltx_weights_path)

    chat_prompt = create_chat_prompt(prompt)

    print(f"  Tokenizing prompt...")
    encoding = tokenizer(
        chat_prompt,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    input_ids = mx.array(encoding["input_ids"])
    attention_mask = mx.array(encoding["attention_mask"])

    num_tokens = int(attention_mask.sum())
    print(f"  Token count: {num_tokens}/{max_length}")

    print(f"  Running Gemma 3 forward pass...")
    last_hidden, all_hidden_states = gemma(
        input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    mx.eval(last_hidden)

    if all_hidden_states is None:
        print("  Error: Gemma model did not return hidden states")
        return None, None

    print(f"  Processing through text encoder pipeline...")
    encoded = text_encoder.feature_extractor.extract_from_hidden_states(
        hidden_states=all_hidden_states,
        attention_mask=attention_mask,
        padding_side="right",
    )

    large_value = 1e9
    connector_mask = (attention_mask.astype(encoded.dtype) - 1) * large_value
    connector_mask = connector_mask.reshape(attention_mask.shape[0], 1, 1, attention_mask.shape[-1])

    encoded, output_mask = text_encoder.embeddings_connector(encoded, connector_mask)
    binary_mask = (output_mask.squeeze(1).squeeze(1) >= -0.5).astype(mx.int32)
    encoded = encoded * binary_mask[:, :, None]

    mx.eval(encoded)
    mx.eval(binary_mask)

    print(f"  Output embedding shape: {encoded.shape}")

    # Clear models from memory
    print(f"  Clearing text encoder from memory...")
    del gemma
    del text_encoder
    del all_hidden_states
    del last_hidden
    del tokenizer
    gc.collect()
    mx.metal.clear_cache()

    return encoded, binary_mask


def create_dummy_encoding(
    batch_size: int = 1,
    max_tokens: int = 256,
    embed_dim: int = 3840,
) -> Tuple[mx.array, mx.array]:
    """Create dummy text encoding for testing."""
    encoding = mx.random.normal(shape=(batch_size, max_tokens, embed_dim)) * 0.1
    mask = mx.ones((batch_size, max_tokens))
    return encoding, mask


def create_null_encoding(
    batch_size: int = 1,
    max_tokens: int = 256,
    embed_dim: int = 3840,
) -> Tuple[mx.array, mx.array]:
    """Create null encoding for CFG unconditional pass."""
    encoding = mx.zeros((batch_size, max_tokens, embed_dim))
    mask = mx.zeros((batch_size, max_tokens))
    return encoding, mask


def parse_image_arg(image_arg: str) -> Tuple[str, int, float]:
    """
    Parse an image argument in format: path:frame_index[:strength].

    Args:
        image_arg: String like "image.png:0:0.95" or "image.png:0"

    Returns:
        Tuple of (path, frame_index, strength)
    """
    parts = image_arg.split(":")
    if len(parts) < 2:
        raise ValueError(f"Invalid image format: {image_arg}. Expected path:frame_index[:strength]")

    path = parts[0]
    frame_index = int(parts[1])
    strength = float(parts[2]) if len(parts) > 2 else 0.95

    return path, frame_index, strength


def save_video(frames: list, output_path: str, fps: int = 24):
    """Save frames as video using ffmpeg."""
    import subprocess
    import tempfile
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmpdir:
        print("  Writing frames...")
        if HAS_TQDM:
            iterator = tqdm(enumerate(frames), desc="  Saving frames", total=len(frames), ncols=80)
        else:
            iterator = enumerate(frames)

        for i, frame in iterator:
            img = Image.fromarray(frame)
            img.save(os.path.join(tmpdir, f"frame_{i:04d}.png"))

        print("\n  Encoding video...")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(tmpdir, "frame_%04d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-loglevel", "error",
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  FFmpeg error: {result.stderr}")
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate video by interpolating between keyframe images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python scripts/interpolate.py \\
      --images frame1.png:0:0.95 frame2.png:96:0.95 \\
      --prompt "A smooth transition" \\
      --output interpolated.mp4

Image format: path:frame_index[:strength]
  - path: Path to the image file
  - frame_index: Target frame in output (0=first, num_frames-1=last)
  - strength: Optional conditioning strength (default 0.95)
"""
    )
    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        required=True,
        help="Keyframe images in format: path:frame_index[:strength]"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt describing the video transition"
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt for CFG guidance"
    )
    parser.add_argument("--height", type=int, default=480, help="Video height (must be divisible by 64)")
    parser.add_argument("--width", type=int, default=704, help="Video width (must be divisible by 64)")
    parser.add_argument("--frames", type=int, default=97, help="Number of frames (8k+1)")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps")
    parser.add_argument("--cfg", type=float, default=7.5, help="CFG guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="gens/interpolated.mp4", help="Output path")
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/ltx-2/ltx-2-19b-dev.safetensors",
        help="Path to LTX-2 weights"
    )
    parser.add_argument(
        "--upscaler-weights",
        type=str,
        default="weights/ltx-2/ltx-2-spatial-upscaler-x2-1.0.safetensors",
        help="Path to spatial upscaler weights"
    )
    parser.add_argument(
        "--gemma-path",
        type=str,
        default="weights/gemma-3-12b",
        help="Path to Gemma 3 weights directory"
    )
    parser.add_argument(
        "--no-gemma",
        action="store_true",
        help="Use dummy embeddings instead of Gemma (for testing)"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 computation for lower memory"
    )
    parser.add_argument(
        "--fp8",
        action="store_true",
        help="Load FP8-quantized weights"
    )
    parser.add_argument(
        "--tiled-vae",
        action="store_true",
        help="Use tiled VAE decoding for lower memory"
    )
    parser.add_argument(
        "--single-stage",
        action="store_true",
        help="Use single-stage generation (no upscaling)"
    )

    args = parser.parse_args()

    # Parse image arguments
    keyframes = []
    for img_arg in args.images:
        path, frame_idx, strength = parse_image_arg(img_arg)
        if not os.path.exists(path):
            print(f"Error: Image not found: {path}")
            return
        keyframes.append(Keyframe(image_path=path, frame_index=frame_idx, strength=strength))

    # Validate keyframes
    if len(keyframes) < 2:
        print("Error: At least 2 keyframes required for interpolation")
        return

    # Sort by frame index
    keyframes.sort(key=lambda k: k.frame_index)

    print(f"\n{'='*50}")
    print(f"LTX-2 MLX Keyframe Interpolation")
    print(f"{'='*50}")
    print(f"Keyframes:")
    for kf in keyframes:
        print(f"  - {kf.image_path} at frame {kf.frame_index} (strength={kf.strength})")
    print(f"Prompt: {args.prompt}")
    print(f"Resolution: {args.width}x{args.height}, {args.frames} frames")
    print(f"Steps: {args.steps}, CFG: {args.cfg}, Seed: {args.seed}")

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Determine dtype
    compute_dtype = mx.float16 if args.fp16 else mx.float32

    # Encode prompts
    print("\n[1/5] Encoding prompts...")
    if args.no_gemma:
        print("  Using DUMMY encoding (test mode)")
        pos_encoding, pos_mask = create_dummy_encoding()
        neg_encoding, neg_mask = create_null_encoding()
    else:
        if not os.path.exists(args.gemma_path):
            print(f"Error: Gemma weights not found at {args.gemma_path}")
            print("Use --no-gemma for testing or download Gemma weights")
            return

        pos_encoding, pos_mask = encode_with_gemma(
            prompt=args.prompt,
            gemma_path=args.gemma_path,
            ltx_weights_path=args.weights,
        )
        if pos_encoding is None:
            print("Error: Failed to encode prompt")
            return

        # Encode negative prompt
        if args.negative_prompt:
            neg_encoding, neg_mask = encode_with_gemma(
                prompt=args.negative_prompt,
                gemma_path=args.gemma_path,
                ltx_weights_path=args.weights,
                max_length=pos_encoding.shape[1],
            )
        else:
            neg_encoding, neg_mask = encode_with_gemma(
                prompt="",
                gemma_path=args.gemma_path,
                ltx_weights_path=args.weights,
                max_length=pos_encoding.shape[1],
            )

        if neg_encoding is None:
            neg_encoding, neg_mask = create_null_encoding(
                batch_size=1,
                max_tokens=pos_encoding.shape[1],
                embed_dim=pos_encoding.shape[2],
            )

    # Load transformer
    print("\n[2/5] Loading transformer...")
    transformer = LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=32,
        attention_head_dim=128,
        in_channels=128,
        out_channels=128,
        num_layers=48,
        cross_attention_dim=4096,
        caption_channels=3840,
        positional_embedding_theta=10000.0,
        compute_dtype=compute_dtype,
    )
    load_transformer_weights(transformer, args.weights, use_fp8=args.fp8)

    # Load VAE encoder and decoder
    print("\n[3/5] Loading VAE...")
    vae_encoder = SimpleVideoEncoder(compute_dtype=compute_dtype)
    load_vae_encoder_weights(vae_encoder, args.weights)

    vae_decoder = SimpleVideoDecoder(compute_dtype=compute_dtype)
    load_vae_decoder_weights(vae_decoder, args.weights)

    # Load spatial upscaler if using two-stage
    spatial_upscaler = None
    if not args.single_stage:
        print("\n[4/5] Loading spatial upscaler...")
        if os.path.exists(args.upscaler_weights):
            spatial_upscaler = SpatialUpscaler()
            load_spatial_upscaler_weights(spatial_upscaler, args.upscaler_weights)
        else:
            print(f"  Warning: Upscaler weights not found at {args.upscaler_weights}")
            print(f"  Falling back to single-stage generation")
            args.single_stage = True
    else:
        print("\n[4/5] Skipping upscaler (single-stage mode)")

    # Create pipeline
    pipeline = create_keyframe_pipeline(
        transformer=transformer,
        video_encoder=vae_encoder,
        video_decoder=vae_decoder,
        spatial_upscaler=spatial_upscaler,
    )

    # Create config
    tiling_config = TilingConfig.default() if args.tiled_vae else None
    config = KeyframeInterpolationConfig(
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        num_inference_steps=args.steps,
        cfg_scale=args.cfg,
        seed=args.seed,
        use_two_stage=not args.single_stage,
        tiling_config=tiling_config,
        dtype=compute_dtype,
    )

    # Progress callback
    def progress_callback(stage: str, step: int, total: int):
        print(f"\r  {stage}: {step}/{total}", end="", flush=True)
        if step == total:
            print()

    # Generate video
    print("\n[5/5] Generating video...")
    video = pipeline(
        text_encoding=pos_encoding,
        text_mask=pos_mask,
        negative_text_encoding=neg_encoding,
        negative_text_mask=neg_mask,
        keyframes=keyframes,
        config=config,
        callback=progress_callback,
    )
    mx.eval(video)

    print(f"\nVideo shape: {video.shape}")

    # Convert to frames
    frames = [np.array(video[f]) for f in range(video.shape[0])]
    print(f"Generated {len(frames)} frames at {frames[0].shape[:2]}")

    # Save video
    print(f"\nSaving video to {args.output}...")
    save_video(frames, args.output, fps=24)

    print(f"\nDone! Video saved to {args.output}")


if __name__ == "__main__":
    main()
