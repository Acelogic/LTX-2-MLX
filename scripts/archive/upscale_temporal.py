#!/usr/bin/env python3
"""Apply 2x temporal upscaling to existing video.

This script takes an existing video, encodes it to latents, applies temporal
upscaling (2x framerate), and decodes back to video.

Example:
    python scripts/upscale_temporal.py gens/output.mp4 \
        --output gens/output_smooth.mp4 \
        --weights weights/ltx-2/ltx-2-19b-distilled.safetensors \
        --temporal-weights weights/ltx-2/ltx-2-temporal-upscaler-x2-1.0.safetensors \
        --fp16
"""

import argparse
import gc
import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from LTX_2_MLX.model.upscaler import (
    TemporalUpscaler,
    load_temporal_upscaler_weights,
)
from LTX_2_MLX.model.video_vae.simple_encoder import (
    SimpleVideoEncoder,
    load_vae_encoder_weights,
)
from LTX_2_MLX.model.video_vae.simple_decoder import (
    SimpleVideoDecoder,
    load_vae_decoder_weights,
    decode_latent,
)

# Try to import video I/O utilities
try:
    from LTX_2_MLX.utils.video_io import load_video, save_video
except ImportError:
    print("Warning: Could not import video_io utilities. Using placeholder functions.")

    def load_video(path: str, height: int = None, width: int = None):
        """Placeholder for loading video - implement based on your video I/O library."""
        raise NotImplementedError("Video loading not implemented. Please add video_io utilities.")

    def save_video(frames, path: str, fps: float = 24.0):
        """Placeholder for saving video - implement based on your video I/O library."""
        raise NotImplementedError("Video saving not implemented. Please add video_io utilities.")


def upscale_video_temporal(
    input_path: str,
    output_path: str,
    weights_path: str,
    temporal_weights_path: str,
    use_fp16: bool = True,
):
    """
    Apply 2x temporal upscaling to a video.

    Args:
        input_path: Path to input video file
        output_path: Path to output video file
        weights_path: Path to LTX-2 weights (for VAE encoder/decoder)
        temporal_weights_path: Path to temporal upscaler weights
        use_fp16: Use FP16 computation (recommended)
    """
    print(f"\n=== Temporal Upscaling 2x ===")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    # Set compute dtype
    compute_dtype = mx.float16 if use_fp16 else mx.float32
    dtype_name = "FP16" if use_fp16 else "FP32"

    # Load video
    print(f"\n[1/5] Loading input video...")
    video_frames = load_video(input_path)
    print(f"  Loaded {len(video_frames)} frames")

    # Convert to tensor [F, H, W, C] -> [1, C, F, H, W]
    video_np = np.stack(video_frames, axis=0)  # [F, H, W, C]
    video_np = video_np / 127.5 - 1.0  # Normalize to [-1, 1]
    video_mx = mx.array(video_np)
    video_mx = mx.transpose(video_mx, (3, 0, 1, 2))  # [C, F, H, W]
    video_mx = video_mx[None, ...]  # [1, C, F, H, W]
    video_mx = video_mx.astype(compute_dtype)

    print(f"  Video shape: {video_mx.shape}")

    # Load VAE encoder
    print(f"\n[2/5] Loading VAE encoder ({dtype_name})...")
    encoder = SimpleVideoEncoder(compute_dtype=compute_dtype)
    load_vae_encoder_weights(encoder, weights_path)

    # Encode to latents
    print(f"\n[3/5] Encoding video to latents...")
    latent = encoder(video_mx)
    mx.eval(latent)
    print(f"  Latent shape: {latent.shape}")

    # Clear encoder
    del encoder, video_mx
    gc.collect()
    mx.metal.clear_cache()

    # Load temporal upscaler
    print(f"\n[4/5] Loading temporal upscaler...")
    upscaler = TemporalUpscaler(compute_dtype=compute_dtype)
    load_temporal_upscaler_weights(upscaler, temporal_weights_path)

    # Apply temporal upscaling (F -> F*2)
    print(f"  Upscaling temporal dimension 2x...")
    upscaled_latent = upscaler(latent)
    mx.eval(upscaled_latent)
    print(f"  Upscaled latent shape: {upscaled_latent.shape}")

    # Clear upscaler
    del upscaler, latent
    gc.collect()
    mx.metal.clear_cache()

    # Load VAE decoder
    print(f"\n[5/5] Loading VAE decoder ({dtype_name})...")
    decoder = SimpleVideoDecoder(compute_dtype=compute_dtype)
    load_vae_decoder_weights(decoder, weights_path)

    # Decode to video
    print(f"  Decoding latents to video...")
    output_video = decode_latent(upscaled_latent, decoder)
    mx.eval(output_video)
    print(f"  Output video shape: {output_video.shape}")

    # Convert to frames
    output_frames = [np.array(output_video[f]) for f in range(output_video.shape[0])]
    print(f"  Generated {len(output_frames)} frames (2x original)")

    # Save video
    print(f"\nSaving video to {output_path}...")
    # Use 16fps for 2x temporal upscaling (original was 8fps)
    save_video(output_frames, output_path, fps=16.0)

    print(f"\nDone! Upscaled video saved to {output_path}")
    print(f"  Original frames: {len(video_frames)}")
    print(f"  Upscaled frames: {len(output_frames)} (2x)")


def main():
    parser = argparse.ArgumentParser(
        description="Apply 2x temporal upscaling to existing video"
    )
    parser.add_argument(
        "input_video",
        type=str,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output video file"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/ltx-2/ltx-2-19b-distilled.safetensors",
        help="Path to LTX-2 weights (for VAE encoder/decoder)"
    )
    parser.add_argument(
        "--temporal-weights",
        type=str,
        default="weights/ltx-2/ltx-2-temporal-upscaler-x2-1.0.safetensors",
        help="Path to temporal upscaler weights"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 computation (recommended for memory efficiency)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.input_video):
        print(f"Error: Input video not found: {args.input_video}")
        return

    if not os.path.exists(args.weights):
        print(f"Error: Weights not found: {args.weights}")
        return

    if not os.path.exists(args.temporal_weights):
        print(f"Error: Temporal upscaler weights not found: {args.temporal_weights}")
        return

    # Run upscaling
    upscale_video_temporal(
        input_path=args.input_video,
        output_path=args.output,
        weights_path=args.weights,
        temporal_weights_path=args.temporal_weights,
        use_fp16=args.fp16,
    )


if __name__ == "__main__":
    main()
