#!/usr/bin/env python3
"""Test the proper two-stage distilled pipeline with spatial upscaler.

The distilled model is designed for:
  Stage 1: Generate at half resolution (7 steps)
  Stage 2: Upsample 2x and refine (3 steps)

This is different from single-stage generation which we've been testing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np
from tqdm import tqdm

# Import components
from LTX_2_MLX.model.transformer import LTXModel, LTXModelType
from LTX_2_MLX.model.video_vae.simple_decoder import (
    SimpleVideoDecoder,
    load_vae_decoder_weights,
)
from LTX_2_MLX.model.video_vae.simple_encoder import (
    SimpleVideoEncoder,
    load_vae_encoder_weights,
)
from LTX_2_MLX.model.upscaler import SpatialUpscaler, load_spatial_upscaler_weights
from LTX_2_MLX.loader import load_transformer_weights
from LTX_2_MLX.pipelines.distilled import (
    DistilledPipeline,
    DistilledConfig,
)

# Import text encoding
from scripts.generate import encode_with_gemma


def main():
    # Paths
    weights_path = "weights/ltx-2/ltx-2-19b-distilled.safetensors"
    upscaler_path = "weights/ltx-2/ltx-2-spatial-upscaler-x2-1.0.safetensors"
    gemma_path = "weights/gemma-3-12b"

    # Config - use smaller resolution for testing
    # Output will be 256x384, so stage 1 will be 128x192
    config = DistilledConfig(
        height=256,  # Must be divisible by 64
        width=384,   # Must be divisible by 64
        num_frames=17,  # 8k+1 = 17 for k=2
        seed=42,
        dtype=mx.float16,  # Use FP16 for memory
    )

    print(f"Output resolution: {config.height}x{config.width}")
    print(f"Stage 1 resolution: {config.height // 2}x{config.width // 2}")
    print(f"Frames: {config.num_frames}")

    # Load text encoding
    print("\n" + "=" * 60)
    print("Loading text encoder and encoding prompt...")
    print("=" * 60)
    prompt = "A blue ball bouncing on green grass"
    text_encoding, text_mask = encode_with_gemma(
        prompt,
        gemma_path=gemma_path,
        ltx_weights_path=weights_path,
    )
    mx.eval(text_encoding, text_mask)
    print(f"Text encoding shape: {text_encoding.shape}")

    # Load transformer
    print("\n" + "=" * 60)
    print("Loading transformer...")
    print("=" * 60)
    transformer = LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=32,
        attention_head_dim=128,
        in_channels=128,
        out_channels=128,
        num_layers=48,
        cross_attention_dim=4096,
        caption_channels=3840,
        compute_dtype=mx.float16,
    )
    load_transformer_weights(transformer, weights_path)

    # Load VAE decoder
    print("\n" + "=" * 60)
    print("Loading VAE encoder and decoder...")
    print("=" * 60)
    video_encoder = SimpleVideoEncoder(compute_dtype=mx.float32)
    load_vae_encoder_weights(video_encoder, weights_path)

    video_decoder = SimpleVideoDecoder(compute_dtype=mx.float32)
    load_vae_decoder_weights(video_decoder, weights_path)

    # Load spatial upscaler
    print("\n" + "=" * 60)
    print("Loading spatial upscaler...")
    print("=" * 60)
    spatial_upscaler = SpatialUpscaler()
    load_spatial_upscaler_weights(spatial_upscaler, upscaler_path)

    print("\n" + "=" * 60)
    print("Creating distilled pipeline...")
    print("=" * 60)

    pipeline = DistilledPipeline(
        transformer=transformer,
        video_encoder=video_encoder,
        video_decoder=video_decoder,
        spatial_upscaler=spatial_upscaler,
    )

    # Progress callback
    pbar = None
    current_stage = ""

    def progress_callback(stage: str, step: int, total: int):
        nonlocal pbar, current_stage
        if stage != current_stage:
            if pbar is not None:
                pbar.close()
            current_stage = stage
            stage_name = "Stage 1 (half-res)" if stage == "stage1" else "Stage 2 (full-res)"
            pbar = tqdm(total=total, desc=stage_name)
        pbar.update(1)

    print("\n" + "=" * 60)
    print("Running two-stage distilled generation...")
    print("=" * 60)

    video = pipeline(
        text_encoding=text_encoding,
        text_mask=text_mask,
        config=config,
        callback=progress_callback,
    )

    if pbar is not None:
        pbar.close()

    mx.eval(video)
    print(f"\nGenerated video shape: {video.shape}")

    # Convert video to frames
    from PIL import Image
    video_np = np.array(video)
    print(f"Video array shape: {video_np.shape}, dtype: {video_np.dtype}")
    print(f"Video range: [{video_np.min():.4f}, {video_np.max():.4f}]")

    # Handle different possible shapes
    if video_np.ndim == 5:  # [B, F, H, W, C]
        video_np = video_np[0]  # [F, H, W, C]
    elif video_np.ndim == 4 and video_np.shape[-1] != 3:  # [F, H, W, C] but wrong order
        # Might be [B, C, H, W] or similar - handle transpose
        pass

    # Ensure [0, 255] range
    if video_np.max() <= 1.0:
        video_np = (video_np * 255).clip(0, 255).astype(np.uint8)
    else:
        video_np = video_np.clip(0, 255).astype(np.uint8)

    # Save first frame as PNG
    frame_path = Path("gens/distilled_twostage_frame0.png")
    frame_path.parent.mkdir(exist_ok=True)
    frame = video_np[0] if video_np.ndim == 4 else video_np
    Image.fromarray(frame).save(frame_path)
    print(f"Saved first frame to: {frame_path}")

    # Save video using ffmpeg
    import subprocess
    import tempfile
    output_path = Path("gens/distilled_twostage_test.mp4")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save frames as PNG
        for i, f in enumerate(video_np):
            Image.fromarray(f).save(f"{tmpdir}/frame_{i:04d}.png")

        # Use ffmpeg to create video
        cmd = [
            "ffmpeg", "-y",
            "-framerate", "24",
            "-i", f"{tmpdir}/frame_%04d.png",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        subprocess.run(cmd, capture_output=True)

    print(f"Saved video to: {output_path}")


if __name__ == "__main__":
    main()
