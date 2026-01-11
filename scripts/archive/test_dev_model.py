#!/usr/bin/env python3
"""Test the dev (non-distilled) model with LTX2Scheduler.

The dev model should use more steps (20-50) with the proper shifted sigma schedule.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np
from tqdm import tqdm

from LTX_2_MLX.model.transformer import LTXModel, LTXModelType, Modality, X0Model
from LTX_2_MLX.model.video_vae.simple_decoder import (
    SimpleVideoDecoder,
    load_vae_decoder_weights,
    decode_latent,
)
from LTX_2_MLX.loader import load_transformer_weights
from LTX_2_MLX.components import VideoLatentPatchifier, LTX2Scheduler
from LTX_2_MLX.components.patchifiers import get_pixel_coords
from LTX_2_MLX.types import VideoLatentShape, SpatioTemporalScaleFactors
from LTX_2_MLX.core_utils import to_velocity

from scripts.generate import encode_with_gemma


def main():
    # Paths - use DEV model, not distilled
    weights_path = "weights/ltx-2/ltx-2-19b-dev.safetensors"
    gemma_path = "weights/gemma-3-12b"

    # Config
    height, width, frames = 256, 384, 17
    latent_h, latent_w, latent_f = height // 32, width // 32, (frames - 1) // 8 + 1
    num_steps = 25  # More steps for dev model
    cfg_scale = 3.0  # Use CFG with dev model

    print(f"Resolution: {height}x{width}, Frames: {frames}")
    print(f"Latent: {latent_h}x{latent_w}x{latent_f}")
    print(f"Steps: {num_steps}, CFG: {cfg_scale}")

    # Load text encoding
    print("\n" + "=" * 60)
    print("Encoding text...")
    print("=" * 60)
    prompt = "A blue ball bouncing on green grass"
    text_encoding, text_mask = encode_with_gemma(
        prompt,
        gemma_path=gemma_path,
        ltx_weights_path=weights_path,
    )
    mx.eval(text_encoding, text_mask)
    print(f"Text encoding shape: {text_encoding.shape}")

    # Load transformer (dev model)
    print("\n" + "=" * 60)
    print("Loading dev transformer...")
    print("=" * 60)
    model = LTXModel(
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
    load_transformer_weights(model, weights_path)
    x0_model = X0Model(model)

    # Load VAE decoder
    print("\n" + "=" * 60)
    print("Loading VAE decoder...")
    print("=" * 60)
    decoder = SimpleVideoDecoder(compute_dtype=mx.float32)
    load_vae_decoder_weights(decoder, weights_path)

    # Setup
    mx.random.seed(42)
    patchifier = VideoLatentPatchifier(patch_size=1)
    output_shape = VideoLatentShape(
        batch=1, channels=128, frames=latent_f, height=latent_h, width=latent_w
    )

    # Get positions
    latent_coords = patchifier.get_patch_grid_bounds(output_shape=output_shape)
    scale_factors = SpatioTemporalScaleFactors.default()
    positions = get_pixel_coords(
        latent_coords=latent_coords,
        scale_factors=scale_factors,
        causal_fix=True,
    ).astype(mx.float32)
    fps = 24.0
    temporal_positions = positions[:, 0:1, ...] / fps
    other_positions = positions[:, 1:, ...]
    positions = mx.concatenate([temporal_positions, other_positions], axis=1)

    # Get sigma schedule using LTX2Scheduler
    print("\n" + "=" * 60)
    print("Computing sigma schedule...")
    print("=" * 60)
    scheduler = LTX2Scheduler()
    # Create a dummy latent to compute tokens
    dummy_latent = mx.zeros((1, 128, latent_f, latent_h, latent_w))
    sigmas = scheduler.execute(steps=num_steps, latent=dummy_latent)
    mx.eval(sigmas)

    print(f"Sigma schedule ({len(sigmas)} values):")
    sigmas_list = [float(s) for s in sigmas]
    for i, s in enumerate(sigmas_list):
        print(f"  Step {i}: sigma = {s:.6f}")

    # Initialize with noise
    latent = mx.random.normal((1, 128, latent_f, latent_h, latent_w))

    # Null embeddings for unconditional
    null_embedding = mx.zeros_like(text_encoding)
    null_mask = mx.zeros_like(text_mask)

    # Denoise loop
    print("\n" + "=" * 60)
    print(f"Denoising ({num_steps} steps)...")
    print("=" * 60)

    for step_idx in tqdm(range(num_steps), desc="Denoising"):
        sigma = float(sigmas[step_idx])
        sigma_next = float(sigmas[step_idx + 1])
        dt = sigma_next - sigma

        # Patchify current latent
        latent_patchified = patchifier.patchify(latent)

        # Conditional pass
        modality_cond = Modality(
            latent=latent_patchified,
            context=text_encoding,
            context_mask=text_mask,
            timesteps=mx.array([sigma]),
            positions=positions,
            enabled=True,
        )
        x0_cond = x0_model(modality_cond)
        mx.eval(x0_cond)

        # Unconditional pass
        modality_uncond = Modality(
            latent=latent_patchified,
            context=null_embedding,
            context_mask=null_mask,
            timesteps=mx.array([sigma]),
            positions=positions,
            enabled=True,
        )
        x0_uncond = x0_model(modality_uncond)
        mx.eval(x0_uncond)

        # CFG
        x0 = x0_uncond + cfg_scale * (x0_cond - x0_uncond)

        # Unpatchify
        x0_unpatch = patchifier.unpatchify(x0, output_shape=output_shape)

        # Compute velocity and step
        velocity = to_velocity(latent, sigma, x0_unpatch)
        latent = latent + velocity * dt

        mx.eval(latent)

    print(f"\nFinal latent shape: {latent.shape}")
    print(f"Final latent stats: mean={float(mx.mean(latent)):.4f}, std={float(mx.std(latent)):.4f}")

    # Decode
    print("\n" + "=" * 60)
    print("Decoding to video...")
    print("=" * 60)
    video = decode_latent(latent, decoder)
    mx.eval(video)

    # Save
    from PIL import Image
    video_np = np.array(video)
    print(f"Video shape: {video_np.shape}, range: [{video_np.min():.4f}, {video_np.max():.4f}]")

    if video_np.ndim == 5:
        video_np = video_np[0]

    if video_np.max() <= 1.0:
        video_np = (video_np * 255).clip(0, 255).astype(np.uint8)
    else:
        video_np = video_np.clip(0, 255).astype(np.uint8)

    frame_path = Path("gens/dev_model_frame0.png")
    frame_path.parent.mkdir(exist_ok=True)
    Image.fromarray(video_np[0]).save(frame_path)
    print(f"Saved first frame to: {frame_path}")


if __name__ == "__main__":
    main()
