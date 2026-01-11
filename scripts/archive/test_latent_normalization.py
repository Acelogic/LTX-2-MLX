#!/usr/bin/env python3
"""
Test if normalizing denoised latent to match encoder statistics fixes output.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np
from PIL import Image

def main():
    from LTX_2_MLX.model.video_vae.simple_decoder import (
        SimpleVideoDecoder,
        load_vae_decoder_weights,
    )

    weights_path = "weights/ltx-2/ltx-2-19b-distilled.safetensors"

    print("Loading VAE decoder...")
    decoder = SimpleVideoDecoder(compute_dtype=mx.float32)
    load_vae_decoder_weights(decoder, weights_path)
    mx.eval(decoder.parameters())

    # Load the encoder roundtrip latent (what correct latent looks like)
    encoder_latent_data = np.load("gens/vae_roundtrip_latent.npz")
    encoder_channel_means = encoder_latent_data["channel_means"]
    encoder_channel_stds = encoder_latent_data["channel_stds"]

    print(f"\nEncoder latent statistics (target):")
    print(f"  Channel means: range=[{encoder_channel_means.min():.4f}, {encoder_channel_means.max():.4f}], mean={encoder_channel_means.mean():.4f}")
    print(f"  Channel stds:  range=[{encoder_channel_stds.min():.4f}, {encoder_channel_stds.max():.4f}], mean={encoder_channel_stds.mean():.4f}")

    # Find a denoised latent to test
    latent_files = list(Path("gens").glob("*_latent.npz"))
    if not latent_files:
        print("No latent files found in gens/")
        return

    for latent_file in latent_files[:2]:
        if "roundtrip" in str(latent_file):
            continue  # Skip the encoder roundtrip

        print(f"\n{'='*70}")
        print(f"Processing: {latent_file}")
        print(f"{'='*70}")

        data = np.load(latent_file)
        if "latent" not in data:
            continue

        latent_np = data["latent"]
        if latent_np.ndim == 5:
            latent_np = latent_np[0]  # Remove batch

        print(f"Original latent shape: {latent_np.shape}")

        # Current statistics
        channel_means = latent_np.mean(axis=(1, 2, 3))
        channel_stds = latent_np.std(axis=(1, 2, 3))

        print(f"\nOriginal denoised latent statistics:")
        print(f"  Channel means: range=[{channel_means.min():.4f}, {channel_means.max():.4f}], mean={channel_means.mean():.4f}")
        print(f"  Channel stds:  range=[{channel_stds.min():.4f}, {channel_stds.max():.4f}], mean={channel_stds.mean():.4f}")

        # Strategy 1: Just center per-channel (current approach)
        latent_centered = latent_np - channel_means[:, None, None, None]
        latent_centered_mx = mx.array(latent_centered[None])

        # Strategy 2: Full normalization to match encoder distribution
        # For each channel: (x - mean) / std * encoder_std + encoder_mean
        latent_normalized = np.zeros_like(latent_np)
        for c in range(128):
            ch = latent_np[c]
            ch = (ch - channel_means[c]) / (channel_stds[c] + 1e-8)  # Standardize to N(0,1)
            ch = ch * encoder_channel_stds[c] + encoder_channel_means[c]  # Match encoder
            latent_normalized[c] = ch
        latent_normalized_mx = mx.array(latent_normalized[None])

        # Strategy 3: Just use the original (for comparison)
        latent_original_mx = mx.array(latent_np[None])

        # Test each strategy
        print(f"\n--- Testing decode strategies ---")

        for name, latent in [
            ("Original (no modification)", latent_original_mx),
            ("Centered (remove per-channel mean)", latent_centered_mx),
            ("Normalized (match encoder distribution)", latent_normalized_mx),
        ]:
            # Decode
            output = decoder(latent, timestep=0.05, show_progress=False)
            mx.eval(output)

            # Convert to numpy
            output_np = np.array(output[0])  # [3, F, H, W]
            output_np = output_np.transpose(1, 2, 3, 0)  # [F, H, W, 3]

            # Scale from [-1, 1] to [0, 1]
            output_np = (output_np + 1) / 2
            output_np = np.clip(output_np, 0, 1)

            print(f"\n{name}:")
            print(f"  Output mean: {output_np.mean():.4f}")
            print(f"  Output std:  {output_np.std():.4f}")
            print(f"  Output range: [{output_np.min():.4f}, {output_np.max():.4f}]")

            # Check RGB variance
            frame = output_np[0]  # [H, W, 3]
            rgb_std = frame.std(axis=(0, 1))  # Per-channel std
            print(f"  RGB stds: R={rgb_std[0]:.4f}, G={rgb_std[1]:.4f}, B={rgb_std[2]:.4f}")

            # Save the frame
            output_dir = Path("gens")
            frame_uint8 = (frame * 255).astype(np.uint8)
            short_name = name.split("(")[0].strip().lower().replace(" ", "_")
            output_path = output_dir / f"test_norm_{latent_file.stem}_{short_name}.png"
            Image.fromarray(frame_uint8).save(output_path)
            print(f"  Saved: {output_path}")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print("""
If 'Normalized' produces better output than 'Original', it confirms
that the denoising is producing incorrect per-channel statistics.

The fix would be to either:
1. Fix the denoising to produce correct statistics
2. Normalize the output before VAE decoding
    """)

if __name__ == "__main__":
    main()
