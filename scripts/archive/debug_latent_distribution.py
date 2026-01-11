#!/usr/bin/env python3
"""
Debug the latent distribution mismatch between denoising and VAE.

Key questions:
1. What distribution does the denoising produce?
2. What distribution does the VAE decoder expect?
3. How should we transform one to match the other?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np

def main():
    from LTX_2_MLX.model.video_vae.simple_decoder import (
        SimpleVideoDecoder,
        load_vae_decoder_weights,
    )

    weights_path = "weights/ltx-2/ltx-2-19b-distilled.safetensors"

    # Load decoder to get statistics
    print("Loading VAE decoder...")
    decoder = SimpleVideoDecoder(compute_dtype=mx.float32)
    load_vae_decoder_weights(decoder, weights_path)
    mx.eval(decoder.parameters())

    # Get per-channel statistics
    mean_of_means = np.array(decoder.mean_of_means)
    std_of_means = np.array(decoder.std_of_means)

    print("\n" + "="*70)
    print("VAE Per-Channel Statistics")
    print("="*70)
    print(f"mean_of_means: range [{mean_of_means.min():.4f}, {mean_of_means.max():.4f}]")
    print(f"               mean: {mean_of_means.mean():.4f}, std: {mean_of_means.std():.4f}")
    print(f"std_of_means:  range [{std_of_means.min():.4f}, {std_of_means.max():.4f}]")
    print(f"               mean: {std_of_means.mean():.4f}, std: {std_of_means.std():.4f}")

    # The VAE encoder normalizes like this:
    # normalized = (latent - mean_of_means) / std_of_means
    #
    # So the normalized latent (which diffusion operates on) should have:
    # - Per-channel mean ≈ 0
    # - Per-channel std ≈ 1
    #
    # And the decoder denormalizes:
    # latent = normalized * std_of_means + mean_of_means

    print("\n" + "="*70)
    print("Expected Normalized Latent Statistics (what denoising should produce)")
    print("="*70)
    print("Per-channel mean: 0 (because encoder subtracts mean_of_means)")
    print("Per-channel std:  1 (because encoder divides by std_of_means)")
    print("Overall range:    approximately [-3, 3] (3-sigma range)")

    # Load a saved latent from denoising if available
    latent_files = list(Path(".").glob("debug_latent*.npz")) + list(Path("gens").glob("*_latent.npz"))

    if latent_files:
        print("\n" + "="*70)
        print("Analyzing saved latent files")
        print("="*70)

        for latent_file in latent_files[:3]:  # Analyze up to 3 files
            print(f"\n--- {latent_file} ---")
            data = np.load(latent_file)
            if 'latent' in data:
                latent = data['latent']
                print(f"Shape: {latent.shape}")
                print(f"Overall: mean={latent.mean():.4f}, std={latent.std():.4f}")
                print(f"         range=[{latent.min():.4f}, {latent.max():.4f}]")

                # Per-channel statistics
                if latent.ndim == 5:  # [B, C, F, H, W]
                    latent = latent[0]  # Remove batch

                # [C, F, H, W] -> compute per-channel stats
                channel_means = latent.mean(axis=(1, 2, 3))  # [C]
                channel_stds = latent.std(axis=(1, 2, 3))    # [C]

                print(f"Channel means: range=[{channel_means.min():.4f}, {channel_means.max():.4f}]")
                print(f"               mean of means: {channel_means.mean():.4f}")
                print(f"Channel stds:  range=[{channel_stds.min():.4f}, {channel_stds.max():.4f}]")
                print(f"               mean of stds: {channel_stds.mean():.4f}")

                # Compare to expected
                mean_error = np.abs(channel_means).mean()
                std_error = np.abs(channel_stds - 1.0).mean()
                print(f"Mean error (vs 0): {mean_error:.4f}")
                print(f"Std error (vs 1):  {std_error:.4f}")

    # Test what happens with different input distributions
    print("\n" + "="*70)
    print("Testing decoder with different input distributions")
    print("="*70)

    # Test parameters
    frames, height, width = 3, 8, 12  # Small test size

    def test_latent(name, latent):
        """Test decoder with a given latent and report statistics."""
        print(f"\n--- {name} ---")
        print(f"Input: mean={float(mx.mean(latent)):.4f}, std={float(mx.std(latent)):.4f}")

        # Run through decoder
        output = decoder(latent, timestep=0.05, show_progress=False)
        mx.eval(output)

        print(f"Output: mean={float(mx.mean(output)):.4f}, std={float(mx.std(output)):.4f}")
        print(f"        range=[{float(mx.min(output)):.4f}, {float(mx.max(output)):.4f}]")

        # Check if output has spatial structure
        output_np = np.array(output[0])  # [C, F, H, W]
        frame_0 = output_np[:, 0, :, :]  # [C, H, W]

        # Check variance across spatial dimensions
        spatial_var = frame_0.var(axis=(1, 2))  # Per-channel spatial variance
        print(f"Spatial variance: mean={spatial_var.mean():.4f}, range=[{spatial_var.min():.4f}, {spatial_var.max():.4f}]")

    # 1. Zero latent (mean=0, std=0)
    latent = mx.zeros((1, 128, frames, height, width))
    test_latent("Zero latent", latent)

    # 2. Standard normal (mean=0, std=1) - what encoder should produce
    mx.random.seed(42)
    latent = mx.random.normal((1, 128, frames, height, width))
    test_latent("Standard normal (μ=0, σ=1)", latent)

    # 3. Shifted normal (mean=1, std=1)
    latent = mx.random.normal((1, 128, frames, height, width)) + 1.0
    test_latent("Shifted normal (μ=1, σ=1)", latent)

    # 4. Scaled normal (mean=0, std=2)
    latent = mx.random.normal((1, 128, frames, height, width)) * 2.0
    test_latent("Scaled normal (μ=0, σ=2)", latent)

    # 5. Random with per-channel biases (simulating denoising output)
    latent = mx.random.normal((1, 128, frames, height, width))
    # Add random per-channel bias
    channel_biases = mx.random.uniform(-2, 2, (128,))
    latent = latent + channel_biases[None, :, None, None, None]
    test_latent("Normal + per-channel bias (μ~[-2,2])", latent)

    # 6. Test with structure: gradient across space
    print("\n--- Structured latent (spatial gradient) ---")
    base = mx.random.normal((1, 128, frames, height, width)) * 0.5
    # Add spatial gradient
    x_coords = mx.linspace(-1, 1, width)[None, None, None, None, :]
    y_coords = mx.linspace(-1, 1, height)[None, None, None, :, None]
    spatial = x_coords + y_coords  # Diagonal gradient
    latent = base + spatial * 0.5
    test_latent("Normal + spatial gradient", latent)

    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print("""
The VAE decoder expects normalized latent (what the encoder produces):
- Per-channel mean ≈ 0
- Per-channel std ≈ 1

The decoder denormalizes: output = input * std_of_means + mean_of_means

If denoising produces latent with:
- Per-channel mean ≈ 0, std ≈ 1: Good, decoder will work correctly
- Per-channel mean ≈ non-zero: Bad, decoder output will be shifted
- Per-channel std ≈ 2+: May cause saturation after decoder

Key insight: The decoder's denormalization EXPECTS zero-mean, unit-variance input.
Any preprocessing we do should ensure we match this distribution.
    """)

if __name__ == "__main__":
    main()
