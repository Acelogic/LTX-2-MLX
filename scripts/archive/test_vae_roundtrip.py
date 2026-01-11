#!/usr/bin/env python3
"""
Test VAE encoder-decoder roundtrip with synthetic video.
This verifies the VAE works correctly and shows what latent distribution we should expect.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np
from PIL import Image

def create_test_video(height, width, frames, pattern="gradient"):
    """Create a synthetic test video with known patterns."""
    video = np.zeros((frames, height, width, 3), dtype=np.float32)

    if pattern == "gradient":
        # Create a video with spatial gradient
        for f in range(frames):
            for y in range(height):
                for x in range(width):
                    # Diagonal gradient from black to white
                    val = (x / width + y / height) / 2
                    # Add temporal variation
                    val = (val + f / frames) % 1.0
                    video[f, y, x] = [val, val, val]

    elif pattern == "colored_gradient":
        # RGB gradients
        for f in range(frames):
            for y in range(height):
                for x in range(width):
                    r = x / width  # Red increases left to right
                    g = y / height  # Green increases top to bottom
                    b = 0.5 + 0.5 * np.sin(2 * np.pi * f / frames)  # Blue pulses
                    video[f, y, x] = [r, g, b]

    elif pattern == "checkerboard":
        # Animated checkerboard
        for f in range(frames):
            for y in range(height):
                for x in range(width):
                    # Checkerboard with phase shift
                    check = ((x // 32) + (y // 32) + f) % 2
                    video[f, y, x] = [check, check, check]

    elif pattern == "uniform":
        # Uniform gray
        video[:] = 0.5

    return video  # [F, H, W, 3] in [0, 1]

def main():
    from LTX_2_MLX.model.video_vae.simple_encoder import (
        SimpleVideoEncoder,
        load_vae_encoder_weights,
    )
    from LTX_2_MLX.model.video_vae.simple_decoder import (
        SimpleVideoDecoder,
        load_vae_decoder_weights,
    )

    weights_path = "weights/ltx-2/ltx-2-19b-distilled.safetensors"

    # Test parameters
    height, width = 256, 384  # Must be divisible by 32
    frames = 17  # Must produce integer latent frames

    print("Loading VAE encoder and decoder...")
    encoder = SimpleVideoEncoder(compute_dtype=mx.float32)
    load_vae_encoder_weights(encoder, weights_path)

    decoder = SimpleVideoDecoder(compute_dtype=mx.float32)
    load_vae_decoder_weights(decoder, weights_path)

    mx.eval(encoder.parameters())
    mx.eval(decoder.parameters())

    # Create test video
    print(f"\nCreating test video: {frames}x{height}x{width}")
    video_np = create_test_video(height, width, frames, pattern="colored_gradient")
    print(f"Video range: [{video_np.min():.4f}, {video_np.max():.4f}]")
    print(f"Video mean: {video_np.mean():.4f}")

    # Convert to model format: [F, H, W, 3] -> [B, C, F, H, W]
    # and scale from [0, 1] to [-1, 1]
    video_scaled = (video_np * 2 - 1)  # [0,1] -> [-1,1]
    video_tensor = mx.array(video_scaled)  # [F, H, W, 3]
    video_tensor = video_tensor.transpose(3, 0, 1, 2)  # [3, F, H, W]
    video_tensor = video_tensor[None]  # [1, 3, F, H, W]

    print(f"Input tensor shape: {video_tensor.shape}")
    print(f"Input tensor range: [{float(mx.min(video_tensor)):.4f}, {float(mx.max(video_tensor)):.4f}]")

    # Encode (encoder doesn't use timestep)
    print("\nEncoding...")
    latent = encoder(video_tensor, show_progress=True)
    mx.eval(latent)

    print(f"\nLatent shape: {latent.shape}")
    print(f"Latent range: [{float(mx.min(latent)):.4f}, {float(mx.max(latent)):.4f}]")
    print(f"Latent mean: {float(mx.mean(latent)):.4f}, std: {float(mx.std(latent)):.4f}")

    # Per-channel statistics
    latent_np = np.array(latent[0])  # [128, F, H, W]
    channel_means = latent_np.mean(axis=(1, 2, 3))
    channel_stds = latent_np.std(axis=(1, 2, 3))

    print(f"\n{'='*70}")
    print("ENCODER OUTPUT (normalized latent) STATISTICS:")
    print(f"{'='*70}")
    print(f"Channel means: range=[{channel_means.min():.4f}, {channel_means.max():.4f}]")
    print(f"               mean={channel_means.mean():.4f}, std={channel_means.std():.4f}")
    print(f"Channel stds:  range=[{channel_stds.min():.4f}, {channel_stds.max():.4f}]")
    print(f"               mean={channel_stds.mean():.4f}, std={channel_stds.std():.4f}")

    # This is what denoising SHOULD produce!
    print("\n>>> These are the statistics that denoising SHOULD produce! <<<")

    # Decode
    print(f"\n{'='*70}")
    print("DECODING...")
    print(f"{'='*70}")

    # Decode with same timestep
    video_reconstructed = decoder(latent, timestep=0.05, show_progress=True)
    mx.eval(video_reconstructed)

    print(f"\nReconstructed shape: {video_reconstructed.shape}")
    print(f"Reconstructed range: [{float(mx.min(video_reconstructed)):.4f}, {float(mx.max(video_reconstructed)):.4f}]")

    # Convert to numpy: [B, 3, F, H, W] -> [F, H, W, 3]
    recon_np = np.array(video_reconstructed[0])  # [3, F, H, W]
    recon_np = recon_np.transpose(1, 2, 3, 0)  # [F, H, W, 3]

    # Scale back from [-1, 1] to [0, 1]
    recon_np = (recon_np + 1) / 2
    recon_np = np.clip(recon_np, 0, 1)

    # Calculate reconstruction error
    mse = np.mean((video_np - recon_np) ** 2)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse))

    print(f"\n{'='*70}")
    print("RECONSTRUCTION QUALITY:")
    print(f"{'='*70}")
    print(f"MSE: {mse:.6f}")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"Original range: [{video_np.min():.4f}, {video_np.max():.4f}]")
    print(f"Reconstructed range: [{recon_np.min():.4f}, {recon_np.max():.4f}]")

    # Check spatial structure
    print(f"\n{'='*70}")
    print("SPATIAL STRUCTURE:")
    print(f"{'='*70}")

    # Compare corners of first frame
    frame_orig = video_np[0]  # [H, W, 3]
    frame_recon = recon_np[0]

    print("Original corners (RGB):")
    print(f"  TL: {frame_orig[0, 0]}")
    print(f"  TR: {frame_orig[0, -1]}")
    print(f"  BL: {frame_orig[-1, 0]}")
    print(f"  BR: {frame_orig[-1, -1]}")

    print("\nReconstructed corners (RGB):")
    print(f"  TL: {frame_recon[0, 0]}")
    print(f"  TR: {frame_recon[0, -1]}")
    print(f"  BL: {frame_recon[-1, 0]}")
    print(f"  BR: {frame_recon[-1, -1]}")

    # Save comparison frames
    output_dir = Path("gens")
    output_dir.mkdir(exist_ok=True)

    print(f"\nSaving comparison frames to {output_dir}/...")

    # Save original
    orig_frame = (video_np[0] * 255).astype(np.uint8)
    Image.fromarray(orig_frame).save(output_dir / "vae_roundtrip_original.png")

    # Save reconstructed
    recon_frame = (recon_np[0] * 255).astype(np.uint8)
    Image.fromarray(recon_frame).save(output_dir / "vae_roundtrip_reconstructed.png")

    # Save latent stats
    np.savez(output_dir / "vae_roundtrip_latent.npz",
             latent=np.array(latent),
             channel_means=channel_means,
             channel_stds=channel_stds)

    print(f"\nSaved: vae_roundtrip_original.png, vae_roundtrip_reconstructed.png, vae_roundtrip_latent.npz")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY:")
    print(f"{'='*70}")

    if psnr > 25:
        print("VAE encoder-decoder roundtrip is WORKING CORRECTLY.")
        print("\nThe latent from encoder has:")
        print(f"  - Per-channel means: {channel_means.mean():.4f} (should be ~0)")
        print(f"  - Per-channel stds: {channel_stds.mean():.4f} (should be ~1)")
        print("\nIf denoising produces different statistics, that's the bug!")
    else:
        print("WARNING: VAE roundtrip quality is poor!")
        print("There may be issues with encoder or decoder.")

if __name__ == "__main__":
    main()
