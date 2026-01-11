#!/usr/bin/env python3
"""Debug full denoising loop to trace the output evolution."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np

def main():
    from LTX_2_MLX.model.transformer import (
        LTXModel, X0Model, Modality, LTXModelType
    )
    from LTX_2_MLX.components import DISTILLED_SIGMA_VALUES, VideoLatentPatchifier
    from LTX_2_MLX.components.patchifiers import get_pixel_coords
    from LTX_2_MLX.types import VideoLatentShape, SpatioTemporalScaleFactors
    from LTX_2_MLX.loader import load_transformer_weights
    from LTX_2_MLX.core_utils import to_velocity

    # Small test case
    height, width, frames = 256, 384, 17
    latent_h, latent_w, latent_f = height // 32, width // 32, (frames - 1) // 8 + 1

    num_steps = 8  # Full distilled schedule (MUST be 8 to reach sigma=0.0!)

    print(f"Latent shape: [{latent_f}, {latent_h}, {latent_w}]")
    sigmas = DISTILLED_SIGMA_VALUES[:num_steps + 1]
    print(f"Using {len(sigmas) - 1} steps")
    print(f"Sigmas: {sigmas}")

    # Load model
    print("\nLoading transformer...")
    model = LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=32,
        attention_head_dim=128,
        in_channels=128,
        out_channels=128,
        num_layers=48,
        cross_attention_dim=4096,
        caption_channels=3840,
    )
    load_transformer_weights(model, "weights/ltx-2/ltx-2-19b-distilled.safetensors")
    x0_model = X0Model(model)

    # Create test inputs
    print("\nCreating test inputs...")
    mx.random.seed(42)

    # Random noise latent (pure noise at sigma=1.0)
    initial_latent = mx.random.normal(shape=(1, 128, latent_f, latent_h, latent_w))
    latent = initial_latent

    # "Distinctive" text embedding (simulating real text)
    # Use non-zero embedding to simulate actual text content
    text_embedding = mx.random.normal(shape=(1, 64, 3840)) * 0.5
    text_mask = mx.ones((1, 64))

    # Null embedding for CFG
    null_embedding = mx.zeros((1, 64, 3840))
    null_mask = mx.zeros((1, 64))

    # Setup
    patchifier = VideoLatentPatchifier(patch_size=1)
    output_shape = VideoLatentShape(
        batch=1, channels=128, frames=latent_f, height=latent_h, width=latent_w
    )
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

    cfg_scale = 3.0

    # Track statistics across steps
    step_stats = []

    print(f"\n{'='*70}")
    print(f"Running denoising loop (CFG scale = {cfg_scale})...")
    print(f"{'='*70}")

    print(f"\nStep | Sigma     | dt        | Latent Mean | Latent Std | Denoised Std | Corr(init)")
    print("-" * 85)

    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        dt = sigma_next - sigma

        # Patchify current latent
        latent_patchified = patchifier.patchify(latent)

        # Conditional pass
        modality_cond = Modality(
            latent=latent_patchified,
            context=text_embedding,
            context_mask=text_mask,
            timesteps=mx.array([sigma]),
            positions=positions,
            enabled=True,
        )
        denoised_cond = x0_model(modality_cond)

        # Unconditional pass
        modality_uncond = Modality(
            latent=latent_patchified,
            context=null_embedding,
            context_mask=null_mask,
            timesteps=mx.array([sigma]),
            positions=positions,
            enabled=True,
        )
        denoised_uncond = x0_model(modality_uncond)

        mx.eval(denoised_cond)
        mx.eval(denoised_uncond)

        # CFG
        denoised_patchified = denoised_uncond + cfg_scale * (denoised_cond - denoised_uncond)
        denoised = patchifier.unpatchify(denoised_patchified, output_shape=output_shape)

        # Euler step
        velocity = to_velocity(latent, sigma, denoised)
        latent = latent.astype(mx.float32) + velocity.astype(mx.float32) * dt
        mx.eval(latent)

        # Compute stats
        latent_np = np.array(latent)
        initial_np = np.array(initial_latent)
        denoised_np = np.array(denoised)

        corr_init = np.corrcoef(latent_np.flatten(), initial_np.flatten())[0, 1]

        stats = {
            "step": i,
            "sigma": float(sigma),
            "sigma_next": float(sigma_next),
            "dt": float(dt),
            "latent_mean": float(latent_np.mean()),
            "latent_std": float(latent_np.std()),
            "denoised_std": float(denoised_np.std()),
            "corr_with_initial": corr_init,
        }
        step_stats.append(stats)

        print(f"{i:4d} | {sigma:.6f} | {dt:+.6f} | {stats['latent_mean']:+.6f} | {stats['latent_std']:.6f} | {stats['denoised_std']:.6f} | {corr_init:.4f}")

    # Final analysis
    print(f"\n{'='*70}")
    print("Final Analysis")
    print(f"{'='*70}")

    final_latent = np.array(latent)
    initial_np = np.array(initial_latent)

    # Channel-wise statistics
    print("\nFinal latent channel statistics:")
    channel_means = final_latent[0].mean(axis=(1, 2, 3))  # Per channel mean
    channel_stds = final_latent[0].std(axis=(1, 2, 3))
    print(f"  Mean of channel means: {channel_means.mean():.6f}")
    print(f"  Std of channel means: {channel_means.std():.6f}")
    print(f"  Mean of channel stds: {channel_stds.mean():.6f}")
    print(f"  Std of channel stds: {channel_stds.std():.6f}")

    # Correlation with initial noise
    final_corr = np.corrcoef(final_latent.flatten(), initial_np.flatten())[0, 1]
    print(f"\nCorrelation with initial noise: {final_corr:.4f}")

    # Check for structure in final latent
    print("\nSpatial structure analysis:")
    # Check if there's spatial coherence (low-frequency content)
    frame_0 = final_latent[0, :, 0, :, :]  # First frame [128, H, W]
    frame_0_mean_per_pos = frame_0.mean(axis=0)  # [H, W]
    print(f"  Frame 0 spatial mean std: {frame_0_mean_per_pos.std():.6f}")
    print(f"  Frame 0 spatial mean range: [{frame_0_mean_per_pos.min():.4f}, {frame_0_mean_per_pos.max():.4f}]")

    # Save final latent for analysis
    print("\nSaving final latent to debug_latent.npz...")
    np.savez("debug_latent.npz",
             latent=final_latent,
             initial=initial_np,
             step_stats=step_stats)

    # Key metrics
    print(f"\n{'='*70}")
    print("KEY METRICS:")
    print(f"{'='*70}")
    print(f"  Initial latent std: {initial_np.std():.4f}")
    print(f"  Final latent std: {final_latent.std():.4f}")
    print(f"  Correlation with initial noise: {final_corr:.4f}")

    if final_corr > 0.5:
        print("\n  WARNING: High correlation with initial noise!")
        print("  The denoising may not be working properly.")
    elif final_latent.std() > 2.0:
        print("\n  WARNING: Final latent has very high variance!")
        print("  CFG may be amplifying noise instead of structure.")
    else:
        print("\n  Denoising appears to be working.")
        print("  Issue may be in VAE decoding or text embedding.")

if __name__ == "__main__":
    main()
