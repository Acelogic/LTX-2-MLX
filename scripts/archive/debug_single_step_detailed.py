#!/usr/bin/env python3
"""Debug single denoising step to trace the issue."""

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

    print(f"Latent shape: [{latent_f}, {latent_h}, {latent_w}]")
    print(f"Sigma schedule: {DISTILLED_SIGMA_VALUES}")

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

    # Random noise latent
    latent = mx.random.normal(shape=(1, 128, latent_f, latent_h, latent_w))

    # Random text embedding (to test if model responds to different embeddings)
    text_embedding = mx.random.normal(shape=(1, 64, 3840)) * 0.1
    text_mask = mx.ones((1, 64))

    # Patchify
    patchifier = VideoLatentPatchifier(patch_size=1)
    latent_patchified = patchifier.patchify(latent)
    print(f"Patchified shape: {latent_patchified.shape}")

    # Create position grid
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

    # Convert temporal to seconds
    fps = 24.0
    temporal_positions = positions[:, 0:1, ...] / fps
    other_positions = positions[:, 1:, ...]
    positions = mx.concatenate([temporal_positions, other_positions], axis=1)

    # Test at first sigma (highest noise)
    sigma = DISTILLED_SIGMA_VALUES[0]  # 1.0
    print(f"\n{'='*60}")
    print(f"Testing at sigma = {sigma}")
    print(f"{'='*60}")

    # Create modality
    modality = Modality(
        latent=latent_patchified,
        context=text_embedding,
        context_mask=text_mask,
        timesteps=mx.array([sigma]),
        positions=positions,
        enabled=True,
    )

    # Run through velocity model (not X0)
    print("\nRunning velocity model...")
    velocity_patchified = model(modality)
    mx.eval(velocity_patchified)

    print(f"Velocity patchified shape: {velocity_patchified.shape}")
    print(f"Velocity stats:")
    print(f"  Mean: {float(velocity_patchified.mean()):.6f}")
    print(f"  Std:  {float(velocity_patchified.std()):.6f}")
    print(f"  Min:  {float(velocity_patchified.min()):.6f}")
    print(f"  Max:  {float(velocity_patchified.max()):.6f}")

    # Check if velocity has structure (not just random)
    velocity_unpatchified = patchifier.unpatchify(velocity_patchified, output_shape=output_shape)
    print(f"\nUnpatchified velocity shape: {velocity_unpatchified.shape}")

    # Compute channel-wise variance
    velocity_np = np.array(velocity_unpatchified[0])  # [128, F, H, W]
    channel_vars = velocity_np.var(axis=(1, 2, 3))
    print(f"Channel variance range: [{channel_vars.min():.6f}, {channel_vars.max():.6f}]")

    # Now test X0 model
    print("\n" + "="*60)
    print("Testing X0 model...")
    print("="*60)

    denoised_patchified = x0_model(modality)
    mx.eval(denoised_patchified)

    print(f"Denoised patchified shape: {denoised_patchified.shape}")
    print(f"Denoised stats:")
    print(f"  Mean: {float(denoised_patchified.mean()):.6f}")
    print(f"  Std:  {float(denoised_patchified.std()):.6f}")
    print(f"  Min:  {float(denoised_patchified.min()):.6f}")
    print(f"  Max:  {float(denoised_patchified.max()):.6f}")

    # Compare with input
    print(f"\nInput latent patchified stats:")
    print(f"  Mean: {float(latent_patchified.mean()):.6f}")
    print(f"  Std:  {float(latent_patchified.std()):.6f}")

    # Correlation between input and denoised
    input_flat = np.array(latent_patchified).flatten()
    denoised_flat = np.array(denoised_patchified).flatten()
    corr = np.corrcoef(input_flat, denoised_flat)[0, 1]
    print(f"\nCorrelation(input, denoised): {corr:.4f}")

    # Test with different text embeddings
    print("\n" + "="*60)
    print("Testing text conditioning effect...")
    print("="*60)

    # Embedding A
    text_embedding_a = mx.random.normal(shape=(1, 64, 3840)) * 0.1
    modality_a = Modality(
        latent=latent_patchified,
        context=text_embedding_a,
        context_mask=text_mask,
        timesteps=mx.array([sigma]),
        positions=positions,
        enabled=True,
    )

    # Embedding B (completely different)
    text_embedding_b = mx.random.normal(shape=(1, 64, 3840)) * 0.1 + 1.0  # Shifted
    modality_b = Modality(
        latent=latent_patchified,
        context=text_embedding_b,
        context_mask=text_mask,
        timesteps=mx.array([sigma]),
        positions=positions,
        enabled=True,
    )

    denoised_a = x0_model(modality_a)
    denoised_b = x0_model(modality_b)
    mx.eval(denoised_a)
    mx.eval(denoised_b)

    denoised_a_flat = np.array(denoised_a).flatten()
    denoised_b_flat = np.array(denoised_b).flatten()
    corr_ab = np.corrcoef(denoised_a_flat, denoised_b_flat)[0, 1]

    diff = np.abs(denoised_a_flat - denoised_b_flat)
    print(f"Denoised A vs B:")
    print(f"  Correlation: {corr_ab:.4f}")
    print(f"  Mean abs diff: {diff.mean():.6f}")
    print(f"  Max abs diff: {diff.max():.6f}")

    if corr_ab > 0.99:
        print("\n  WARNING: Very high correlation - text conditioning may not be working!")
    else:
        print(f"\n  OK: Different embeddings produce different outputs")

    # Test CFG effect
    print("\n" + "="*60)
    print("Testing CFG effect...")
    print("="*60)

    # Null embedding (zeros)
    null_embedding = mx.zeros((1, 64, 3840))
    null_mask = mx.zeros((1, 64))
    modality_null = Modality(
        latent=latent_patchified,
        context=null_embedding,
        context_mask=null_mask,
        timesteps=mx.array([sigma]),
        positions=positions,
        enabled=True,
    )

    denoised_cond = x0_model(modality_a)
    denoised_uncond = x0_model(modality_null)
    mx.eval(denoised_cond)
    mx.eval(denoised_uncond)

    # CFG formula
    cfg_scale = 3.0
    denoised_cfg = denoised_uncond + cfg_scale * (denoised_cond - denoised_uncond)

    cond_flat = np.array(denoised_cond).flatten()
    uncond_flat = np.array(denoised_uncond).flatten()
    cfg_flat = np.array(denoised_cfg).flatten()

    corr_cond_uncond = np.corrcoef(cond_flat, uncond_flat)[0, 1]
    print(f"Conditional vs Unconditional:")
    print(f"  Correlation: {corr_cond_uncond:.4f}")
    print(f"  Cond mean: {cond_flat.mean():.6f}, std: {cond_flat.std():.6f}")
    print(f"  Uncond mean: {uncond_flat.mean():.6f}, std: {uncond_flat.std():.6f}")
    print(f"  CFG mean: {cfg_flat.mean():.6f}, std: {cfg_flat.std():.6f}")

    # Euler step test
    print("\n" + "="*60)
    print("Testing Euler step...")
    print("="*60)

    sigma_next = DISTILLED_SIGMA_VALUES[1]  # 0.99375
    dt = sigma_next - sigma
    print(f"sigma: {sigma} -> sigma_next: {sigma_next}, dt: {dt}")

    denoised_unpatch = patchifier.unpatchify(denoised_cfg, output_shape=output_shape)

    # Velocity from denoised
    velocity_from_x0 = to_velocity(latent, sigma, denoised_unpatch)

    # Euler step
    latent_next = latent.astype(mx.float32) + velocity_from_x0.astype(mx.float32) * dt
    mx.eval(latent_next)

    print(f"\nLatent after one step:")
    print(f"  Mean: {float(latent_next.mean()):.6f}")
    print(f"  Std:  {float(latent_next.std()):.6f}")

    latent_flat = np.array(latent).flatten()
    next_flat = np.array(latent_next).flatten()
    corr_step = np.corrcoef(latent_flat, next_flat)[0, 1]

    print(f"  Correlation(latent, latent_next): {corr_step:.4f}")
    print(f"  Change magnitude: {np.abs(next_flat - latent_flat).mean():.6f}")

    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    issues = []
    if corr_ab > 0.99:
        issues.append("Text conditioning may not be working (A vs B corr > 0.99)")
    if corr_cond_uncond > 0.99:
        issues.append("CFG has no effect (cond vs uncond corr > 0.99)")
    if corr_step > 0.999:
        issues.append("Euler step not changing latent enough")
    if np.abs(velocity_np).mean() < 0.01:
        issues.append("Velocity predictions too small")

    if issues:
        print("POTENTIAL ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No obvious issues detected in single step test.")
        print("Issue may be elsewhere (VAE, multi-step accumulation, etc.)")

if __name__ == "__main__":
    main()
