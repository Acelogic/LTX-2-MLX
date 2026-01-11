#!/usr/bin/env python3
"""
Analyze transformer output to understand if it's predicting meaningful x0.
"""

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
    from scripts.generate import encode_with_gemma

    # Test parameters
    height, width, frames = 256, 384, 17
    latent_h, latent_w, latent_f = height // 32, width // 32, (frames - 1) // 8 + 1

    print("Loading transformer...")
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

    # Load real text embedding
    print("\nEncoding text prompt...")
    prompt = "A blue ball bouncing on green grass"
    text_embedding, text_mask = encode_with_gemma(
        prompt,
        gemma_path="weights/gemma-3-12b",
        ltx_weights_path="weights/ltx-2/ltx-2-19b-distilled.safetensors",
    )
    mx.eval(text_embedding, text_mask)

    print(f"Text embedding shape: {text_embedding.shape}")
    print(f"Text mask shape: {text_mask.shape}")
    print(f"Text embedding mean: {float(mx.mean(text_embedding)):.6f}")
    print(f"Text embedding std: {float(mx.std(text_embedding)):.6f}")

    # Create noise latent
    mx.random.seed(42)
    noise = mx.random.normal((1, 128, latent_f, latent_h, latent_w))

    # Setup patchifier and positions
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

    # Test at different sigma levels
    sigmas_to_test = [1.0, 0.725, 0.421875, 0.1]

    print(f"\n{'='*70}")
    print("Testing transformer predictions at different noise levels")
    print(f"{'='*70}")

    for sigma in sigmas_to_test:
        if sigma not in DISTILLED_SIGMA_VALUES:
            continue

        print(f"\n--- Sigma = {sigma} ---")

        # Patchify latent
        latent_patchified = patchifier.patchify(noise)

        # Create modality
        modality = Modality(
            latent=latent_patchified,
            context=text_embedding,
            context_mask=text_mask,
            timesteps=mx.array([sigma]),
            positions=positions,
            enabled=True,
        )

        # Get velocity prediction
        velocity = model(modality)
        mx.eval(velocity)

        print(f"Velocity prediction:")
        print(f"  Mean: {float(mx.mean(velocity)):.6f}")
        print(f"  Std:  {float(mx.std(velocity)):.6f}")
        print(f"  Range: [{float(mx.min(velocity)):.6f}, {float(mx.max(velocity)):.6f}]")

        # Get x0 prediction
        x0 = x0_model(modality)
        mx.eval(x0)

        # Unpatchify x0
        x0_unpatch = patchifier.unpatchify(x0, output_shape=output_shape)
        x0_np = np.array(x0_unpatch[0])  # [128, F, H, W]

        print(f"\nX0 prediction (denoised):")
        print(f"  Mean: {x0_np.mean():.6f}")
        print(f"  Std:  {x0_np.std():.6f}")
        print(f"  Range: [{x0_np.min():.6f}, {x0_np.max():.6f}]")

        # Per-channel statistics
        channel_means = x0_np.mean(axis=(1, 2, 3))
        channel_stds = x0_np.std(axis=(1, 2, 3))
        print(f"  Channel means: range=[{channel_means.min():.4f}, {channel_means.max():.4f}]")
        print(f"  Channel stds:  range=[{channel_stds.min():.4f}, {channel_stds.max():.4f}]")

        # Check correlation with input noise
        noise_np = np.array(noise[0])
        corr = np.corrcoef(x0_np.flatten(), noise_np.flatten())[0, 1]
        print(f"  Correlation with input noise: {corr:.4f}")

        # Check spatial structure
        frame_0 = x0_np[:, 0, :, :]  # [128, H, W]
        channel_spatial_vars = frame_0.var(axis=(1, 2))
        print(f"  Spatial variance per channel: mean={channel_spatial_vars.mean():.6f}")

    # Compare conditional vs unconditional
    print(f"\n{'='*70}")
    print("Comparing conditional vs unconditional predictions at sigma=1.0")
    print(f"{'='*70}")

    sigma = 1.0
    latent_patchified = patchifier.patchify(noise)

    # Conditional
    modality_cond = Modality(
        latent=latent_patchified,
        context=text_embedding,
        context_mask=text_mask,
        timesteps=mx.array([sigma]),
        positions=positions,
        enabled=True,
    )
    x0_cond = x0_model(modality_cond)
    mx.eval(x0_cond)

    # Unconditional (null embedding)
    null_embedding = mx.zeros_like(text_embedding)
    null_mask = mx.zeros_like(text_mask)
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

    x0_cond_np = np.array(x0_cond)
    x0_uncond_np = np.array(x0_uncond)

    # Difference
    diff = x0_cond_np - x0_uncond_np

    print(f"\nConditional X0:")
    print(f"  Mean: {x0_cond_np.mean():.6f}, Std: {x0_cond_np.std():.6f}")

    print(f"\nUnconditional X0:")
    print(f"  Mean: {x0_uncond_np.mean():.6f}, Std: {x0_uncond_np.std():.6f}")

    print(f"\nDifference (cond - uncond):")
    print(f"  Mean: {diff.mean():.6f}")
    print(f"  Std:  {diff.std():.6f}")
    print(f"  Range: [{diff.min():.6f}, {diff.max():.6f}]")

    corr = np.corrcoef(x0_cond_np.flatten(), x0_uncond_np.flatten())[0, 1]
    print(f"\nCorrelation(cond, uncond): {corr:.4f}")

    # CFG effect
    cfg_scale = 3.0
    x0_cfg = x0_uncond_np + cfg_scale * diff
    print(f"\nCFG (scale={cfg_scale}) result:")
    print(f"  Mean: {x0_cfg.mean():.6f}, Std: {x0_cfg.std():.6f}")
    print(f"  Range: [{x0_cfg.min():.6f}, {x0_cfg.max():.6f}]")

    # Key diagnostic
    print(f"\n{'='*70}")
    print("KEY DIAGNOSTICS")
    print(f"{'='*70}")

    if np.abs(diff).mean() < 0.01:
        print("WARNING: Conditional and unconditional outputs are nearly identical!")
        print("  -> Text conditioning is not influencing the transformer output.")
        print("  -> Check: text embedding values, cross-attention mask, cross-attention weights")
    elif corr > 0.99:
        print("WARNING: High correlation between cond and uncond.")
        print("  -> Text is only slightly influencing output.")
        print("  -> CFG will have limited effect.")
    else:
        print(f"OK: Text conditioning has measurable effect (diff std = {diff.std():.4f})")

    # Check if x0 looks like data or noise
    x0_cond_unpatch = patchifier.unpatchify(mx.array(x0_cond_np), output_shape=output_shape)
    x0_np = np.array(x0_cond_unpatch[0])
    channel_means = x0_np.mean(axis=(1, 2, 3))
    channel_stds = x0_np.std(axis=(1, 2, 3))

    print(f"\nX0 statistics analysis:")
    print(f"  Expected for encoder output: mean ~0, std ~1")
    print(f"  Actual X0 channel means: mean={channel_means.mean():.4f}, range=[{channel_means.min():.4f}, {channel_means.max():.4f}]")
    print(f"  Actual X0 channel stds:  mean={channel_stds.mean():.4f}, range=[{channel_stds.min():.4f}, {channel_stds.max():.4f}]")

    if channel_stds.mean() < 0.5:
        print("WARNING: X0 has very low per-channel variance.")
        print("  -> Model may be predicting something too uniform.")
    elif channel_stds.mean() > 2.0:
        print("WARNING: X0 has very high per-channel variance.")
        print("  -> Model output may have scale issues.")

if __name__ == "__main__":
    main()
