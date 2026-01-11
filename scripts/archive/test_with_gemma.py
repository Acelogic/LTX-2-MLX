"""Test denoising with real Gemma embeddings at high sigma."""

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np


def analyze_array(name: str, x):
    """Analyze array statistics."""
    if x.ndim == 5:
        # Video latent
        spatial = x[0, 0, 0]
        h_diff = float(mx.mean(mx.abs(spatial[1:, :] - spatial[:-1, :])))
        w_diff = float(mx.mean(mx.abs(spatial[:, 1:] - spatial[:, :-1])))
        std = float(mx.std(spatial))
        coherence = f", coherence_ratio={w_diff/std:.2f}"
    else:
        coherence = ""

    print(f"{name}:")
    print(f"  Range: [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]")
    print(f"  Mean: {float(mx.mean(x)):.4f}, Std: {float(mx.std(x)):.4f}{coherence}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/ltx-2/ltx-2-19b-distilled.safetensors",
    )
    parser.add_argument(
        "--gemma-path",
        type=str,
        default="weights/gemma-3-12b",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cat walking on grass",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Testing denoising with real Gemma embeddings")
    print("=" * 60)

    import os
    from LTX_2_MLX.model.transformer import LTXModel, LTXModelType, Modality
    from LTX_2_MLX.components import VideoLatentPatchifier
    from LTX_2_MLX.components.patchifiers import get_pixel_coords
    from LTX_2_MLX.types import VideoLatentShape, SpatioTemporalScaleFactors
    from LTX_2_MLX.loader import load_transformer_weights

    compute_dtype = mx.float16
    mx.random.seed(42)

    # Small test
    latent_frames, latent_height, latent_width = 3, 8, 12

    # Get real Gemma embeddings
    print("\n[1] Getting real Gemma embeddings...")
    if not os.path.exists(args.gemma_path):
        print(f"ERROR: Gemma weights not found at {args.gemma_path}")
        print("Using zeros instead for testing...")
        context = mx.zeros((1, 64, 3840), dtype=compute_dtype)
        context_mask = mx.ones((1, 64))
    else:
        from scripts.generate import encode_with_gemma
        context, context_mask = encode_with_gemma(
            prompt=args.prompt,
            gemma_path=args.gemma_path,
            ltx_weights_path=args.weights,
            max_length=256,  # Must be divisible by 128
        )
        if context is None:
            print("Failed to encode, using zeros...")
            context = mx.zeros((1, 64, 3840), dtype=compute_dtype)
            context_mask = mx.ones((1, 64))
        else:
            context = context.astype(compute_dtype)

    print(f"\nContext shape: {context.shape}")
    print(f"Context mask sum: {float(mx.sum(context_mask))} (active tokens)")
    analyze_array("Context embedding", context)

    # Load model
    print("\n[2] Loading velocity model...")
    model = LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=32,
        attention_head_dim=128,
        in_channels=128,
        out_channels=128,
        num_layers=48,
        cross_attention_dim=4096,
        caption_channels=3840,
        compute_dtype=compute_dtype,
    )
    load_transformer_weights(model, args.weights, use_fp8=True, target_dtype="float16")

    # Setup patchifier
    patchifier = VideoLatentPatchifier(patch_size=1)
    output_shape = VideoLatentShape(
        batch=1, channels=128, frames=latent_frames,
        height=latent_height, width=latent_width
    )
    latent_coords = patchifier.get_patch_grid_bounds(output_shape=output_shape)
    scale_factors = SpatioTemporalScaleFactors.default()
    positions = get_pixel_coords(latent_coords, scale_factors, causal_fix=True).astype(mx.float32)
    positions = mx.concatenate([positions[:, 0:1, ...] / 24.0, positions[:, 1:, ...]], axis=1)

    # Create pure noise input
    print("\n[3] Creating test input...")
    noise = mx.random.normal((1, 128, latent_frames, latent_height, latent_width))
    analyze_array("Input noise", noise)

    # Test at different sigma values with real text
    print("\n[4] Testing denoising at different sigma values...")

    for sigma in [1.0, 0.7, 0.4, 0.1]:
        noisy_patchified = patchifier.patchify(noise)

        modality = Modality(
            latent=noisy_patchified.astype(compute_dtype),
            context=context,
            context_mask=context_mask,
            timesteps=mx.array([sigma]),
            positions=positions,
            enabled=True,
        )

        velocity_pred = model(modality)
        mx.eval(velocity_pred)
        velocity_spatial = patchifier.unpatchify(velocity_pred, output_shape=output_shape)

        # Compute X0 prediction
        x0_pred = noise - sigma * velocity_spatial

        print(f"\n  sigma={sigma}:")
        analyze_array("    Velocity prediction", velocity_spatial)
        analyze_array("    X0 prediction", x0_pred)

    # Test FULL denoising loop
    print("\n" + "=" * 60)
    print("[5] Running full denoising loop...")
    print("=" * 60)

    from LTX_2_MLX.components import DISTILLED_SIGMA_VALUES

    latent = mx.array(noise)
    sigmas = DISTILLED_SIGMA_VALUES

    print(f"\nSigma schedule: {sigmas}")

    for step, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        latent_patchified = patchifier.patchify(latent)

        modality = Modality(
            latent=latent_patchified.astype(compute_dtype),
            context=context,
            context_mask=context_mask,
            timesteps=mx.array([sigma]),
            positions=positions,
            enabled=True,
        )

        velocity_pred = model(modality)
        mx.eval(velocity_pred)
        velocity_spatial = patchifier.unpatchify(velocity_pred, output_shape=output_shape)

        # X0 prediction
        x0_pred = latent - sigma * velocity_spatial

        # Euler step using X0
        velocity_for_step = (latent - x0_pred) / sigma
        dt = sigma_next - sigma
        new_latent = latent.astype(mx.float32) + velocity_for_step.astype(mx.float32) * dt
        mx.eval(new_latent)

        print(f"\n  Step {step}: sigma {sigma:.4f} -> {sigma_next:.4f}")
        analyze_array("    X0 prediction", x0_pred)
        analyze_array("    New latent", new_latent)

        latent = new_latent

    # Final analysis
    print("\n" + "=" * 60)
    print("[6] Final latent analysis")
    print("=" * 60)
    analyze_array("\nFinal denoised latent", latent)

    # Save for inspection
    print("\nSaving latent to /tmp/gemma_test_latent.npy...")
    np.save("/tmp/gemma_test_latent.npy", np.array(latent))


if __name__ == "__main__":
    main()
