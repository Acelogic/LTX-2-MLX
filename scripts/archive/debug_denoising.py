"""Debug denoising loop to verify model predictions are correct."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np


def analyze_latent(name: str, x):
    """Analyze latent statistics."""
    print(f"\n{name}:")
    print(f"  Shape: {x.shape}")
    print(f"  Range: [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]")
    print(f"  Mean: {float(mx.mean(x)):.4f}, Std: {float(mx.std(x)):.4f}")

    # Spatial coherence
    if x.ndim == 5:
        spatial = x[0, 0, 0]  # First frame, first channel
        h_diff = float(mx.mean(mx.abs(spatial[1:, :] - spatial[:-1, :])))
        w_diff = float(mx.mean(mx.abs(spatial[:, 1:] - spatial[:, :-1])))
        std = float(mx.std(spatial))
        print(f"  Spatial coherence: h_diff={h_diff:.4f}, w_diff={w_diff:.4f}, ratio={w_diff/std:.2f}")


def run_denoising_debug(
    weights_path: str,
    gemma_path: str,
    prompt: str = "A cat walking on grass",
    use_fp16: bool = True,
):
    """Debug denoising with detailed analysis."""

    print("=" * 60)
    print("Denoising Loop Debug")
    print("=" * 60)

    from LTX_2_MLX.model.transformer import LTXModel, LTXModelType, Modality, X0Model
    from LTX_2_MLX.components import DISTILLED_SIGMA_VALUES, VideoLatentPatchifier
    from LTX_2_MLX.components.patchifiers import get_pixel_coords
    from LTX_2_MLX.types import VideoLatentShape, SpatioTemporalScaleFactors
    from LTX_2_MLX.loader import load_transformer_weights

    # Setup
    compute_dtype = mx.float16 if use_fp16 else mx.float32
    mx.random.seed(42)

    # Small test resolution
    height, width, num_frames = 256, 384, 17
    latent_height = height // 32
    latent_width = width // 32
    latent_frames = (num_frames - 1) // 8 + 1

    print(f"\nResolution: {width}x{height}, {num_frames} frames")
    print(f"Latent: {latent_frames}x{latent_height}x{latent_width}")

    # Load model
    print("\nLoading transformer...")
    velocity_model = LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=32,
        attention_head_dim=128,
        in_channels=128,
        out_channels=128,
        num_layers=48,
        cross_attention_dim=4096,
        caption_channels=3840,
        positional_embedding_theta=10000.0,
        compute_dtype=compute_dtype,
    )
    target_dtype = "float16" if use_fp16 else "float32"
    load_transformer_weights(velocity_model, weights_path, use_fp8=True, target_dtype=target_dtype)
    model = X0Model(velocity_model)

    # Create text encoding (zeros for unconditioned test)
    print("\nUsing zero text encoding (unconditioned)...")
    context_len = 64
    text_encoding = mx.zeros((1, context_len, 3840), dtype=compute_dtype)
    text_mask = mx.ones((1, context_len))

    # Initial noise
    print("\nInitializing noise...")
    latent = mx.random.normal((1, 128, latent_frames, latent_height, latent_width))
    analyze_latent("Initial noise", latent)

    # Setup patchifier and positions
    patchifier = VideoLatentPatchifier(patch_size=1)
    output_shape = VideoLatentShape(
        batch=1, channels=128, frames=latent_frames,
        height=latent_height, width=latent_width
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

    # Sigma schedule
    sigmas = DISTILLED_SIGMA_VALUES
    print(f"\nSigmas: {sigmas}")

    # Run denoising loop
    print("\n" + "=" * 60)
    print("Running denoising steps...")
    print("=" * 60)

    for step in range(len(sigmas) - 1):
        sigma = sigmas[step]
        sigma_next = sigmas[step + 1]

        print(f"\n--- Step {step}: sigma {sigma:.4f} -> {sigma_next:.4f} ---")

        # Patchify
        latent_patchified = patchifier.patchify(latent)

        # Create modality
        modality = Modality(
            latent=latent_patchified.astype(compute_dtype),
            context=text_encoding,
            context_mask=text_mask,
            timesteps=mx.array([sigma]),
            positions=positions,
            enabled=True,
        )

        # Get prediction
        x0_patchified = model(modality)
        mx.eval(x0_patchified)

        # Unpatchify
        denoised = patchifier.unpatchify(x0_patchified, output_shape=output_shape)
        mx.eval(denoised)

        # Analyze prediction
        analyze_latent("Denoised (X0) prediction", denoised)

        # Compute velocity
        velocity = (latent - denoised) / sigma
        analyze_latent("Implied velocity", velocity)

        # Euler step
        dt = sigma_next - sigma
        new_latent = latent.astype(mx.float32) + velocity.astype(mx.float32) * dt
        mx.eval(new_latent)

        analyze_latent("New latent", new_latent)

        # Correlation between input and output
        lat_flat = latent.flatten().astype(mx.float32)
        den_flat = denoised.flatten().astype(mx.float32)
        corr = float(mx.sum(lat_flat * den_flat)) / (
            float(mx.sqrt(mx.sum(lat_flat**2))) *
            float(mx.sqrt(mx.sum(den_flat**2))) + 1e-8
        )
        print(f"  Correlation(input, denoised): {corr:.4f}")

        latent = new_latent

    # Final analysis
    print("\n" + "=" * 60)
    print("Final latent analysis")
    print("=" * 60)
    analyze_latent("Final denoised latent", latent)


def run_single_step_analysis(weights_path: str, use_fp16: bool = True):
    """Analyze a single denoising step in detail."""

    print("\n" + "=" * 60)
    print("Single Step Analysis at sigma=1.0")
    print("=" * 60)

    from LTX_2_MLX.model.transformer import LTXModel, LTXModelType, Modality, X0Model
    from LTX_2_MLX.components import VideoLatentPatchifier
    from LTX_2_MLX.components.patchifiers import get_pixel_coords
    from LTX_2_MLX.types import VideoLatentShape, SpatioTemporalScaleFactors
    from LTX_2_MLX.loader import load_transformer_weights

    compute_dtype = mx.float16 if use_fp16 else mx.float32
    mx.random.seed(42)

    # Small resolution
    latent_frames, latent_height, latent_width = 3, 8, 12

    # Load model
    print("\nLoading transformer...")
    velocity_model = LTXModel(
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
    target_dtype = "float16" if use_fp16 else "float32"
    load_transformer_weights(velocity_model, weights_path, use_fp8=True, target_dtype=target_dtype)
    model = X0Model(velocity_model)

    # Setup
    patchifier = VideoLatentPatchifier(patch_size=1)
    output_shape = VideoLatentShape(
        batch=1, channels=128, frames=latent_frames,
        height=latent_height, width=latent_width
    )
    latent_coords = patchifier.get_patch_grid_bounds(output_shape=output_shape)
    scale_factors = SpatioTemporalScaleFactors.default()
    positions = get_pixel_coords(latent_coords, scale_factors, causal_fix=True).astype(mx.float32)
    positions = mx.concatenate([positions[:, 0:1, ...] / 24.0, positions[:, 1:, ...]], axis=1)

    # Create synthetic "clean" data and noise
    print("\nCreating synthetic test case...")
    clean_data = mx.random.normal((1, 128, latent_frames, latent_height, latent_width)) * 0.3
    noise = mx.random.normal((1, 128, latent_frames, latent_height, latent_width))

    # Check if clean data has spatial structure
    analyze_latent("Clean data (synthetic)", clean_data)

    # At sigma=1.0, noisy sample is: x_t = (1-t)*clean + t*noise
    sigma = 1.0
    noisy = (1 - sigma) * clean_data + sigma * noise
    analyze_latent("Noisy sample at sigma=1.0", noisy)

    # Use zeros for text (unconditioned)
    context = mx.zeros((1, 64, 3840), dtype=compute_dtype)
    context_mask = mx.ones((1, 64))

    # Run model
    print("\nRunning model...")
    latent_patchified = patchifier.patchify(noisy)
    modality = Modality(
        latent=latent_patchified.astype(compute_dtype),
        context=context,
        context_mask=context_mask,
        timesteps=mx.array([sigma]),
        positions=positions,
        enabled=True,
    )

    x0_pred = model(modality)
    mx.eval(x0_pred)
    denoised = patchifier.unpatchify(x0_pred, output_shape=output_shape)
    mx.eval(denoised)

    analyze_latent("Model's X0 prediction", denoised)

    # Compare distances
    print("\n--- Distance Analysis ---")
    d_to_clean = float(mx.mean((denoised - clean_data) ** 2))
    d_to_noise = float(mx.mean((denoised - noise) ** 2))
    d_noisy_to_clean = float(mx.mean((noisy - clean_data) ** 2))

    print(f"  MSE(denoised, clean): {d_to_clean:.4f}")
    print(f"  MSE(noisy, clean):    {d_noisy_to_clean:.4f}")
    print(f"  MSE(denoised, noise): {d_to_noise:.4f}")

    if d_to_clean < d_noisy_to_clean:
        improvement = 100 * (1 - d_to_clean / d_noisy_to_clean)
        print(f"  Model moved {improvement:.1f}% closer to clean")
    else:
        degradation = 100 * (d_to_clean / d_noisy_to_clean - 1)
        print(f"  WARNING: Model moved {degradation:.1f}% AWAY from clean!")

    # Check what the model predicts for PURE NOISE input
    print("\n--- Test with pure noise input ---")
    pure_noise = mx.random.normal((1, 128, latent_frames, latent_height, latent_width))
    noise_patchified = patchifier.patchify(pure_noise)
    noise_modality = Modality(
        latent=noise_patchified.astype(compute_dtype),
        context=context,
        context_mask=context_mask,
        timesteps=mx.array([1.0]),
        positions=positions,
        enabled=True,
    )

    noise_x0 = model(noise_modality)
    mx.eval(noise_x0)
    noise_denoised = patchifier.unpatchify(noise_x0, output_shape=output_shape)
    mx.eval(noise_denoised)

    analyze_latent("X0 prediction for pure noise", noise_denoised)

    # Check if model output has any spatial structure
    print("\n--- Checking model output spatial structure ---")
    out_np = np.array(noise_denoised[0, 0, 0])  # First frame, first channel
    h_smooth = np.abs(out_np[1:, :] - out_np[:-1, :]).mean()
    w_smooth = np.abs(out_np[:, 1:] - out_np[:, :-1]).mean()
    out_std = out_np.std()

    print(f"  Output spatial smoothness: h={h_smooth:.4f}, w={w_smooth:.4f}")
    print(f"  Ratio to std: {w_smooth/out_std:.2f}")

    if w_smooth / out_std < 0.5:
        print("  Model output has spatial structure (good)")
    else:
        print("  WARNING: Model output has no spatial structure!")


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
        "--fp32",
        action="store_true",
        help="Use FP32 instead of FP16"
    )
    parser.add_argument(
        "--single-step",
        action="store_true",
        help="Run single step analysis only"
    )
    args = parser.parse_args()

    if args.single_step:
        run_single_step_analysis(args.weights, use_fp16=not args.fp32)
    else:
        run_denoising_debug(args.weights, args.gemma_path, use_fp16=not args.fp32)


if __name__ == "__main__":
    main()
