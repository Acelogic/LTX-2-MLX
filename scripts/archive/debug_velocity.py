"""Debug raw velocity output from the transformer."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np


def analyze_array(name: str, x):
    """Analyze array statistics."""
    print(f"\n{name}:")
    print(f"  Shape: {x.shape}")
    print(f"  Range: [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]")
    print(f"  Mean: {float(mx.mean(x)):.4f}, Std: {float(mx.std(x)):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/ltx-2/ltx-2-19b-distilled.safetensors",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Raw Velocity Analysis")
    print("=" * 60)

    from LTX_2_MLX.model.transformer import LTXModel, LTXModelType, Modality
    from LTX_2_MLX.components import VideoLatentPatchifier
    from LTX_2_MLX.components.patchifiers import get_pixel_coords
    from LTX_2_MLX.types import VideoLatentShape, SpatioTemporalScaleFactors
    from LTX_2_MLX.loader import load_transformer_weights

    compute_dtype = mx.float16
    mx.random.seed(42)

    # Small test
    latent_frames, latent_height, latent_width = 3, 8, 12

    # Load model (without X0 wrapper)
    print("\nLoading velocity model (no X0 wrapper)...")
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

    # Create synthetic test case
    print("\nCreating synthetic test case...")
    clean_data = mx.random.normal((1, 128, latent_frames, latent_height, latent_width)) * 0.3
    noise = mx.random.normal((1, 128, latent_frames, latent_height, latent_width))

    # True velocity should be: v = noise - x_0
    true_velocity = noise - clean_data

    analyze_array("Clean data", clean_data)
    analyze_array("Noise", noise)
    analyze_array("True velocity (noise - clean)", true_velocity)

    # At sigma=1.0, noisy sample = (1-1)*clean + 1*noise = noise
    sigma = 1.0
    noisy = (1 - sigma) * clean_data + sigma * noise

    analyze_array(f"Noisy sample at sigma={sigma}", noisy)

    # Patchify
    noisy_patchified = patchifier.patchify(noisy)
    true_vel_patchified = patchifier.patchify(true_velocity)

    # Zero text context
    context = mx.zeros((1, 64, 3840), dtype=compute_dtype)
    context_mask = mx.ones((1, 64))

    # Run model
    print("\n" + "=" * 60)
    print(f"Running model at sigma={sigma}...")
    print("=" * 60)

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

    # Unpatchify velocity
    velocity_spatial = patchifier.unpatchify(velocity_pred, output_shape=output_shape)

    analyze_array("Predicted velocity (raw model output)", velocity_spatial)
    analyze_array("True velocity", true_velocity)

    # Compare predicted vs true velocity
    print("\n--- Velocity Comparison ---")
    mse_to_true = float(mx.mean((velocity_spatial - true_velocity) ** 2))
    mse_to_noise = float(mx.mean((velocity_spatial - noise) ** 2))
    mse_to_clean = float(mx.mean((velocity_spatial - clean_data) ** 2))
    mse_to_noisy = float(mx.mean((velocity_spatial - noisy) ** 2))

    print(f"  MSE(predicted, true_velocity): {mse_to_true:.4f}")
    print(f"  MSE(predicted, noise): {mse_to_noise:.4f}")
    print(f"  MSE(predicted, clean): {mse_to_clean:.4f}")
    print(f"  MSE(predicted, noisy_input): {mse_to_noisy:.4f}")

    # Correlation with true velocity
    pred_flat = velocity_spatial.flatten().astype(mx.float32)
    true_flat = true_velocity.flatten().astype(mx.float32)
    corr = float(mx.sum(pred_flat * true_flat)) / (
        float(mx.sqrt(mx.sum(pred_flat**2))) *
        float(mx.sqrt(mx.sum(true_flat**2))) + 1e-8
    )
    print(f"  Correlation(predicted, true_velocity): {corr:.4f}")

    # What X0 would be computed as
    print("\n--- X0 Reconstruction ---")
    x0_pred = noisy - sigma * velocity_spatial

    analyze_array("X0 from predicted velocity", x0_pred)
    analyze_array("True X0 (clean data)", clean_data)

    mse_x0_to_clean = float(mx.mean((x0_pred - clean_data) ** 2))
    print(f"\n  MSE(reconstructed_x0, clean): {mse_x0_to_clean:.4f}")

    # Check if model is predicting something close to input
    print("\n--- Is model predicting close to input? ---")
    input_pred_corr = float(mx.sum(noisy.flatten() * velocity_spatial.flatten())) / (
        float(mx.sqrt(mx.sum(noisy**2))) * float(mx.sqrt(mx.sum(velocity_spatial**2))) + 1e-8
    )
    print(f"  Correlation(noisy_input, velocity_pred): {input_pred_corr:.4f}")

    # Test at different sigma values
    print("\n" + "=" * 60)
    print("Testing at multiple sigma values...")
    print("=" * 60)

    for sigma in [1.0, 0.5, 0.1, 0.01]:
        noisy = (1 - sigma) * clean_data + sigma * noise
        noisy_patchified = patchifier.patchify(noisy)

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

        # Compute X0
        x0_pred = noisy - sigma * velocity_spatial

        # Compare to true values
        mse_x0 = float(mx.mean((x0_pred - clean_data) ** 2))
        mse_v = float(mx.mean((velocity_spatial - true_velocity) ** 2))

        print(f"\n  sigma={sigma}:")
        print(f"    Velocity pred std: {float(mx.std(velocity_spatial)):.4f}")
        print(f"    X0 pred std: {float(mx.std(x0_pred)):.4f}")
        print(f"    MSE(X0_pred, clean): {mse_x0:.4f}")
        print(f"    MSE(v_pred, true_v): {mse_v:.4f}")


if __name__ == "__main__":
    main()
