#!/usr/bin/env python3
"""Compare full denoising loop MLX vs PyTorch.

Tests that multiple denoising steps produce identical results.
"""

import sys
from pathlib import Path

# Mock triton before any imports
import types
mock_triton = types.ModuleType('triton')
mock_triton.cdiv = lambda a, b: (a + b - 1) // b
mock_triton.jit = lambda fn: fn

mock_triton_language = types.ModuleType('triton.language')
mock_triton_language.constexpr = int

class MockDtype:
    pass
mock_triton_language.dtype = MockDtype

mock_triton.language = mock_triton_language

sys.modules['triton'] = mock_triton
sys.modules['triton.language'] = mock_triton_language

sys.path.insert(0, str(Path(__file__).parent.parent))

# Add PyTorch LTX-2 to path
pytorch_ltx_path = Path(__file__).parent.parent.parent / "LTX-2-PyTorch"
if pytorch_ltx_path.exists():
    sys.path.insert(0, str(pytorch_ltx_path / "packages" / "ltx-core" / "src"))
    sys.path.insert(0, str(pytorch_ltx_path / "packages" / "ltx-pipelines" / "src"))

import numpy as np
import torch
import mlx.core as mx


def compare_arrays(name, a, b, rtol=0.01, atol=0.01, verbose=True):
    """Compare two arrays and print results."""
    if a.shape != b.shape:
        print(f"  {name}: SHAPE MISMATCH - {a.shape} vs {b.shape}")
        return False, 0

    abs_diff = np.abs(a - b)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()

    try:
        corr = np.corrcoef(a.flatten(), b.flatten())[0, 1]
    except:
        corr = float('nan')

    close = np.allclose(a, b, rtol=rtol, atol=atol)
    status = "OK" if close else "DIFF"

    if verbose:
        print(f"  {name}: {status} max={max_diff:.6f}, mean={mean_diff:.6f}, corr={corr:.4f}")
        if not close:
            print(f"    MLX: mean={a.mean():.6f}, std={a.std():.6f}")
            print(f"    PT:  mean={b.mean():.6f}, std={b.std():.6f}")

    return close, corr


def create_pixel_space_positions(frames, height, width, fps=24.0, time_scale=8, spatial_scale=32):
    """Create pixel-space positions matching PyTorch production pipeline."""
    batch_size = 1

    frame_coords = np.arange(0, frames)
    height_coords = np.arange(0, height)
    width_coords = np.arange(0, width)

    grid_f, grid_h, grid_w = np.meshgrid(frame_coords, height_coords, width_coords, indexing="ij")

    patch_starts = np.stack([grid_f, grid_h, grid_w], axis=0)
    patch_ends = patch_starts + 1

    latent_coords = np.stack([patch_starts, patch_ends], axis=-1)
    num_tokens = frames * height * width
    latent_coords = latent_coords.reshape(3, num_tokens, 2)
    latent_coords = latent_coords[None, ...]

    scale_factors = np.array([time_scale, spatial_scale, spatial_scale]).reshape(1, 3, 1, 1)
    pixel_coords = latent_coords * scale_factors

    pixel_coords[:, 0, ...] = np.maximum(pixel_coords[:, 0, ...] + 1 - time_scale, 0)
    pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / fps

    return pixel_coords.astype(np.float32)


def main():
    weights_path = "weights/ltx-2/ltx-2-19b-distilled.safetensors"

    print("=" * 70)
    print("Full Denoising Loop Comparison")
    print("=" * 70)

    if not Path(weights_path).exists():
        print(f"Weights not found: {weights_path}")
        return

    # Test parameters
    np.random.seed(42)
    batch_size = 1
    latent_channels = 128
    frames, height, width = 3, 8, 12
    context_dim = 3840
    context_len = 256

    # Distilled sigma schedule
    sigmas = np.array([1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0])

    # Create inputs - start with pure noise
    initial_noise = np.random.randn(batch_size, latent_channels, frames, height, width).astype(np.float32)
    context = np.random.randn(batch_size, context_len, context_dim).astype(np.float32) * 0.1

    def patchify(x):
        B, C, F, H, W = x.shape
        return x.transpose(0, 2, 3, 4, 1).reshape(B, F * H * W, C)

    def unpatchify(x, F, H, W):
        B, T, C = x.shape
        return x.reshape(B, F, H, W, C).transpose(0, 4, 1, 2, 3)

    initial_latent_patchified = patchify(initial_noise)
    positions = create_pixel_space_positions(frames, height, width)

    print(f"\nInput shapes:")
    print(f"  Initial noise (patchified): {initial_latent_patchified.shape}")
    print(f"  Context: {context.shape}")
    print(f"  Positions: {positions.shape}")
    print(f"  Sigmas: {sigmas}")

    # Load PyTorch model
    print("\n" + "=" * 70)
    print("Loading models")
    print("=" * 70)

    from ltx_core.model.transformer.model_configurator import (
        LTXV_MODEL_COMFY_RENAMING_MAP,
        LTXVideoOnlyModelConfigurator,
    )
    from ltx_core.loader import SingleGPUModelBuilder
    from ltx_core.model.transformer.model import X0Model as PTX0Model
    from ltx_core.model.transformer.modality import Modality as PTModality

    builder = SingleGPUModelBuilder(
        model_class_configurator=LTXVideoOnlyModelConfigurator,
        model_path=weights_path,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    )
    pt_velocity_model = builder.build(device=torch.device("cpu"), dtype=torch.float32)
    pt_velocity_model.eval()
    pt_x0_model = PTX0Model(pt_velocity_model)

    # Load MLX model
    from LTX_2_MLX.model.transformer import LTXModel, LTXModelType, X0Model as MLXX0Model
    from LTX_2_MLX.model.transformer.model import Modality as MLXModality
    from LTX_2_MLX.loader import load_transformer_weights

    mlx_velocity_model = LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=32,
        attention_head_dim=128,
        in_channels=128,
        out_channels=128,
        num_layers=48,
        cross_attention_dim=4096,
        caption_channels=3840,
        compute_dtype=mx.float32,
    )
    load_transformer_weights(mlx_velocity_model, weights_path)
    mlx_x0_model = MLXX0Model(mlx_velocity_model)

    # Initialize latents
    mlx_latent = mx.array(initial_latent_patchified)
    pt_latent = torch.from_numpy(initial_latent_patchified)

    # Run denoising loop
    print("\n" + "=" * 70)
    print("Running Denoising Loop")
    print("=" * 70)

    num_tokens = initial_latent_patchified.shape[1]

    for step_idx in range(len(sigmas) - 1):
        sigma = sigmas[step_idx]
        sigma_next = sigmas[step_idx + 1]
        dt = sigma_next - sigma

        print(f"\n--- Step {step_idx}: sigma={sigma:.4f} -> {sigma_next:.4f}, dt={dt:.4f} ---")

        # MLX forward pass
        mlx_modality = MLXModality(
            latent=mlx_latent,
            context=mx.array(context),
            context_mask=None,
            timesteps=mx.array([sigma]),
            positions=mx.array(positions),
            enabled=True,
        )
        mlx_denoised = mlx_x0_model(mlx_modality)
        mx.eval(mlx_denoised)

        # PyTorch forward pass
        pt_timesteps = np.ones((batch_size, num_tokens, 1), dtype=np.float32) * sigma
        pt_modality = PTModality(
            enabled=True,
            latent=pt_latent,
            timesteps=torch.from_numpy(pt_timesteps),
            positions=torch.from_numpy(positions),
            context=torch.from_numpy(context),
            context_mask=None,
        )
        with torch.no_grad():
            pt_denoised, _ = pt_x0_model(video=pt_modality, audio=None, perturbations=None)

        # Compare denoised predictions
        match, corr = compare_arrays(f"X0 prediction", np.array(mlx_denoised), pt_denoised.numpy())

        # Euler step: x_next = x + (x - denoised) / sigma * dt
        # Equivalently: velocity = (x - denoised) / sigma, x_next = x + velocity * dt
        mlx_velocity = (mlx_latent - mlx_denoised) / sigma
        mlx_latent = mlx_latent + mlx_velocity * dt
        mx.eval(mlx_latent)

        pt_velocity = (pt_latent - pt_denoised) / sigma
        pt_latent = pt_latent + pt_velocity * dt

        # Compare updated latents
        match, corr = compare_arrays(f"Updated latent", np.array(mlx_latent), pt_latent.numpy())

    # Final comparison
    print("\n" + "=" * 70)
    print("Final Results")
    print("=" * 70)

    final_mlx = unpatchify(np.array(mlx_latent), frames, height, width)
    final_pt = unpatchify(pt_latent.numpy(), frames, height, width)

    match, corr = compare_arrays("Final latent (5D)", final_mlx, final_pt)

    # Spatial autocorrelation
    mlx_frame0 = final_mlx[0, :, 0, :, :].mean(axis=0)
    pt_frame0 = final_pt[0, :, 0, :, :].mean(axis=0)

    mlx_autocorr = np.corrcoef(mlx_frame0[:-1, :].flatten(), mlx_frame0[1:, :].flatten())[0, 1]
    pt_autocorr = np.corrcoef(pt_frame0[:-1, :].flatten(), pt_frame0[1:, :].flatten())[0, 1]

    print(f"\n  MLX final spatial autocorr: {mlx_autocorr:.4f}")
    print(f"  PT final spatial autocorr: {pt_autocorr:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    if corr > 0.99:
        print("\n  Full denoising loop MATCHES between MLX and PyTorch!")
        print("  The implementation is correct end-to-end.")
    else:
        print(f"\n  WARNING: Final outputs differ (corr={corr:.4f})")


if __name__ == "__main__":
    main()
